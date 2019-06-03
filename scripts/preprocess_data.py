import argparse
import json
import os
from typing import List

import nltk
import numpy as np
import pandas as pd
import spacy
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from spacy.tokenizer import Tokenizer
from tqdm import tqdm

from vampire.common.util import read_text, save_sparse, write_to_json


def load_data(data_path: str, tokenize: bool = False, tokenizer_type: str = "just_spaces") -> List[str]:
    if tokenizer_type == "just_spaces":
        tokenizer = SpacyWordSplitter()
    elif tokenizer_type == "spacy":
        nlp = spacy.load('en')
        tokenizer = Tokenizer(nlp.vocab)

    tokenized_examples = []
    with tqdm(open(data_path, "r"), desc=f"loading {data_path}") as f:
        for line in f:
            example = json.loads(line)
            if tokenize:
                if tokenizer_type == 'just_spaces':
                    tokens = list(map(str, tokenizer.split_words(example['text'])))
                elif tokenizer_type == 'spacy':
                    tokens = list(map(str, tokenizer(example['text'])))
                text = ' '.join(tokens)
            else:
                text = example['text']
            tokenized_examples.append(text)
    return tokenized_examples

def main():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-path", type=str, required=True,
                        help="Path to the train jsonl file.")
    parser.add_argument("--dev-path", type=str, required=True,
                        help="Path to the dev jsonl file.")
    parser.add_argument("--serialization-dir", "-s", type=str, required=True,
                        help="Path to store the preprocessed output.")
    parser.add_argument("--vocab-size", type=int, required=False, default=10000,
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--tokenize", action='store_true',
                        help="Path to store the preprocessed corpus vocabulary (output file name).") 
    parser.add_argument("--tokenizer-type", type=str, default="just_spaces",
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    args = parser.parse_args()

    if not os.path.isdir(args.serialization_dir):
        os.mkdir(args.serialization_dir)
    
    vocabulary_dir = os.path.join(args.serialization_dir, "vocabulary")

    if not os.path.isdir(vocabulary_dir):
        os.mkdir(vocabulary_dir)

    tokenized_train_examples = load_data(args.train_path, args.tokenize, args.tokenizer_type)
    tokenized_dev_examples = load_data(args.dev_path, args.tokenize, args.tokenizer_type)

    print("fitting count vectorizer...")

    count_vectorizer = CountVectorizer(stop_words='english', max_features=args.vocab_size, token_pattern=r'\b[^\d\W]{3,30}\b')
    
    text = tokenized_train_examples + tokenized_dev_examples
    
    count_vectorizer.fit(text)

    vectorized_train_examples = count_vectorizer.transform(tokenized_train_examples)
    vectorized_dev_examples = count_vectorizer.transform(tokenized_dev_examples)

    dev_count_vectorizer = CountVectorizer(stop_words='english', max_features=args.vocab_size, token_pattern=r'\b[^\d\W]{3,30}\b')
    reference_matrix = dev_count_vectorizer.fit_transform(tokenized_dev_examples)
    reference_vocabulary = dev_count_vectorizer.get_feature_names()

    # add @@unknown@@ token vector
    vectorized_train_examples = sparse.hstack((np.array([0] * len(tokenized_train_examples))[:,None], vectorized_train_examples))
    vectorized_dev_examples = sparse.hstack((np.array([0] * len(tokenized_dev_examples))[:,None], vectorized_dev_examples))
    master = sparse.vstack([vectorized_train_examples, vectorized_dev_examples])

    # generate background frequency
    print("generating background frequency...")
    bgfreq = dict(zip(count_vectorizer.get_feature_names(), master.toarray().sum(1) / args.vocab_size))

    print("saving data...")
    save_sparse(vectorized_train_examples, os.path.join(args.serialization_dir, "train.npz"))
    save_sparse(vectorized_dev_examples, os.path.join(args.serialization_dir, "dev.npz"))

    write_to_json(bgfreq, os.path.join(args.serialization_dir, "vampire.bgfreq"))
    
    write_list_to_file(['@@UNKNOWN@@'] + count_vectorizer.get_feature_names(), os.path.join(vocabulary_dir, "vampire.txt"))
    write_list_to_file(['*tags', '*labels', 'vampire'], os.path.join(vocabulary_dir, "non_padded_namespaces.txt"))

def write_list_to_file(ls, save_path):
    """
    Write each json object in 'jsons' as its own line in the file designated by 'save_path'.
    """
    # Open in appendation mode given that this function may be called multiple
    # times on the same file (positive and negative sentiment are in separate
    # directories).
    out_file = open(save_path, "w+")
    for example in ls:
        out_file.write(example)
        out_file.write('\n')

if __name__ == '__main__':
    main()
