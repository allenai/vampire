import argparse
import json
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
                tokens = list(map(str, tokenizer.split_words(example['text'])))
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
    parser.add_argument("--save-vocab", type=str, required=True,
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--save-bgfreq", type=str, required=True,
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--save-sparse-train", type=str, required=False,
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--save-sparse-dev", type=str, required=False,
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--save-ref-matrix", type=str, required=False,
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--save-ref-vocab", type=str, required=False,
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--vocab_size", type=int, required=False,
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--vocab_json", action='store_true', 
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--tokenize", action='store_true',
                        help="Path to store the preprocessed corpus vocabulary (output file name).") 
    parser.add_argument("--tokenizer_type", type=str, default="just_spaces",
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    args = parser.parse_args()

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
    save_sparse(vectorized_train_examples, args.save_sparse_train)
    save_sparse(vectorized_dev_examples, args.save_sparse_dev)
    save_sparse(reference_matrix, args.save_ref_matrix)

    write_to_json(bgfreq, args.save_bgfreq)
    
    write_list_to_file(['@@UNKNOWN@@'] + count_vectorizer.get_feature_names(), args.save_vocab)
    write_list_to_file(reference_vocabulary, args.save_ref_vocab)

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
