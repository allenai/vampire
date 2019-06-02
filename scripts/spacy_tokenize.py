

import argparse
import json
from scipy import sparse
import pandas as pd
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from tqdm import tqdm
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from vampire.common.util import save_sparse, write_to_json, read_text

def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-path", type=str, required=True,
                        help="Path to the IMDB jsonl file.")
    parser.add_argument("--dev-path", type=str, required=True,
                        help="Path to the IMDB jsonl file.")
    parser.add_argument("--ngram-range", dest="ngram_range", type=list, default=[1], required=False,
                        help="ngram range length")
    parser.add_argument("--save-vocab", type=str, required=True,
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--save-bgfreq", type=str, required=True,
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--save-sparse-train", type=str, required=False,
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--save-sparse-dev", type=str, required=False,
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--predefined_vocab", type=str, required=False,
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--vocab_size", type=int, required=False,
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--vocab_json", action='store_true', 
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--tokenize", action='store_true',
                        help="Path to store the preprocessed corpus vocabulary (output file name).") 
    args = parser.parse_args()

    tokenizer = SpacyWordSplitter()
    tokenized_train_examples = []
    tokenized_dev_examples = []

    with tqdm(open(args.train_path, "r")) as f:
        for line in f:
            example = json.loads(line)
            if args.tokenize:
                tokens = list(map(str, tokenizer.split_words(example['text'])))
                token_ngrams = []
                for ngram in args.ngram_range:
                    token_ngram = ["_".join(x) for x in list(nltk.ngrams(tokens, ngram))]
                    token_ngrams.append(token_ngram)
                example['text'] = ' '.join(tokens)
            tokenized_train_examples.append(example)
    
    with tqdm(open(args.dev_path, "r")) as f:
        for line in f:
            example = json.loads(line)
            if args.tokenize:
                tokens = list(map(str, tokenizer.split_words(example['text'])))
                token_ngrams = []
                for ngram in args.ngram_range:
                    token_ngram = ["_".join(x) for x in list(nltk.ngrams(tokens, ngram))]
                    token_ngrams.append(token_ngram)
                example['text'] = ' '.join(tokens)
            tokenized_dev_examples.append(example)

    print("fitting count vectorizer...")

    if args.predefined_vocab:
        predefined_vocab = read_text(args.predefined_vocab)
    else:
        predefined_vocab = None

    count_vectorizer = CountVectorizer(stop_words='english', max_features=args.vocab_size, vocabulary=predefined_vocab, token_pattern=r'\b[^\d\W]{3,30}\b')
    count_vectorizer.fit([tokenized_example['text'] for tokenized_example in tokenized_train_examples] + [tokenized_example['text'] for tokenized_example in tokenized_dev_examples])

    vectorized_train_examples = count_vectorizer.transform([tokenized_example['text'] for tokenized_example in tokenized_train_examples])
    vectorized_dev_examples = count_vectorizer.transform([tokenized_example['text'] for tokenized_example in tokenized_dev_examples])

    vectorized_train_examples = sparse.hstack((np.array([0] * len(tokenized_train_examples))[:,None], vectorized_train_examples))
    vectorized_dev_examples = sparse.hstack((np.array([0] * len(tokenized_dev_examples))[:,None], vectorized_dev_examples))
    master = sparse.vstack([vectorized_train_examples, vectorized_dev_examples])
    print("generating background frequency...")
    bgfreq = dict(zip(count_vectorizer.get_feature_names(), master.toarray().sum(1) / args.vocab_size))
    bgfreq["@@UNKNOWN@@"] = 0

    print("saving vectorized data...")
    save_sparse(vectorized_train_examples, args.save_sparse_train)
    save_sparse(vectorized_dev_examples, args.save_sparse_dev)

    
    print("saving...")

    write_to_json(bgfreq, args.save_bgfreq)

    if args.vocab_json:
        write_to_json(['@@UNKNOWN@@'] + count_vectorizer.get_feature_names(), args.save_vocab)
    else:
        write_list_to_file(['@@UNKNOWN@@'] + count_vectorizer.get_feature_names(), args.save_vocab)

def write_list_to_file(ls, save_path):
    """
    Write each json object in 'jsons' as its own line in the file designated by 'save_path'.
    """
    # Open in appendation mode given that this function may be called multiple
    # times on the same file (positive and negative sentiment are in separate
    # directories).
    out_file = open(save_path, "w+")
    for example in tqdm(ls):
        out_file.write(example)
        out_file.write('\n')


if __name__ == '__main__':
    main()
