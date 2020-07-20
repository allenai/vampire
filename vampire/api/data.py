import argparse
import json
import logging
import os
import sys
from typing import List

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
from allennlp.common.util import lazy_groups_of
from vampire.common.util import (generate_config, save_sparse,
                                 write_list_to_file, write_to_json)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

def load_data(data_path: str) -> List[str]:
    tokenized_examples = []
    with tqdm(open(data_path, "r"), desc=f"loading {data_path}") as f:
        for line in f:
            if data_path.endswith(".jsonl") or data_path.endswith(".json"):
                example = json.loads(line)
            else:
                example = {"text": line}
            text = example['text']
            tokenized_examples.append(text)
    return tokenized_examples

def transform_text(input_file: str,
                   vocabulary_path: str,
                   tfidf: bool,
                   serialization_dir: str,
                   shard: bool = False,
                   shard_size: int=100):
    tokenized_examples = load_data(input_file)
    with open(vocabulary_path, 'r') as f:
        vocabulary = f.readlines()
    if tfidf:
        count_vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    else:
        count_vectorizer = CountVectorizer(vocabulary=vocabulary)
    count_vectorizer.fit(tqdm(tokenized_examples))
    vectorized_examples = count_vectorizer.transform(tqdm(tokenized_examples))
    vectorized_examples = sparse.hstack((np.array([0] * len(tokenized_examples))[:,None], vectorized_examples))
    if shard:
        for ix, batch in tqdm(enumerate(lazy_groups_of(vectorized_examples, shard_size)), total=len(vectorized_examples) / shard_size):
            save_sparse(batch, os.path.join(serialization_dir, f"{ix}.npz"))
    else:
        save_sparse(batch, os.path.join(serialization_dir, "data.npz"))

def preprocess_data(train_path: str,
                    dev_path: str,
                    serialization_dir: str,
                    tfidf: bool,
                    vocab_size: int,
                    vocabulary_path: str=None,
                    reference_corpus_path: str=None) -> None:

    if not os.path.isdir(serialization_dir):
        os.mkdir(serialization_dir)

    vocabulary_dir = os.path.join(serialization_dir, "vocabulary")

    if not os.path.isdir(vocabulary_dir):
        os.mkdir(vocabulary_dir)

    tokenized_train_examples = load_data(train_path)
    tokenized_dev_examples = load_data(dev_path)

    logging.info("fitting count vectorizer...")
    if tfidf:
        count_vectorizer = TfidfVectorizer(stop_words='english', max_features=vocab_size, token_pattern=r'\b[^\d\W]{3,30}\b')
    else:
        count_vectorizer = CountVectorizer(stop_words='english', max_features=vocab_size, token_pattern=r'\b[^\d\W]{3,30}\b')
    
    text = tokenized_train_examples + tokenized_dev_examples
    
    count_vectorizer.fit(tqdm(text))

    vectorized_train_examples = count_vectorizer.transform(tqdm(tokenized_train_examples))
    vectorized_dev_examples = count_vectorizer.transform(tqdm(tokenized_dev_examples))

    if tfidf:
        reference_vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'\b[^\d\W]{3,30}\b')
    else:
        reference_vectorizer = CountVectorizer(stop_words='english', token_pattern=r'\b[^\d\W]{3,30}\b')
    if not reference_corpus_path:
        logging.info("fitting reference corpus using development data...")
        reference_matrix = reference_vectorizer.fit_transform(tqdm(tokenized_dev_examples))
    else:
        logging.info(f"loading reference corpus at {reference_corpus_path}...")
        reference_examples = load_data(reference_corpus_path)
        logging.info("fitting reference corpus...")
        reference_matrix = reference_vectorizer.fit_transform(tqdm(reference_examples))

    reference_vocabulary = reference_vectorizer.get_feature_names()

    # add @@unknown@@ token vector
    vectorized_train_examples = sparse.hstack((np.array([0] * len(tokenized_train_examples))[:,None], vectorized_train_examples))
    vectorized_dev_examples = sparse.hstack((np.array([0] * len(tokenized_dev_examples))[:,None], vectorized_dev_examples))
    master = sparse.vstack([vectorized_train_examples, vectorized_dev_examples])

    # generate background frequency
    logging.info("generating background frequency...")
    bgfreq = dict(zip(count_vectorizer.get_feature_names(), (np.array(master.sum(0)) / vocab_size).squeeze()))

    logging.info("saving data...")
    save_sparse(vectorized_train_examples, os.path.join(serialization_dir, "train.npz"))
    save_sparse(vectorized_dev_examples, os.path.join(serialization_dir, "dev.npz"))
    if not os.path.isdir(os.path.join(serialization_dir, "reference")):
        os.mkdir(os.path.join(serialization_dir, "reference"))
    save_sparse(reference_matrix, os.path.join(serialization_dir, "reference", "ref.npz"))
    write_to_json(reference_vocabulary, os.path.join(serialization_dir, "reference", "ref.vocab.json"))
    write_to_json(bgfreq, os.path.join(serialization_dir, "vampire.bgfreq"))
    
    write_list_to_file(['@@UNKNOWN@@'] + count_vectorizer.get_feature_names(), os.path.join(vocabulary_dir, "vampire.txt"))
    write_list_to_file(['*tags', '*labels', 'vampire'], os.path.join(vocabulary_dir, "non_padded_namespaces.txt"))
    return
