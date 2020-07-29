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
from numpy.lib.format import open_memmap
from pathlib import Path


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

def load_data(data_path: Path) -> (List[str], List[int]):
    tokenized_examples = []
    indices = []
    is_json = data_path.suffix in [".jsonl" , ".json"]
    with tqdm(open(data_path, "r"), desc=f"loading {data_path}") as f:
        for ix, line in enumerate(f):
            if is_json:
                example = json.loads(line)
            else:
                example = {"text": line}
            text = example['text']
            if 'index' not in example.keys():
                example['index'] = ix
            indices.append(example['index'])
            tokenized_examples.append(text)
    return tokenized_examples, indices


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        # Iterating over the rows this way is significantly more efficient
        # than csr_matrix[row_index,:] and csr_matrix.getrow(row_index)
        for row_start, row_end in tqdm(zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:])):
             data.append(csr_matrix.data[row_start:row_end])
             indices.append(csr_matrix.indices[row_start:row_end])
             indptr.append(row_end-row_start) # nnz of the row

        self.data = np.array(data)
        self.indices = np.array(indices)
        self.indptr = np.array(indptr)
        self.n_columns = csr_matrix.shape[1]

    def __getitem__(self, row_selector):
        data = np.concatenate(self.data[row_selector])
        indices = np.concatenate(self.indices[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))

        shape = [indptr.shape[0]-1, self.n_columns]

        return sparse.csr_matrix((data, indices, indptr), shape=shape)
        

def transform_text(input_file: Path,
                   vocabulary_path: Path,
                   tfidf: bool,
                   serialization_dir: Path,
                   shard: bool = False,
                   num_shards: int=64):
    tokenized_examples, indices = load_data(input_file)
    indices = np.array(indices)
    if not os.path.exists(serialization_dir):
        os.mkdir(serialization_dir) 
    with open(vocabulary_path, 'r') as f:
        vocabulary = [x.strip() for x in f.readlines()]
    if tfidf:
        count_vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    else:
        count_vectorizer = CountVectorizer(vocabulary=vocabulary)
    vectorized_examples = count_vectorizer.fit_transform(tqdm(tokenized_examples))

    # optionally sample the matrix
    if shard:
        iteration_indices = list(range(vectorized_examples.shape[0]))
        vectorized_examples = vectorized_examples.tocsr()
        row_indexer = SparseRowIndexer(vectorized_examples)
        shard_size = len(iteration_indices) // num_shards
        iteration_indices_batches = batch(iteration_indices, n=shard_size)
        for ix, index_batch in tqdm(enumerate(iteration_indices_batches),
                                    total=len(indices) // shard_size):
            rows = row_indexer[index_batch]
            indices_ = indices[index_batch]
            np.savez_compressed( serialization_dir / f"{ix}.npz",
                                ids=np.array(indices_),
                                emb=rows)
    else:
        np.savez_compressed(serialization_dir / f"0.npz",
                            ids=np.array(indices),
                            emb=vectorized_examples)

def preprocess_data(train_path: Path,
                    dev_path: Path,
                    serialization_dir: Path,
                    tfidf: bool,
                    vocab_size: int,
                    vocabulary_path: Path=None,
                    reference_corpus_path: Path=None) -> None:

    if not os.path.isdir(serialization_dir):
        os.mkdir(serialization_dir)

    vocabulary_dir = serialization_dir / "vocabulary"

    if not vocabulary_dir.exists():
        vocabulary_dir.mkdir()

    tokenized_train_examples, train_indices = load_data(train_path)
    tokenized_dev_examples, dev_indices = load_data(dev_path)

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
    save_sparse(vectorized_train_examples, serialization_dir / "train.npz")
    save_sparse(vectorized_dev_examples, serialization_dir / "dev.npz")
    if not (serialization_dir / "reference").exists():
        (serialization_dir / "reference").mkdir()
    save_sparse(reference_matrix, serialization_dir / "reference" / "ref.npz")
    write_to_json(reference_vocabulary, serialization_dir / "reference" / "ref.vocab.json")
    write_to_json(bgfreq, serialization_dir / "vampire.bgfreq")
    
    write_list_to_file(['@@UNKNOWN@@'] + count_vectorizer.get_feature_names(), vocabulary_dir / "vampire.txt")
    write_list_to_file(['*tags', '*labels', 'vampire'], vocabulary_dir / "non_padded_namespaces.txt")
    return
