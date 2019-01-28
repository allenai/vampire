import os
from ast import literal_eval
import argparse
import glob
import time
import numpy as np
from vae.common.util import load_sparse, read_json


def main():
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument('-r', '--reference_dir', type=str, help='path to directory containing reference counts and vocab', required=True)
    parser.add_argument('-s', '--serialization_dir', type=str, help='model serialization directory', required=True)
    args = parser.parse_args()
    prior = 0
    ref_counts_file, ref_vocab_file = get_files(args.reference_dir)
    ref_vocab = read_json(ref_vocab_file)
    ref_counts = load_sparse(ref_counts_file).tocsc()
    try:
        while True:
            time.sleep(10)
            num_files = len(glob.glob(os.path.join(args.serialization_dir, "topics", "*")))
            if num_files != prior:
                time.sleep(10)
                prior += (num_files - prior)
                topics = read_topics(os.path.join(args.serialization_dir, "topics"))
                mean_npmi = compute_npmi(topics, ref_vocab, ref_counts)
                print("epoch {}: {}".format(num_files, mean_npmi))
    except KeyboardInterrupt:
        print("Interrupted, exiting now.")


def get_files(directory):
    for file in os.listdir(directory):
        if "npz" in file:
            ref_counts_file = os.path.join(directory, file)
        elif "vocab.json" in file:
            ref_vocab_file = os.path.join(directory, file)
    return ref_counts_file, ref_vocab_file


def load_and_compute_npmi(topics, ref_vocab_file, ref_counts_file, cols_to_skip=0):
    print("Loading reference counts")
    ref_vocab = read_json(ref_vocab_file)
    ref_counts = load_sparse(ref_counts_file).tocsc()
    mean_npmi = compute_npmi(topics, ref_vocab, ref_counts, cols_to_skip)
    return mean_npmi


def compute_npmi(topics, ref_vocab, ref_counts, cols_to_skip=0):
    mean_npmi = compute_npmi_at_n(topics, ref_vocab, ref_counts, cols_to_skip=cols_to_skip)
    return mean_npmi


def read_topics(topic_dir):
    list_of_files = glob.glob(os.path.join(topic_dir, "*"))
    latest_file = max(list_of_files, key=os.path.getctime)
    with open(latest_file, 'r') as f:
        lines = f.readlines()
    topics = []
    for line in lines[3:]:
        topic_list = literal_eval(line.split("".join([' '] * 9))[1].strip())
        topics.append(" ".join(topic_list))
    return topics


def compute_npmi_at_n(topics, ref_vocab, ref_counts, n=10, cols_to_skip=0):

    vocab_index = dict(zip(ref_vocab, range(len(ref_vocab))))
    n_docs, _ = ref_counts.shape

    npmi_means = []
    for topic in topics:
        words = topic.split()[cols_to_skip:]
        npmi_vals = []
        for word_i, word1 in enumerate(words[:n]):
            if word1 in vocab_index:
                index1 = vocab_index[word1]
            else:
                index1 = None
            for word2 in words[word_i+1:n]:
                if word2 in vocab_index:
                    index2 = vocab_index[word2]
                else:
                    index2 = None
                if index1 is None or index2 is None:
                    npmi = 0.0
                else:
                    col1 = np.array(ref_counts[:, index1].todense() > 0, dtype=int)
                    col2 = np.array(ref_counts[:, index2].todense() > 0, dtype=int)
                    c1 = col1.sum()
                    c2 = col2.sum()
                    c12 = np.sum(col1 * col2)
                    if c12 == 0:
                        npmi = 0.0
                    else:
                        npmi = (np.log10(n_docs) + np.log10(c12) - np.log10(c1) - np.log10(c2)) / (np.log10(n_docs) - np.log10(c12))
                npmi_vals.append(npmi)
        npmi_means.append(np.mean(npmi_vals))
    return np.mean(npmi_means)


if __name__ == '__main__':
    main()
