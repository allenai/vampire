import argparse
import json
import os
import random
import re
import sys
from collections import Counter

import numpy as np
from tqdm import tqdm

from common import file_handling as fh
from bin.preprocess_data import replace, tokenize


def main():
    """
    Given a dataset in jsonl format containing `text` fields, append a new
    field, `stopless`, which is a filtered version of the text with no stop words.

    Cleans both `text` and `filtered`.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(  # Escape out into project directory.
            os.path.dirname( # Escape out into scripts directory.
                os.path.realpath(__file__))))))
    parser.add_argument("--data-path", type=str,
                        help="Path to the dataset directory.")
    parser.add_argument("--save-path", type=str,
                        default=project_root,
                        help="Directory to store the preprocessed corpus.")
    parser.add_argument("--stopwords-path", type=str,
                        default="common/stopwords/snowball_stopwords.txt",
                        help="Path to the stopwords used.")
    parser.add_argument("--produce-background", type=bool, default=True,
                        help="If true, yields a background frequency for the dataset.")
    parser.add_argument("--seed", type=int,
                        default=1337,
                        help="Random seed to use when shuffling data.")
    args = parser.parse_args()


    data_file = open(args.data_path, "r")
    out_file = open(args.save_path, "w")

    stopword_list = fh.read_text(args.stopwords_path)
    stopword_set = {s.strip() for s in stopword_list}

    full_word_counts = Counter()
    filtered_word_counts = Counter()
    for line in tqdm(data_file):
        line = line.strip("\n")
        if not line:
            continue
        example = json.loads(line)
        clean_text, _ = tokenize(example['tokens'], keep_numbers=True, keep_alphanum=True, min_length=1)

        # There are lot of break tags in IMDB that don't get caught
        clean_text = list(filter(lambda x: x != 'br', clean_text))
        clean_stopless, _ = tokenize(example['tokens'], stopwords=stopword_set)
        example['cleaned'] = ' '.join(clean_text)
        example['stopless'] = ' '.join(clean_stopless) if clean_stopless else ""

        filtered_word_counts.update(clean_stopless)
        full_word_counts.update(clean_text)

        json.dump(example, out_file, ensure_ascii=False)
        out_file.write('\n')

    total = np.sum([c for k, c in full_word_counts.items()])
    full_freqs = {k: c / float(total) for k, c in full_word_counts.items()}
    total = np.sum([c for k, c in filtered_word_counts.items()])
    filtered_freqs = {k: c / float(total) for k, c in filtered_word_counts.items()}
    bgfreq_out = os.path.basename(args.data_path).split('.')[0] + ".bgfreq.filtered.json"
    fh.write_to_json(filtered_freqs, os.path.join(os.path.dirname(args.data_path), bgfreq_out))
    bgfreq_out = os.path.basename(args.data_path).split('.')[0] + ".bgfreq.json"
    fh.write_to_json(full_freqs, os.path.join(os.path.dirname(args.data_path), bgfreq_out))

if __name__ == "__main__":
    main()
