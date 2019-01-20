import os
import re
import sys
import string
from optparse import OptionParser
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import savemat
from tqdm import tqdm
import common.file_handling as fh
import argparse
from shutil import copyfile
from typing import List, Optional


def split_data(df: pd.DataFrame, num: int):
    other = df.sample(n=num)
    df = df.drop(other.index)
    df = df.reset_index(drop=True)
    other = other.reset_index(drop=True)
    return df, other


def run(data_dir: str, output_dir: str, subsamples: Optional[List[int]]=None, split_dev: Optional[int]=None, split_unlabeled: Optional[int]=None):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    full_dir = os.path.join(output_dir, "full")
    
    if not os.path.exists(full_dir):
        os.mkdir(full_dir)

    with open(os.path.join(data_dir, 'train.jsonl'), 'r') as f:
        train = pd.read_json(f, lines=True)
        orig_size = train.shape[0]

    if split_dev is not None:
        train, dev = split_data(train, split_dev)
        dev.to_json(os.path.join(full_dir, "dev_raw.jsonl"), lines=True, orient='records')
    else:
        copyfile(os.path.join(data_dir, "dev.jsonl"),
                 os.path.join(full_dir, "dev_raw.jsonl"))

    if split_unlabeled is not None:
        train, unlabeled = split_data(train, split_unlabeled)
        unlabeled["label"] = 0
        out_dir = os.path.join(output_dir, "unlabeled")
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        unlabeled.to_json(os.path.join(out_dir, "train_raw.jsonl"), lines=True, orient='records')
    else:
        out_dir = os.path.join(output_dir, "unlabeled")
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        copyfile(os.path.join(data_dir, "unlabeled.jsonl"),
                 os.path.join(out_dir, "train_raw.jsonl"))

    if train.shape[0] < orig_size:
        train.to_json(os.path.join(full_dir, "train_raw.jsonl"), lines=True, orient='records')
    else:
        copyfile(os.path.join(data_dir, "train.jsonl"),
                 os.path.join(output_dir, "full", "train_raw.jsonl"))

    copyfile(os.path.join(data_dir, "test.jsonl"),
             os.path.join(output_dir, "full", "test_raw.jsonl"))

    samples = {}
    for size in subsamples:
        sample = train.sample(n=size)
        samples[size] = sample

    for size, sample in samples.items():
        out_dir = os.path.join(output_dir, str(size))
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        sample.to_json(os.path.join(out_dir, "train_raw.jsonl"), lines=True, orient='records')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_dir', dest='data_dir', type=str, help='path to data directory', required=True)
    parser.add_argument('-o','--output_dir', type=str, help='output directory', required=True)
    parser.add_argument('-x','--split_dev', type=int, help='size of dev data', required=False)
    parser.add_argument('-s','--subsamples', nargs='+', type=int, help='subsample sizes', required=True)
    parser.add_argument('-u','--split_unlabeled', type=int, help='size of unlabeled data', required=False)

    args = parser.parse_args()

    run(**args)

    print("Done!")