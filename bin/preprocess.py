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
import logging
from typing import List, Set, Optional, Dict

logger = logging.getLogger(__name__)

# pre-compile some regexes
punct_chars = list(set(string.punctuation) - set("'"))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))
alpha = re.compile('^[a-zA-Z_]+$')
alpha_or_num = re.compile('^[a-zA-Z_]+|[0-9_]+$')
alphanum = re.compile('^[a-zA-Z0-9_]+$')


def generate_vocab(doc_counts: Counter,
                   n_items: int,
                   vocab_size: int,
                   max_doc_freq: int=1,
                   min_doc_count: int=0,
                   ):
    """
    Generate a vocabulary given a set of documents, using document counts.

    Parameters
    ----------

    doc_counts : ``Counter``
        number of documents that each token appears
    max_doc_freq: ``int``, optional (default = 1)
        maximum normalized document frequency
    min_doc_count: ``int``, optional (default = 0)
        minimum document count per frequency
    vocab_size: ``int``
        total number of tokens

    Returns
    -------

    vocab: ``List``
        List of tokens
    """
    most_common = doc_counts.most_common()
    words, doc_counts = zip(*most_common)
    doc_freqs = np.array(doc_counts) / float(n_items)
    vocab = [word for i, word in enumerate(words) if doc_counts[i] >= min_doc_count and doc_freqs[i] <= max_doc_freq]
    most_common = [word for i, word in enumerate(words) if doc_freqs[i] > max_doc_freq]
    if max_doc_freq < 1.0:
        tqdm.write("Excluding words with frequency > {:0.2f}:".format(max_doc_freq), most_common)

    tqdm.write("Vocab size after filtering = %d" % len(vocab))
    if vocab_size is not None:
        if len(vocab) > int(vocab_size):
            vocab = vocab[:int(vocab_size)]
    # convert to a sparse representation
    if "@@UNKNOWN@@" not in vocab:
        vocab.append("@@UNKNOWN@@")

    vocab_size = len(vocab)
    tqdm.write("Final vocab size = %d" % vocab_size)
    vocab.sort()
    return vocab


def process_subset(df: pd.DataFrame, label_field: str, unique_labels: List[str], vocab: List):
    """
    process a subset of data into output format
    """
    n_items = df.shape[0]
    vocab_size = len(vocab)
    vocab_index = dict(zip(vocab, range(vocab_size)))
    output = pd.DataFrame()

    # create a label index using string representations
    n_labels = len(unique_labels)
    label_list_strings = [str(label) for label in unique_labels]
    label_index = dict(zip(label_list_strings, range(n_labels)))

    df['parsed'] = df.parsed.apply(lambda x: [word for word in x if word in vocab_index])
    output['text'] = df.parsed.apply(lambda x: " ".join(x))
    output['label'] = df[label_field].apply(lambda x: label_index[str(x)])
    return df, output


def tokenize(text: str,
             strip_html: bool=False,
             lower: bool=True,
             keep_emails: bool=False,
             keep_at_mentions: bool=False,
             keep_numbers: bool=False,
             keep_alphanum: bool=False,
             min_length: int=3,
             stopwords: Optional[Set]=None):
    """
    clean and tokenize a piece of text

    Parameters
    ----------
    text : ``str``
        piece of text to tokenize
    strip_html: ``bool``, optional (default = ``False``)
        whether or not to strip html
    lower: ``bool``, optional (default = ``True``)
        whether or not to lowercase tokens
    keep_emails: ``bool``, optional (default = ``False``)
        whether or not to keep emails
    keep_at_mentions: ``bool``, optional (default = ``False``)
        whether or not to keep @ mentions
    keep_numbers: ``bool``, optional (default = ``False``)
        whether or not to keep numbers
    keep_alphanum: ``bool``, optional (default = ``False``)
        whether or not to keep alphanumeric characters
    min_length: ``int``, optional (default = ``3``)
        minimum length of tokens to keep
    stopwords: Set, optional (default  = ``None``)
        set of stopwords to filter with

    Returns
    -------

    unigrams: ``List[str]``
        cleaned, tokenized text

    """
    text = clean_text(text, strip_html, lower, keep_emails, keep_at_mentions)
    tokens = text.split()

    if stopwords is not None:
        tokens = ['_' if t in stopwords else t for t in tokens]

    # remove tokens that contain numbers
    if not keep_alphanum and not keep_numbers:
        tokens = [t if alpha.match(t) else '_' for t in tokens]

    # or just remove tokens that contain a combination of letters and numbers
    elif not keep_alphanum:
        tokens = [t if alpha_or_num.match(t) else '_' for t in tokens]

    # drop short tokens
    if min_length > 0:
        tokens = [t if len(t) >= min_length else '_' for t in tokens]

    unigrams = [t for t in tokens if t != '_']

    return unigrams


def clean_text(text: str,
               strip_html: bool=False,
               lower: bool=True,
               keep_emails: bool=False,
               keep_at_mentions: bool=False):
    """
    clean a piece of text

    Parameters
    ----------
    text : ``str``
        piece of text to tokenize
    strip_html: ``bool``, optional (default = ``False``)
        whether or not to strip html
    lower: ``bool``, optional (default = ``True``)
        whether or not to lowercase tokens
    keep_emails: ``bool``, optional (default = ``False``)
        whether or not to keep emails
    keep_at_mentions: ``bool``, optional (default = ``False``)
        whether or not to keep @ mentions

    Returns
    -------
    text: ``str``
        cleaned text
    """
    # remove html tags
    if strip_html:
        text = re.sub(r'<[^>]+>', '', text)
    else:
        # replace angle brackets
        text = re.sub(r'<', '(', text)
        text = re.sub(r'>', ')', text)
    # lower case
    if lower:
        text = text.lower()
    # eliminate email addresses
    if not keep_emails:
        text = re.sub(r'\S+@\S+', ' ', text)
    # eliminate @mentions
    if not keep_at_mentions:
        text = re.sub(r'\s@\S+', ' ', text)
    # replace underscores with spaces
    text = re.sub(r'_', ' ', text)
    # break off single quotes at the ends of words
    text = re.sub(r'\s\'', ' ', text)
    text = re.sub(r'\'\s', ' ', text)
    # remove periods
    text = re.sub(r'\.', '', text)
    # replace all other punctuation (except single quotes) with spaces
    text = replace.sub(' ', text)
    # remove single quotes
    text = re.sub(r'\'', '', text)
    # replace all whitespace with a single space
    text = re.sub(r'\s', ' ', text)
    # strip off spaces on either end
    text = text.strip()
    return text


def generate_bg_frequency(parsed_text: pd.Series):
    """
    generate a background frequency of tokens in parsed text

    Parameters
    ----------
    parsed_text, ``pd.Series``
        pandas Series of parsed text

    Returns
    -------
    freqs, ``Dict``
        normalized background frequencies
    """
    word_counts: Counter = Counter()
    parsed_text.apply(lambda x: word_counts.update(x))
    total = np.sum([c for k, c in word_counts.items()])
    freqs = {k: c / float(total) for k, c in word_counts.items()}
    return freqs


def write_to_files(output_dir,
                   train_output,
                   vocab,
                   freqs,
                   unique_labels,
                   test_output=None,
                   dev_output=None,
                   train_prefix="train",
                   test_prefix="test",
                   dev_prefix="dev"):
    """
    write items to file
    """
    fh.write_to_json(freqs, os.path.join(output_dir, train_prefix + '.bgfreq.json'))
    train_output.to_json(os.path.join(output_dir, train_prefix + '.jsonl'), lines=True, orient='records')
    train_output['text'].to_csv(os.path.join(output_dir, train_prefix + '.txt'), header=False, index=None)
    if test_output is not None:
        test_output.to_json(os.path.join(output_dir, test_prefix + '.jsonl'), lines=True, orient='records')
        test_output['text'].to_csv(os.path.join(output_dir, test_prefix + '.txt'), header=False, index=None)
    if dev_output is not None:
        dev_output.to_json(os.path.join(output_dir, dev_prefix + '.jsonl'), lines=True, orient='records')
        dev_output['text'].to_csv(os.path.join(output_dir, dev_prefix + '.txt'), header=False, index=None)

    # write AllenNLP vocabulary files
    if not os.path.isdir(os.path.join(output_dir, 'vocabulary')):
        os.mkdir(os.path.join(output_dir, 'vocabulary'))
    fh.write_list_to_text(vocab, os.path.join(output_dir, 'vocabulary', 'tokens.txt'))
    fh.write_list_to_text(list(train_output.label.unique().astype(str)), os.path.join(output_dir, 'vocabulary', 'labels.txt'))
    fh.write_list_to_text(["0", "1"], os.path.join(output_dir, 'vocabulary', 'is_labeled.txt'))
    fh.write_list_to_text(["tokens", "labels", "is_labeled"], os.path.join(output_dir, 'vocabulary', 'non_padded_namespaces.txt'))


def run(train_infile: str,
        test_infile: str,
        dev_infile: str,
        output_dir: str,
        train_prefix: str,
        test_prefix: str,
        dev_prefix: str,
        min_doc_count: int=0,
        max_doc_freq: int=1,
        vocab_size: int=None,
        sample: int=None,
        stopwords: str=None,
        keep_num: bool=False,
        keep_alphanum: bool=False,
        strip_html: bool=False,
        lower: bool=True,
        min_length: int=3,
        label_field: str=None):

    if stopwords == 'mallet':
        tqdm.write("Using Mallet stopwords")
        stopword_list = fh.read_text("/home/ubuntu/vae/" + os.path.join('common', 'stopwords', 'mallet_stopwords.txt'))
    elif stopwords == 'snowball':
        tqdm.write("Using snowball stopwords")
        stopword_list = fh.read_text("/home/ubuntu/vae/" + os.path.join('common', 'stopwords', 'snowball_stopwords.txt'))
    elif stopwords is not None:
        tqdm.write("Using custom stopwords")
        stopword_list = fh.read_text("/home/ubuntu/vae/" + os.path.join('common', 'stopwords', stopwords + '_stopwords.txt'))
    else:
        stopword_list = []

    stopword_set = {s.strip() for s in stopword_list}

    tqdm.write("Reading data files")

    train = pd.read_json(train_infile, lines=True)
    n_train = train.shape[0]

    tqdm.write("Found {:d} training documents".format(n_train))

    all_items_dict = {"train": train}

    if test_infile is not None:
        test = pd.read_json(test_infile, lines=True)
        n_test = test.shape[0]
        all_items_dict['test'] = test
        tqdm.write("Found {:d} test documents".format(n_test))
    else:
        n_test = 0

    if dev_infile is not None:
        dev = pd.read_json(dev_infile, lines=True)
        n_dev = dev.shape[0]
        all_items_dict['dev'] = dev
        tqdm.write("Found {:d} dev documents".format(n_dev))
    else:
        n_dev = 0

    n_items = n_train + n_test + n_dev

    unique_labels = train[label_field].unique()

    n_labels = len(unique_labels)
    tqdm.write("Found label %s with %d classes" % (label_field, n_labels))

    tqdm.write("Parsing %d documents" % n_items)

    doc_counts: Counter = Counter()

    pbar = tqdm(all_items_dict.items())

    for data_name, df in pbar:
        pbar.set_description(data_name)
        tqdm.pandas()
        df['parsed'] = df.text.progress_apply(lambda x: tokenize(text=x,
                                                                 strip_html=strip_html,
                                                                 lower=lower,
                                                                 keep_numbers=keep_num,
                                                                 keep_alphanum=keep_alphanum,
                                                                 min_length=min_length,
                                                                 stopwords=stopword_set))
        # keep track of the number of documents with each word
        df.parsed.apply(lambda x: doc_counts.update(set(x)))

    vocab = generate_vocab(doc_counts=doc_counts,
                           n_items=n_items,
                           vocab_size=vocab_size,
                           max_doc_freq=max_doc_freq,
                           min_doc_count=min_doc_count)

    train, train_output = process_subset(df=train, label_field=label_field, unique_labels=unique_labels, vocab=vocab)

    if n_test > 0:
        test, test_output = process_subset(df=test, label_field=label_field, unique_labels=unique_labels, vocab=vocab)
    else:
        test_output = None

    if n_dev > 0:
        dev, dev_output = process_subset(df=dev, label_field=label_field, unique_labels=unique_labels, vocab=vocab)
    else:
        dev_output = None

    # generate background frequency from training data
    freqs = generate_bg_frequency(train.parsed)

    # write output
    write_to_files(output_dir, train_output, vocab, freqs, unique_labels, test_output, dev_output, train_prefix, test_prefix, dev_prefix)
    tqdm.write("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_dir', dest='data_dir', type=str, help='path to data directory', required=True)
    parser.add_argument('-l', '--label', type=str, help='label name', required=False, default='label')
    parser.add_argument('-v', '--vocab_size', type=int, help='vocab size', required=True)
    parser.add_argument('-s', '--stopwords', type=str, help='stopword type', required=False)
    parser.add_argument('-x', '--sample', type=int, help='sample data further', required=False)
    parser.add_argument('-m', "--min_doc_count", type=int, help="minimum document-level frequency of tokens", required=False, default=0)
    parser.add_argument('-r', "--min_doc_freq", type=int, help="minimum document frequency", required=False, default=1.0)
    parser.add_argument('-k', "--keep_num", action="store_true")
    parser.add_argument('-a', "--keep_alphanum", action="store_true")
    parser.add_argument('-q', "--strip_html", action="store_true")
    parser.add_argument('-p', "--lower", action="store_true")
    parser.add_argument('-n', "--min_length", type=int, help="minimum token length", required=False, default=3)

    args = parser.parse_args()

    train_infile = os.path.join(args.data_dir, "train_raw.jsonl")

    if os.path.exists(os.path.join(args.data_dir, "dev_raw.jsonl")):
        dev_infile = os.path.join(args.data_dir, "dev_raw.jsonl")
    else:
        dev_infile = None

    if os.path.exists(os.path.join(args.data_dir, "test_raw.jsonl")):
        test_infile = os.path.join(args.data_dir, "test_raw.jsonl")
    else:
        test_infile = None

    run(train_infile=train_infile,
        test_infile=test_infile,
        dev_infile=dev_infile,
        output_dir=args.data_dir,
        train_prefix="train",
        test_prefix="test",
        dev_prefix="dev",
        min_doc_count=args.min_doc_count,
        max_doc_freq=args.min_doc_freq,
        vocab_size=args.vocab_size,
        sample=args.sample,
        stopwords=args.stopwords,
        keep_num=args.keep_num,
        keep_alphanum=args.keep_alphanum,
        strip_html=args.strip_html,
        lower=args.lower,
        min_length=args.min_length,
        label_field=args.label)

    fh.write_to_json(args.__dict__, os.path.join(args.data_dir, "preprocessing.json"))
    tqdm.write("Done!")
