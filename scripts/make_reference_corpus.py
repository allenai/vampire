### from https://github.com/dallascard/scholar/blob/master/preprocess_data.py
import os
import re
import string
import sys
from collections import Counter
from optparse import OptionParser

import numpy as np
from scipy import sparse
from tqdm import tqdm

from vampire.common.util import (read_jsonlist,
                                 read_text,
                                 save_sparse,
                                 write_to_json)

# compile some regexes
PUNCT_CHARS = list(set(string.punctuation) - set("'"))
PUNCT_CHARS.sort()
PUNCTUATION = ''.join(PUNCT_CHARS)
REPLACE = re.compile('[%s]' % re.escape(PUNCTUATION))
ALPHA = re.compile('^[a-zA-Z_]+$')
ALPHA_OR_NUM = re.compile('^[a-zA-Z_]+|[0-9_]+$')
ALPHA_OR_NUM = re.compile('^[a-zA-Z0-9_]+$')


def main(args):
    usage = "%prog train.jsonlist output_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--test', dest='test', default=None,
                      help='Test data (test.jsonlist): default=%default')
    parser.add_option('--train-prefix', dest='train_prefix', default='train',
                      help='Output prefix for training data: default=%default')
    parser.add_option('--test-prefix', dest='test_prefix', default='test',
                      help='Output prefix for test data: default=%default')
    parser.add_option('--stopwords', dest='stopwords', default='snowball',
                      help='List of stopwords to exclude [None|mallet|snowball]: default=%default')
    parser.add_option('--keep-num', action="store_true", dest="keep_num", default=False,
                      help='Keep tokens made of only numbers: default=%default')
    parser.add_option('--keep-alphanum', action="store_true", dest="keep_alphanum", default=False,
                      help="Keep tokens made of a mixture of letters and numbers: default=%default")
    parser.add_option('--strip-html', action="store_true", dest="strip_html", default=False,
                      help='Strip HTML tags: default=%default')
    parser.add_option('--no-lower', action="store_true", dest="no_lower", default=False,
                      help='Do not lowercase text: default=%default')
    parser.add_option('--min-length', dest='min_length', default=3,
                      help='Minimum token length: default=%default')
    parser.add_option('--vocab-size', dest='vocab_size', default=None,
                      help='Size of the vocabulary (by most common, following above exclusions): default=%default')
    parser.add_option('--seed', dest='seed', default=42,
                      help='Random integer seed (only relevant for choosing test set): default=%default')

    (options, args) = parser.parse_args(args)

    train_infile = args[0]
    output_dir = args[1]

    test_infile = options.test
    train_prefix = options.train_prefix
    test_prefix = options.test_prefix
    vocab_size = options.vocab_size
    stopwords = options.stopwords
    if stopwords == 'None':
        stopwords = None
    keep_num = options.keep_num
    keep_alphanum = options.keep_alphanum
    strip_html = options.strip_html
    lower = not options.no_lower
    min_length = int(options.min_length)
    seed = options.seed
    if seed is not None:
        np.random.seed(int(seed))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    preprocess_data(train_infile,
                    test_infile,
                    output_dir,
                    train_prefix,
                    test_prefix,
                    vocab_size,
                    stopwords,
                    keep_num,
                    keep_alphanum,
                    strip_html,
                    lower,
                    min_length)


def preprocess_data(train_infile,
                    test_infile,
                    output_dir,
                    train_prefix,
                    test_prefix,
                    vocab_size=None,
                    stopwords=None,
                    keep_num=False,
                    keep_alphanum=False,
                    strip_html=False,
                    lower=True,
                    min_length=3):

    if stopwords == 'mallet':
        print("Using Mallet stopwords")
        stopword_list = read_text(os.path.join('vampire', 'common', 'stopwords', 'mallet_stopwords.txt'))
    elif stopwords == 'snowball':
        print("Using snowball stopwords")
        stopword_list = read_text(os.path.join('vampire', 'common', 'stopwords', 'snowball_stopwords.txt'))
    elif stopwords is not None:
        print("Using custom stopwords")
        stopword_list = read_text(os.path.join('vampire', 'common', 'stopwords', stopwords + '_stopwords.txt'))
    else:
        stopword_list = []
    stopword_set = {s.strip() for s in stopword_list}

    print("Reading data files")
    train_items = read_jsonlist(train_infile)
    n_train = len(train_items)
    print("Found {:d} training documents".format(n_train))

    if test_infile is not None:
        test_items = read_jsonlist(test_infile)
        n_test = len(test_items)
        print("Found {:d} test documents".format(n_test))
    else:
        test_items = []
        n_test = 0

    all_items = train_items + test_items
    n_items = n_train + n_test

    # make vocabulary
    train_parsed = []
    test_parsed = []

    print("Parsing %d documents" % n_items)
    word_counts = Counter()
    doc_counts = Counter()
    count = 0

    vocab = None
    for i, item in tqdm(enumerate(all_items), total=n_items):
        if i % 1000 == 0 and count > 0:
            print(i)

        text = item['text']
        tokens, _ = tokenize(text,
                             strip_html=strip_html,
                             lower=lower,
                             keep_numbers=keep_num,
                             keep_alphanum=keep_alphanum,
                             min_length=min_length,
                             stopwords=stopword_set,
                             vocab=vocab)

        # store the parsed documents
        if i < n_train:
            train_parsed.append(tokens)
        else:
            test_parsed.append(tokens)

        # keep track fo the number of documents with each word
        word_counts.update(tokens)
        doc_counts.update(set(tokens))

    print("Size of full vocabulary=%d" % len(word_counts))

    print("Selecting the vocabulary")
    most_common = word_counts.most_common(n=vocab_size)
    words, word_counts = zip(*most_common)
    vocab = list(words)
    print("Vocab size after filtering = %d" % len(vocab))
    if vocab_size is not None:
        if len(vocab) > int(vocab_size):
            vocab = vocab[:int(vocab_size)]

    vocab_size = len(vocab)
    print("Final vocab size = %d" % vocab_size)

    print("Most common words remaining:", ' '.join(vocab[:10]))
    vocab.sort()

    write_to_json(vocab, os.path.join(output_dir,  'ref.vocab.json'))

    process_subset(train_items, train_parsed, vocab, output_dir, train_prefix)
    if n_test > 0:
        process_subset(test_items, test_parsed, vocab, output_dir, test_prefix)


def process_subset(items, parsed, vocab, output_dir, output_prefix):
    n_items = len(items)
    vocab_size = len(vocab)
    vocab_index = dict(zip(vocab, range(vocab_size)))

    X = np.zeros([n_items, vocab_size], dtype=int)

    counter = Counter()
    word_counter = Counter()
    print("Converting to count representations")
    for i, words in tqdm(enumerate(parsed), total=len(parsed)):
        # get the vocab indices of words that are in the vocabulary
        indices = [vocab_index[word] for word in words if word in vocab_index]
        word_subset = [word for word in words if word in vocab_index]

        counter.clear()
        counter.update(indices)
        word_counter.clear()
        word_counter.update(word_subset)

        if len(counter.keys()) > 0:
            # update the counts
            values = list(counter.values())
            X[np.ones(len(counter.keys()), dtype=int) * i, list(counter.keys())] += values

    # convert to a sparse representation
    sparse_X = sparse.csr_matrix(X)
    save_sparse(sparse_X, os.path.join(output_dir, 'ref.npz'))


def tokenize(text,
             strip_html=False,
             lower=True,
             keep_emails=False,
             keep_at_mentions=False,
             keep_numbers=False,
             keep_alphanum=False,
             min_length=3,
             stopwords=None,
             vocab=None):
    text = clean_text(text, strip_html, lower, keep_emails, keep_at_mentions)
    tokens = text.split()

    if stopwords is not None:
        tokens = ['_' if t in stopwords else t for t in tokens]

    # remove tokens that contain numbers
    if not keep_alphanum and not keep_numbers:
        tokens = [t if ALPHA.match(t) else '_' for t in tokens]

    # or just remove tokens that contain a combination of letters and numbers
    elif not keep_alphanum:
        tokens = [t if ALPHA_OR_NUM.match(t) else '_' for t in tokens]

    # drop short tokens
    if min_length > 0:
        tokens = [t if len(t) >= min_length else '_' for t in tokens]

    counts = Counter()

    unigrams = [t for t in tokens if t != '_']
    counts.update(unigrams)

    if vocab is not None:
        tokens = [token for token in unigrams if token in vocab]
    else:
        tokens = unigrams

    return tokens, counts


def clean_text(text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False):
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
    text = REPLACE.sub(' ', text)
    # remove single quotes
    text = re.sub(r'\'', '', text)
    # replace all whitespace with a single space
    text = re.sub(r'\s', ' ', text)
    # strip off spaces on either end
    text = text.strip()
    return text


if __name__ == '__main__':
    main(sys.argv[1:])
