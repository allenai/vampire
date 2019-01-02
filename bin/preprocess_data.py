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

import common.file_handling as fh

"""
Convert a dataset into the required format (as well as formats required by other tools).
Input format is one line per item.
Each line should be a json object.
At a minimum, each json object should have a "text" field, with the document text.
Any other field can be used as a label (specified with the --label option).
If training and test data are to be processed separately, the same input directory should be used
Run "python preprocess_data -h" for more options.
If an 'id' field is provided, this will be used as an identifier in the dataframes, otherwise index will be used 
"""

# compile some regexes
punct_chars = list(set(string.punctuation) - set("'"))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))
alpha = re.compile('^[a-zA-Z_]+$')
alpha_or_num = re.compile('^[a-zA-Z_]+|[0-9_]+$')
alphanum = re.compile('^[a-zA-Z0-9_]+$')


def main(args):
    usage = "%prog train.jsonlist output_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--label', dest='label', default=None,
                      help='field(s) to use as label (comma-separated): default=%default')
    parser.add_option('--test', dest='test', default=None,
                      help='Test data (test.jsonlist): default=%default')
    parser.add_option('--train-prefix', dest='train_prefix', default='train',
                      help='Output prefix for training data: default=%default')
    parser.add_option('--test-prefix', dest='test_prefix', default='test',
                      help='Output prefix for test data: default=%default')
    parser.add_option('--stopwords', dest='stopwords', default='snowball',
                      help='List of stopwords to exclude [None|mallet|snowball]: default=%default')
    parser.add_option('--min-doc-count', dest='min_doc_count', default=0,
                      help='Exclude words that occur in less than this number of documents')
    parser.add_option('--max-doc-freq', dest='max_doc_freq', default=1.0,
                      help='Exclude words that occur in more than this proportion of documents')
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
    label_fields = options.label
    min_doc_count = int(options.min_doc_count)
    max_doc_freq = float(options.max_doc_freq)
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

    preprocess_data(train_infile, test_infile, output_dir, train_prefix, test_prefix, min_doc_count, max_doc_freq, vocab_size, stopwords, keep_num, keep_alphanum, strip_html, lower, min_length, label_fields=label_fields)


def preprocess_data(train_infile, test_infile, output_dir, train_prefix, test_prefix, min_doc_count=0, max_doc_freq=1.0, vocab_size=None, stopwords=None, keep_num=False, keep_alphanum=False, strip_html=False, lower=True, min_length=3, label_fields=None):

    if stopwords == 'mallet':
        print("Using Mallet stopwords")
        stopword_list = fh.read_text("/home/ubuntu/vae/" + os.path.join('common', 'stopwords', 'mallet_stopwords.txt'))
    elif stopwords == 'snowball':
        print("Using snowball stopwords")
        stopword_list = fh.read_text("/home/ubuntu/vae/" + os.path.join('common', 'stopwords', 'snowball_stopwords.txt'))
    elif stopwords is not None:
        print("Using custom stopwords")
        stopword_list = fh.read_text("/home/ubuntu/vae/" + os.path.join('common', 'stopwords', stopwords + '_stopwords.txt'))
    else:
        stopword_list = []
    stopword_set = {s.strip() for s in stopword_list}

    print("Reading data files")
    train_items = fh.read_jsonlist(train_infile)
    n_train = len(train_items)
    print("Found {:d} training documents".format(n_train))

    if test_infile is not None:
        test_items = fh.read_jsonlist(test_infile)
        n_test = len(test_items)
        print("Found {:d} test documents".format(n_test))
    else:
        test_items = []
        n_test = 0

    all_items = train_items + test_items
    n_items = n_train + n_test

    label_lists = {}
    if label_fields is not None:
        if ',' in label_fields:
            label_fields = label_fields.split(',')
        else:
            label_fields = [label_fields]
        for label_name in label_fields:
            label_set = set()
            for i, item in enumerate(all_items):
                if label_name is not None:
                    label_set.add(str(item[label_name]))
            label_list = list(label_set)
            label_list.sort()
            n_labels = len(label_list)
            print("Found label %s with %d classes" % (label_name, n_labels))
            label_lists[label_name] = label_list
    else:
        label_fields = []

    # make vocabulary
    train_parsed = []
    test_parsed = []

    print("Parsing %d documents" % n_items)
    word_counts = Counter()
    doc_counts = Counter()
    count = 0

    vocab = None
    for i, item in enumerate(all_items):
        if i % 1000 == 0 and count > 0:
            print(i)

        text = item['tokens']
        tokens, _ = tokenize(text, strip_html=strip_html, lower=lower, keep_numbers=keep_num, keep_alphanum=keep_alphanum, min_length=min_length, stopwords=stopword_set, vocab=vocab)

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
    most_common = doc_counts.most_common()
    words, doc_counts = zip(*most_common)
    doc_freqs = np.array(doc_counts) / float(n_items)
    vocab = [word for i, word in enumerate(words) if doc_counts[i] >= min_doc_count and doc_freqs[i] <= max_doc_freq]
    most_common = [word for i, word in enumerate(words) if doc_freqs[i] > max_doc_freq]
    if max_doc_freq < 1.0:
        print("Excluding words with frequency > {:0.2f}:".format(max_doc_freq), most_common)

    print("Vocab size after filtering = %d" % len(vocab))
    if vocab_size is not None:
        if len(vocab) > int(vocab_size):
            vocab = vocab[:int(vocab_size)]

    vocab_size = len(vocab)
    print("Final vocab size = %d" % vocab_size)

    print("Most common words remaining:", ' '.join(vocab[:10]))
    vocab.sort()

    fh.write_jsonlist(vocab, os.path.join(output_dir, train_prefix + '.vocab.json'))

    train_X_sage, tr_aspect, tr_no_aspect, tr_widx, vocab_for_sage = process_subset(train_items, train_parsed, label_fields, label_lists, vocab, output_dir, train_prefix)
    if n_test > 0:
        test_X_sage, te_aspect, te_no_aspect, _, _= process_subset(test_items, test_parsed, label_fields, label_lists, vocab, output_dir, test_prefix)

    train_sum = np.array(train_X_sage.sum(axis=0))
    print("%d words missing from training data" % np.sum(train_sum == 0))

    if n_test > 0:
        test_sum = np.array(test_X_sage.sum(axis=0))
        print("%d words missing from test data" % np.sum(test_sum == 0))

    sage_output = {'tr_data': train_X_sage, 'tr_aspect': tr_aspect, 'widx': tr_widx, 'vocab': vocab_for_sage}
    if n_test > 0:
        sage_output['te_data'] = test_X_sage
        sage_output['te_aspect'] = te_aspect
    savemat(os.path.join(output_dir, 'sage_labeled.mat'), sage_output)
    sage_output['tr_aspect'] = tr_no_aspect
    if n_test > 0:
        sage_output['te_aspect'] = te_no_aspect
    savemat(os.path.join(output_dir, 'sage_unlabeled.mat'), sage_output)

    print("Done!")


def process_subset(items, parsed, label_fields, label_lists, vocab, output_dir, output_prefix):
    n_items = len(items)
    vocab_size = len(vocab)
    vocab_index = dict(zip(vocab, range(vocab_size)))

    ids = []
    for i, item in enumerate(items):
        if 'id' in item:
            ids.append(item['id'])
    if len(ids) != n_items:
        ids = [str(i) for i in range(n_items)]

    # create a label index using string representations
    for label_field in label_fields:
        label_list = label_lists[label_field]
        n_labels = len(label_list)
        label_list_strings = [str(label) for label in label_list]
        label_index = dict(zip(label_list_strings, range(n_labels)))

        # convert labels to a data frame
        if n_labels > 0:
            label_matrix = np.zeros([n_items, n_labels], dtype=int)
            label_vector = np.zeros(n_items, dtype=int)
            label_str = []


            for i, item in enumerate(items):
                label = item[label_field]
                label_matrix[i, label_index[str(label)]] = 1
                label_vector[i] = label_index[str(label)]
                label_str.append(str(label_index[str(label)]))

            labels_df = pd.DataFrame(label_matrix, index=ids, columns=label_list_strings)
            labels_df.to_csv(os.path.join(output_dir, output_prefix + '.' + label_field + '.csv'))
            label_vector_df = pd.DataFrame(label_vector, index=ids, columns=[label_field])
            if n_labels == 2:
                label_vector_df.to_csv(os.path.join(output_dir, output_prefix + '.' + label_field + '_vector.csv'))

    X = np.zeros([n_items, vocab_size], dtype=int)

    dat_strings = []
    dat_labels = []
    mallet_strings = []
    fast_text_lines = []

    counter = Counter()
    word_counter = Counter()
    doc_lines = []
    tokens = []
    print("Converting to count representations")

    for i, words in enumerate(parsed):
        # get the vocab indices of words that are in the vocabulary
        indices = [vocab_index[word] for word in words if word in vocab_index]
        word_subset = [word for word in words if word in vocab_index]

        counter.clear()
        counter.update(indices)
        word_counter.clear()
        word_counter.update(word_subset)

        if len(counter.keys()) > 0:
            # udpate the counts
            mallet_strings.append(str(i) + '\t' + 'en' + '\t' + ' '.join(word_subset))

            dat_string = str(int(len(counter))) + ' '
            dat_string += ' '.join([str(k) + ':' + str(int(v)) for k, v in zip(list(counter.keys()), list(counter.values()))])
            dat_strings.append(dat_string)
            # for dat formart, assume just one label is given
            if len(label_fields) > 0:
                label = items[i][label_fields[-1]]
                dat_labels.append(str(label_index[str(label)]))

            tokens.append({"tokens":" ".join(word_subset), "category": label_index[str(label)]})
            values = list(counter.values())
            X[np.ones(len(counter.keys()), dtype=int) * i, list(counter.keys())] += values

    # convert to a sparse representation
    vocab.append("@@UNKNOWN@@")
    if not os.path.isdir(os.path.join(output_dir, 'vocabulary')):
        os.mkdir(os.path.join(output_dir, 'vocabulary'))
    fh.write_list_to_text(vocab, os.path.join(output_dir, 'vocabulary', 'full.txt'))
    fh.write_list_to_text(np.unique(np.array(label_str)), os.path.join(output_dir, 'vocabulary', 'labels.txt'))
    fh.write_jsonlist(tokens, os.path.join(output_dir, output_prefix + '.jsonl'))
    # save output for Mallet
    fh.save_sparse(sparse.csr_matrix(X, dtype=float), os.path.join(output_dir, output_prefix + '.npz'))

    # save output for Jacob Eisenstein's SAGE code:
    sparse_X_sage = sparse.csr_matrix(X, dtype=float)
    vocab.remove("@@UNKNOWN@@")
    vocab_for_sage = np.zeros((vocab_size,), dtype=np.object)
    vocab_for_sage[:] = vocab

    # for SAGE, assume only a single label has been given
    if len(label_fields) > 0:
        # convert array to vector of labels for SAGE
        sage_aspect = np.argmax(np.array(labels_df.values, dtype=float), axis=1) + 1
    else:
        sage_aspect = np.ones([n_items, 1], dtype=float)
    sage_no_aspect = np.array([n_items, 1], dtype=float)
    widx = np.arange(vocab_size, dtype=float) + 1

    return sparse_X_sage, sage_aspect, sage_no_aspect, widx, vocab_for_sage


def tokenize(text, strip_html=False, lower=True, keep_emails=False, keep_at_mentions=False, keep_numbers=False, keep_alphanum=False, min_length=3, stopwords=None, vocab=None):
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
    text = replace.sub(' ', text)
    # remove single quotes
    text = re.sub(r'\'', '', text)
    # replace all whitespace with a single space
    text = re.sub(r'\s', ' ', text)
    # strip off spaces on either end
    text = text.strip()
    return text


if __name__ == '__main__':
    main(sys.argv[1:])

