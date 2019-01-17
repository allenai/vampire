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


punct_chars = list(set(string.punctuation) - set("'"))
punct_chars.sort()
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))
alpha = re.compile('^[a-zA-Z_]+$')
alpha_or_num = re.compile('^[a-zA-Z_]+|[0-9_]+$')
alphanum = re.compile('^[a-zA-Z0-9_]+$')


def preprocess_data(train_infile, test_infile, dev_infile, output_dir, train_prefix, test_prefix, dev_prefix, min_doc_count=0, max_doc_freq=1.0, vocab_size=None, sample=None, stopwords=None, keep_num=False, keep_alphanum=False, strip_html=False, lower=True, min_length=3, label_fields=None):

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

    train_items = fh.read_jsonlist(train_infile, sample=sample)
    n_train = len(train_items)
    print("Found {:d} training documents".format(n_train))

    if test_infile is not None:
        test_items = fh.read_jsonlist(test_infile)
        n_test = len(test_items)
        print("Found {:d} test documents".format(n_test))
    else:
        test_items = []
        n_test = 0

    if dev_infile is not None:
        dev_items = fh.read_jsonlist(dev_infile)
        n_dev = len(dev_items)
        print("Found {:d} dev documents".format(n_dev))
    else:
        dev_items = []
        n_dev = 0

    all_items = train_items + test_items + dev_items
    all_items_dict = {"train" : train_items, "test": test_items, "dev": dev_items}

    n_items = n_train + n_test + n_dev

    label_lists = {}
    if label_fields is not None:
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
    dev_parsed = []

    print("Parsing %d documents" % n_items)
    word_counts = Counter()
    doc_counts = Counter()
    count = 0

    vocab = None
    for data_name, items in tqdm(all_items_dict.items()):
        for item in tqdm(items):
            text = item['tokens']
            tokens, _ = tokenize(text, strip_html=strip_html, lower=lower, keep_numbers=keep_num, keep_alphanum=keep_alphanum, min_length=min_length, stopwords=stopword_set, vocab=vocab)

            # store the parsed documents
            if data_name == 'train':
                train_parsed.append(tokens)
            elif data_name == 'test':
                test_parsed.append(tokens)
            elif data_name == 'dev':
                dev_parsed.append(tokens)
            if data_name == 'train':
                # keep track of the number of documents with each word
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
    # convert to a sparse representation
    if "@@UNKNOWN@@" not in vocab:
        vocab.append("@@UNKNOWN@@")

    vocab_size = len(vocab)
    print("Final vocab size = %d" % vocab_size)

    print("Most common words remaining:", ' '.join(vocab[:10]))
    vocab.sort()

    fh.write_jsonlist(vocab, os.path.join(output_dir, train_prefix + '.vocab.json'))

    process_subset(train_items, train_parsed, label_fields, label_lists, vocab, output_dir, train_prefix)
    if n_test > 0:
        process_subset(test_items, test_parsed, label_fields, label_lists, vocab, output_dir, test_prefix)

    if n_dev > 0:
        process_subset(dev_items, dev_parsed, label_fields, label_lists, vocab, output_dir, dev_prefix)


    total = np.sum([c for k, c in word_counts.items()])
    freqs = {k: c / float(total) for k, c in word_counts.items()}
    fh.write_to_json(freqs, os.path.join(output_dir, train_prefix + '.bgfreq.json'))

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

    # X = np.zeros([n_items, vocab_size], dtype=int)

    dat_strings = []
    dat_labels = []
    mallet_strings = []
    fast_text_lines = []

    counter = Counter()
    word_counter = Counter()
    doc_lines = []
    tokens = []
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
            # udpate the counts
            mallet_strings.append(str(i) + '\t' + 'en' + '\t' + ' '.join(word_subset))

            dat_string = str(int(len(counter))) + ' '
            dat_string += ' '.join([str(k) + ':' + str(int(v)) for k, v in zip(list(counter.keys()), list(counter.values()))])
            dat_strings.append(dat_string)
            # for dat formart, assume just one label is given
            if len(label_fields) > 0:
                label = items[i][label_fields[-1]]
                dat_labels.append(str(label_index[str(label)]))

            tokens.append({"text":" ".join(word_subset), "label": label_index[str(label)]})
            values = list(counter.values())
            # X[np.ones(len(counter.keys()), dtype=int) * i, list(counter.keys())] += values

    
    assert len([x for x in vocab if x == '@@UNKNOWN@@']) == 1
    if not os.path.isdir(os.path.join(output_dir, 'vocabulary')):
        os.mkdir(os.path.join(output_dir, 'vocabulary'))
    fh.write_list_to_text(vocab, os.path.join(output_dir, 'vocabulary', 'full.txt'))
    fh.write_list_to_text(np.unique(np.array(label_str)), os.path.join(output_dir, 'vocabulary', 'labels.txt'))
    fh.write_jsonlist(tokens, os.path.join(output_dir, output_prefix + '.jsonl'))
    fh.write_list_to_text(["0", "1"], os.path.join(output_dir, 'vocabulary', 'is_labeled.txt'))
    fh.write_list_to_text(["full", "labels", "is_labeled"], os.path.join(output_dir, 'vocabulary', 'non_padded_namespaces.txt'))
    # save output for Mallet
    # fh.save_sparse(sparse.csr_matrix(X, dtype=float), os.path.join(output_dir, output_prefix + '.npz'))

    # save output for Jacob Eisenstein's SAGE code:
    # sparse_X_sage = sparse.csr_matrix(X, dtype=float)
    # vocab_for_sage = np.zeros((vocab_size,), dtype=np.object)
    # vocab_for_sage[:] = vocab

    # # for SAGE, assume only a single label has been given
    # if len(label_fields) > 0:
    #     # convert array to vector of labels for SAGE
    #     sage_aspect = np.argmax(np.array(labels_df.values, dtype=float), axis=1) + 1
    # else:
    #     sage_aspect = np.ones([n_items, 1], dtype=float)
    # sage_no_aspect = np.array([n_items, 1], dtype=float)
    # widx = np.arange(vocab_size, dtype=float) + 1

    # return sparse_X_sage, sage_aspect, sage_no_aspect, widx, vocab_for_sage


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
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_dir', dest='data_dir', type=str, help='path to data directory', required=True)
    parser.add_argument('-o','--output_dir', type=str, help='output directory', required=True)
    parser.add_argument('-l','--label', type=str, help='label name', required=False, default='label')
    parser.add_argument('-v','--vocab_size', type=int, help='vocab size', required=True)
    parser.add_argument('-s','--subsamples', nargs='+', type=int, help='subsample sizes', required=True)
    parser.add_argument('-u','--unlabeled_size', type=int, help='size of unlabeled data', required=False)
    parser.add_argument('-r','--stopwords', type='str', help='stopword type', required=False)
    args = parser.parse_args()
    
    if not os.path.exists(os.path.join(args.output_dir, "full", "train_raw.jsonl")):
        copyfile(os.path.join(args.data_dir, "train.jsonl"), os.path.join(args.output_dir, "full", "train_raw.jsonl"))
    
    if not os.path.exists(os.path.join(args.output_dir, "full", "dev_raw.jsonl")):
        copyfile(os.path.join(args.data_dir, "dev.jsonl"), os.path.join(args.output_dir, "full", "dev_raw.jsonl"))
    
    if not os.path.exists(os.path.join(args.output_dir, "full", "test_raw.jsonl")):
        copyfile(os.path.join(args.data_dir, "test.jsonl"), os.path.join(args.output_dir, "full", "test_raw.jsonl"))

    if not os.path.exists(os.path.join(args.output_dir, "unlabeled", "train_raw.jsonl")):
        copyfile(os.path.join(args.data_dir, "unlabeled.jsonl"), os.path.join(args.output_dir, "unlabeled", "train_raw.jsonl"))


    full_train = os.path.join(args.output_dir, "full", "train_raw.jsonl")
    full_test = os.path.join(args.output_dir, "full", "test_raw.jsonl")
    full_dev = os.path.join(args.output_dir, "full", "dev_raw.jsonl")

    with open(full_train, 'r') as f:
        labeled_data = pd.read_json(f, lines=True)
    
   
    if args.unlabeled_size is not None:
        unlabeled_data = labeled_data.sample(n=args.unlabeled_size)
        unlabeled_data[args.label] = 0
        labeled_data = labeled_data.drop(unlabeled_data.index)
        out_dir = os.path.join(args.output_dir, "unlabeled")
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        unlabeled_data.to_json(os.path.join(out_dir, "train_raw.jsonl"), lines=True, orient='records')

    samples = {}
    for size in args.subsamples:
        sample = labeled_data.sample(n=size)
        samples[size] = sample

    preprocess_data(train_infile=full_train,
                    test_infile=full_test,
                    dev_infile=full_dev,
                    output_dir=os.path.join(args.output_dir, "full"),
                    train_prefix="train",
                    test_prefix="test",
                    dev_prefix="dev",
                    min_doc_count=0,
                    max_doc_freq=1.0,
                    vocab_size=args.vocab_size,
                    sample=None,
                    stopwords=args.stopwords,
                    keep_num=False,
                    keep_alphanum=False,
                    strip_html=False,
                    lower=True,
                    min_length=3,
                    label_fields=args.label)

    for size, sample in samples.items():
        out_dir = os.path.join(args.output_dir, str(size))
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        sample.to_json(os.path.join(out_dir, "train_raw.jsonl"), lines=True, orient='records')
        preprocess_data(train_infile=os.path.join(out_dir, "train_raw.jsonl"),
                    test_infile=None,
                    dev_infile=None,
                    output_dir=out_dir,
                    train_prefix="train",
                    test_prefix="test",
                    dev_prefix="dev",
                    min_doc_count=0,
                    max_doc_freq=1.0,
                    vocab_size=args.vocab_size,
                    sample=None,
                    stopwords=args.stopwords,
                    keep_num=False,
                    keep_alphanum=False,
                    strip_html=False,
                    lower=True,
                    min_length=3,
                    label_fields=args.label)


    preprocess_data(train_infile=os.path.join(args.output_dir, "unlabeled", "train_raw.jsonl"),
                test_infile=None,
                dev_infile=None,
                output_dir=os.path.join(args.output_dir, "unlabeled"),
                train_prefix="train",
                test_prefix="test",
                dev_prefix="dev",
                min_doc_count=0,
                max_doc_freq=1.0,
                vocab_size=10000000,
                sample=120000,
                stopwords=args.stopwords,
                keep_num=False,
                keep_alphanum=False,
                strip_html=False,
                lower=True,
                min_length=3,
                label_fields=args.label)

    print("Done!")