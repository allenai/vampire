import codecs
import json
import os
import pickle
from typing import Any, Dict, List

import numpy as np
import torch
from allennlp.data import Vocabulary
from scipy import sparse

def generate_config(seed,
                    z_dropout,
                    kld_clamp,
                    hidden_dim,
                    encoder_activation,
                    encoder_layers,
                    vocab_size,
                    batch_size,
                    cuda_device,
                    num_epochs,
                    patience,
                    learning_rate,
                    validation_metric,
                    train_path,
                    dev_path,
                    vocabulary_directory,
                    reference_counts,
                    reference_vocabulary,
                    background_data_path,
                    lazy,
                    sample,
                    min_sequence_length):
    PARAMS = {
        "numpy_seed": seed,
        "pytorch_seed": seed,
        "random_seed": seed,
        "dataset_reader": {
            "lazy": lazy,
            "sample": sample,
            "type": "vampire_reader",
            "min_sequence_length": min_sequence_length
        },
        "vocabulary": {
            "type": "from_files",
            "directory": vocabulary_directory
        },
        "train_data_path": train_path,
        "validation_data_path": dev_path,
        "model": {
            "type": "vampire",
            "bow_embedder": {
                "type": "bag_of_word_counts",
                "vocab_namespace": "vampire",
                "ignore_oov": True
            },
            "update_background_freq": False,
            "reference_counts": reference_counts,
            "reference_vocabulary": reference_vocabulary,
            "background_data_path": background_data_path,
            "vae": {
                "z_dropout": z_dropout,
                "kld_clamp": kld_clamp,

                "encoder": {
                    "activations": encoder_activation,
                    "hidden_dims": [hidden_dim] * encoder_layers,
                    "input_dim": vocab_size,
                    "num_layers": encoder_layers
                },
                "mean_projection": {
                    "activations": "linear",
                    "hidden_dims": [hidden_dim],
                    "input_dim": hidden_dim,
                    "num_layers": 1
                },
                "log_variance_projection": {
                    "activations": "linear",
                    "hidden_dims": [hidden_dim],
                    "input_dim": hidden_dim,
                    "num_layers": 1
                },
                "decoder": {
                    "activations": "linear",
                    "input_dim": hidden_dim,
                    "hidden_dims": [vocab_size],
                    "num_layers": 1 
                },
                "type": "logistic_normal"
            }
        },
        "data_loader": {
            "batch_sampler": {
                "type": "bucket",
                "batch_size": batch_size,
                "drop_last": False
            }
        },
        "trainer": {
            "epoch_callbacks": [{"type": "compute_topics"}, 
                                {"type": "kl_anneal", 
                                "kl_weight_annealing": "linear",
                                "linear_scaling": 1000}],
            "batch_callbacks": [{"type": "track_learning_rate"}],
            "cuda_device": cuda_device,
            "num_epochs": num_epochs,
            "patience": patience,
            "optimizer": {
                "lr": learning_rate,
                "type": "adam_str_lr"
            },
            "validation_metric": validation_metric,
            
        } 
    }
    return PARAMS

def write_list_to_file(ls, save_path):
    """
    Write each json object in 'jsons' as its own line in the file designated by 'save_path'.
    """
    # Open in appendation mode given that this function may be called multiple
    # times on the same file (positive and negative sentiment are in separate
    # directories).
    with open(save_path, "w+") as out_file:
        for example in ls:
            out_file.write(example)
            out_file.write('\n')

def compute_background_log_frequency(vocab: Vocabulary, vocab_namespace: str, precomputed_bg_file=None):
    """
    Load in the word counts from the JSON file and compute the
    background log term frequency w.r.t this vocabulary.
    """
    # precomputed_word_counts = json.load(open(precomputed_word_counts, "r"))
    log_term_frequency = torch.FloatTensor(vocab.get_vocab_size(vocab_namespace))
    if precomputed_bg_file is not None:
        with open(precomputed_bg_file, "r") as file_:
            precomputed_bg = json.load(file_)
    else:
        precomputed_bg = vocab._retained_counter.get(vocab_namespace)  # pylint: disable=protected-access
        if precomputed_bg is None:
            return log_term_frequency
    for i in range(vocab.get_vocab_size(vocab_namespace)):
        token = vocab.get_token_from_index(i, vocab_namespace)
        if token in ("@@UNKNOWN@@", "@@PADDING@@", '@@START@@', '@@END@@') or token not in precomputed_bg:
            log_term_frequency[i] = 1e-12
        elif token in precomputed_bg:
            if precomputed_bg[token] == 0:
                log_term_frequency[i] = 1e-12
            else:
                log_term_frequency[i] = precomputed_bg[token]
    log_term_frequency = torch.log(log_term_frequency)
    return log_term_frequency


def log_standard_categorical(logits: torch.Tensor):
    """
    Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.
    :param p: one-hot categorical distribution
    :return: H(p, u)

    Originally from https://github.com/wohlert/semi-supervised-pytorch.
    """
    # Uniform prior over y
    prior = torch.softmax(torch.ones_like(logits), dim=1)
    prior.requires_grad = False

    cross_entropy = -torch.sum(logits * torch.log(prior + 1e-8), dim=1)

    return cross_entropy


def separate_labeled_unlabeled_instances(text: torch.LongTensor,
                                         classifier_text: torch.Tensor,
                                         label: torch.LongTensor,
                                         metadata: List[Dict[str, Any]]):
    """
    Given a batch of examples, separate them into labeled and unlablled instances.
    """
    labeled_instances = {}
    unlabeled_instances = {}
    is_labeled = [int(md['is_labeled']) for md in metadata]

    is_labeled = np.array(is_labeled)
    # labeled is zero everywhere an example is unlabeled and 1 otherwise.
    labeled_indices = (is_labeled != 0).nonzero()  # type: ignore
    labeled_instances["tokens"] = text[labeled_indices]
    labeled_instances["classifier_tokens"] = classifier_text[labeled_indices]
    labeled_instances["label"] = label[labeled_indices]

    unlabeled_indices = (is_labeled == 0).nonzero()  # type: ignore
    unlabeled_instances["tokens"] = text[unlabeled_indices]
    unlabeled_instances["classifier_tokens"] = classifier_text[unlabeled_indices]

    return labeled_instances, unlabeled_instances


def schedule(batch_num, anneal_type="sigmoid"):
    """
    weight annealing scheduler
    """
    if anneal_type == "linear":
        return min(1, batch_num / 2500)
    elif anneal_type == "sigmoid":
        return float(1/(1+np.exp(-0.0025*(batch_num-2500))))
    elif anneal_type == "constant":
        return 1.0
    elif anneal_type == "reverse_sigmoid":
        return float(1/(1+np.exp(0.0025*(batch_num-2500))))
    else:
        return 0.01


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_to_json(data, output_filename, indent=2, sort_keys=True):
    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, indent=indent, sort_keys=sort_keys)


def read_json(input_filename):
    with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file, encoding='utf-8')
    return data


def read_jsonlist(input_filename):
    data = []
    with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            data.append(json.loads(line, encoding='utf-8'))
    return data


def write_jsonlist(list_of_json_objects, output_filename, sort_keys=True):
    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        for obj in list_of_json_objects:
            output_file.write(json.dumps(obj, sort_keys=sort_keys) + '\n')


def pickle_data(data, output_filename):
    with open(output_filename, 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)


def unpickle_data(input_filename):
    with open(input_filename, 'rb') as infile:
        data = pickle.load(infile)
    return data


def read_text(input_filename):
    with codecs.open(input_filename, 'r', encoding='utf-8') as input_file:
        lines = [x.strip() for x in input_file.readlines()]
    return lines


def write_list_to_text(lines, output_filename, add_newlines=True, add_final_newline=False):
    if add_newlines:
        lines = '\n'.join(lines)
        if add_final_newline:
            lines += '\n'
    else:
        lines = ''.join(lines)
        if add_final_newline:
            lines[-1] += '\n'

    with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.writelines(lines)


def save_sparse(sparse_matrix, output_filename):
    assert sparse.issparse(sparse_matrix)
    if sparse.isspmatrix_coo(sparse_matrix):
        coo = sparse_matrix
    else:
        coo = sparse_matrix.tocoo()
    row = coo.row
    col = coo.col
    data = coo.data
    shape = coo.shape
    np.savez(output_filename, row=row, col=col, data=data, shape=shape)


def load_sparse(input_filename):
    npy = np.load(input_filename)
    coo_matrix = sparse.coo_matrix((npy['data'], (npy['row'], npy['col'])), shape=npy['shape'])
    return coo_matrix.tocsc()
