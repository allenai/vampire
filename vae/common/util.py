from typing import Any, Dict, List
import json
import torch
import numpy as np
from allennlp.data import Vocabulary


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
                                         filtered_text: torch.Tensor,
                                         label: torch.LongTensor,
                                         metadata: List[Dict[str, Any]]):
    """
    Given a batch of examples, separate them into labeled and unlablled instances.
    """
    labeled_instances = {}
    unlabeled_instances = {}
    is_labeled = [int(md['is_labeled']) for md in metadata]

    # labeled is zero everywhere an example is unlabeled and 1 otherwise.
    labeled_indices = (is_labeled != 0).nonzero().squeeze()
    labeled_instances["text"] = text[labeled_indices]
    labeled_instances["filtered_text"] = filtered_text[labeled_indices]
    labeled_instances["label"] = label[labeled_indices]

    unlabeled_indices = (is_labeled == 0).nonzero().squeeze()
    unlabeled_instances["text"] = text[unlabeled_indices]
    unlabeled_instances["filtered_text"] = filtered_text[unlabeled_indices]

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
