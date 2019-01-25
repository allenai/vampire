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


def separate_labelled_unlabelled_instances(input_tokens: torch.LongTensor,
                                           filtered_tokens: torch.Tensor,
                                           sentiment: torch.LongTensor,
                                           labelled: torch.LongTensor):
    """
    Given a batch of examples, separate them into labelled and unlablled instances.
    """
    labelled_instances = {}
    unlabelled_instances = {}

    # Labelled is zero everywhere an example is unlabelled and 1 otherwise.
    labelled_indices = (labelled != 0).nonzero().squeeze()
    labelled_instances["tokens"] = input_tokens[labelled_indices]
    labelled_instances["stopless_tokens"] = filtered_tokens[labelled_indices]
    labelled_instances["sentiment"] = sentiment[labelled_indices]
    labelled_instances["labelled"] = True

    unlabelled_indices = (labelled == 0).nonzero().squeeze()
    unlabelled_instances["tokens"] = input_tokens[unlabelled_indices]
    unlabelled_instances["stopless_tokens"] = filtered_tokens[unlabelled_indices]
    unlabelled_instances["labelled"] = False

    return labelled_instances, unlabelled_instances


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
