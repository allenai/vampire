import json
import torch
import numpy as np
from allennlp.data import Vocabulary


def check_dispersion(vecs, num_sam=10):
    """
    Check the dispersion of vecs.
    :param vecs:  [batch_sz, lat_dim]
    :param num_sam: number of samples to check
    :return:
    """
    vecs = vecs.unsqueeze(0)
    # vecs: n_samples, batch_sz, lat_dim
    if vecs.size(1) <= 2:
        return torch.zeros(1)
    cos_sim = 0
    for _ in range(num_sam):
        idx1 = np.random.randint(0, vecs.size(1) - 1)
        while True:
            idx2 = np.random.randint(0, vecs.size(1) - 1)
            if idx1 != idx2:
                break
        cos_sim += np.cos(vecs[0][idx1].detach().cpu().numpy(), vecs[0][idx2].detach().cpu().numpy())
    return cos_sim / num_sam


def sample(dist, strategy='greedy'):
    if strategy == 'greedy':
        dist = torch.nn.functional.softmax(dist, dim=-1)
        sample_ = torch.multinomial(dist, 1)
    sample_ = sample_.squeeze()
    return sample_


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


def one_hot(idxs, new_dim_size):
    return (idxs.unsqueeze(-1) == torch.arange(new_dim_size, device=idxs.device)).float()


def log_standard_categorical(probabilities):
    """
    Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.
    :param p: one-hot categorical distribution
    :return: H(p, u)
    """
    # Uniform prior over y
    prior = torch.nn.functional.softmax(torch.ones_like(probabilities).float(), dim=-1)
    prior.requires_grad = False
    cross_entropy = torch.sum(probabilities.float() * torch.log(prior + 1e-8), dim=-1)
    return cross_entropy


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


def interpolate(start, end, steps):

    interpolation = np.zeros((start.shape[0], steps + 2))

    for dim, (start_, end_) in enumerate(zip(start, end)):
        interpolation[dim] = np.linspace(start_, end_, steps+2)

    return interpolation.T
