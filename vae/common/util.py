import torch
from typing import Dict, Tuple
import numpy as np
from allennlp.data import Vocabulary
from allennlp.nn.util import get_text_field_mask
import json

def compute_bow(tokens: Dict[str, torch.Tensor],
                vocab_size: int,
                stopword_indicator: torch.Tensor=None) -> torch.Tensor:
    """
    Compute a bag of words representation (matrix of size NUM_DOCS X VOCAB_SIZE) of tokens.

    Params
    ______
    tokens : ``Dict[str, torch.Tensor]``
        tokens to compute BOW of
    index_to_token_vocabulary : ``Dict``
        vocabulary mapping index to token
    stopword_indicator: torch.Tensor, optional
        onehot tensor of size 1 x VOCAB_SIZE, indicating words in vocabulary to ignore when 
        generating BOW representation.
    """
    bow_vectors = []
    mask = get_text_field_mask({"tokens": tokens})
    for document, doc_mask in zip(tokens, mask):
        document = torch.masked_select(document, doc_mask.byte())
        vec = torch.bincount(document, minlength=vocab_size).float()
        if stopword_indicator is not None:
            vec = torch.masked_select(vec, 1 - stopword_indicator.to(vec).byte())
        vec = vec.view(1, -1)
        bow_vectors.append(vec)
    return torch.cat(bow_vectors, 0)

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
    for i in range(num_sam):
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
        sample = torch.multinomial(dist, 1)
    sample = sample.squeeze()
    return sample


def compute_background_log_frequency(vocab: Vocabulary, vocab_namespace: str, precomputed_bg_file=None):
    """ Load in the word counts from the JSON file and compute the
        background log term frequency w.r.t this vocabulary. """
    # precomputed_word_counts = json.load(open(precomputed_word_counts, "r"))
    log_term_frequency = torch.FloatTensor(vocab.get_vocab_size(vocab_namespace))
    if precomputed_bg_file is not None:
        precomputed_bg = json.load(open(precomputed_bg_file, "r"))
    else:
        precomputed_bg = vocab._retained_counter.get(vocab_namespace)
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


def log_standard_categorical(p):
    """
    Calculates the cross entropy between a (one-hot) categorical vector
    and a standard (uniform) categorical distribution.
    :param p: one-hot categorical distribution
    :return: H(p, u)
    """
    # Uniform prior over y
    prior = torch.nn.functional.softmax(torch.ones_like(p).float(), dim=-1)
    prior.requires_grad = False
    cross_entropy = torch.sum(p.float() * torch.log(prior + 1e-8), dim=-1)
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

    for dim, (s,e) in enumerate(zip(start,end)):
        interpolation[dim] = np.linspace(s,e,steps+2)

    return interpolation.T
