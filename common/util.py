import torch
from typing import Dict, Tuple
import numpy as np


def compute_bow(tokens: Dict[str, torch.Tensor],
                index_to_token_vocabulary: Dict,
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
    for document in tokens:
        vec = tokens.new_zeros(len(index_to_token_vocabulary)).float()
        for word_idx in document:
            if index_to_token_vocabulary.get(int(word_idx)):
                vec[word_idx] += 1
        if stopword_indicator is not None:
            vec = torch.masked_select(vec, 1 - stopword_indicator.to(vec).byte())
        vec = vec.view(1, -1)
        bow_vectors.append(vec)
    return torch.cat(bow_vectors, 0)


def sample(dist, strategy='greedy'):
    if strategy == 'greedy':
        dist = torch.nn.functional.softmax(dist, dim=-1)
        sample = torch.multinomial(dist, 1)
    sample = sample.squeeze()
    return sample

def split_instances(tokens: Dict[str, torch.Tensor],
                    unlabeled_index: int=None,
                    label: torch.IntTensor=None,
                    metadata: torch.IntTensor=None,
                    embedded_tokens: torch.FloatTensor=None) -> Tuple[Dict[str, torch.Tensor],
                                                        Dict[str, torch.Tensor]]:
        """
        Given a batch of examples, separate them into labelled and unlablled instances.
        """
        labeled_instances = {}
        unlabeled_instances = {}
        if unlabeled_index is None:
            unlabeled_index = -1
        
        labeled_indices = (label != unlabeled_index).nonzero().squeeze()

        if labeled_indices.nelement() > 0:
            labeled_tokens = tokens['tokens'][labeled_indices, :]
            labeled_labels = label[labeled_indices]
            if embedded_tokens is not None:
                labeled_instances["embedded_tokens"] = embedded_tokens[labeled_indices, : , :]
            if len(labeled_tokens.shape) == 1:
                labeled_tokens = labeled_tokens.unsqueeze(0)
            if len(labeled_labels.shape) == 0:
                labeled_labels = labeled_labels.unsqueeze(0)
            labeled_instances["tokens"] = {"tokens": labeled_tokens}
            labeled_instances["label"] = labeled_labels
            if metadata is not None:
                labeled_instances["metadata"] = metadata[labeled_indices]
        

        unlabeled_indices = (label == unlabeled_index).nonzero().squeeze()
        if unlabeled_indices.nelement() > 0:
            unlabeled_tokens = tokens['tokens'][unlabeled_indices, :]
            if len(unlabeled_tokens.shape) == 1:
                unlabeled_tokens = unlabeled_tokens.unsqueeze(0)
            unlabeled_instances["tokens"] = {"tokens": unlabeled_tokens}
            if embedded_tokens is not None:
                unlabeled_instances["embedded_tokens"] = embedded_tokens[unlabeled_indices, : , :]
            if metadata is not None:
                unlabeled_instances["metadata"] = metadata[unlabeled_indices]
        
        return labeled_instances, unlabeled_instances

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
    else:
        return 1

def interpolate(start, end, steps):

    interpolation = np.zeros((start.shape[0], steps + 2))

    for dim, (s,e) in enumerate(zip(start,end)):
        interpolation[dim] = np.linspace(s,e,steps+2)

    return interpolation.T
