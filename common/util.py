import torch
from typing import Dict, Tuple


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
    for document in tokens['tokens']:
        vec = tokens["tokens"].new_zeros(len(index_to_token_vocabulary)).float()
        for word_idx in document:
            if index_to_token_vocabulary.get(int(word_idx)):
                vec[word_idx] += 1
        if stopword_indicator is not None:
            vec = torch.masked_select(vec, 1 - stopword_indicator.to(vec).byte())
        vec = vec.view(1, -1)
        bow_vectors.append(vec)
    return torch.cat(bow_vectors, 0)

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

def enumerate_discrete(x, y_dim):
    """
    Generates a `torch.Tensor` of size batch_size x n_labels of
    the given label.
    Example: generate_label(2, 1, 3) #=> torch.Tensor([[0, 1, 0],
                                                       [0, 1, 0]])
    :param x: tensor with batch size to mimic
    :param y_dim: number of total labels
    :return variable
    """
    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.size(0)
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])

    if x.is_cuda:
        generated = generated.cuda()

    return Variable(generated.float())