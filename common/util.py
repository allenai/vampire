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
                    label: torch.IntTensor=None,
                    metadata: torch.IntTensor=None) -> Tuple[Dict[str, torch.Tensor],
                                                        Dict[str, torch.Tensor]]:
        """
        Given a batch of examples, separate them into labelled and unlablled instances.
        """
        labeled_instances = {}
        unlabeled_instances = {}

        labeled_indices = (label != -1).nonzero().squeeze()
        labeled_instances["tokens"] = tokens['tokens'][labeled_indices, :]
        if label is not None:
            labeled_instances["label"] = label[labeled_indices]
        if metadata is not None:
            labeled_instances["metadata"] = metadata[labeled_indices]

        unlabeled_indices = (label == -1).nonzero().squeeze()
        unlabeled_instances["tokens"] = tokens['tokens'][unlabeled_indices, :]
        unlabeled_instances["label"] = None
        if metadata is not None:
            unlabeled_instances["metadata"] = metadata[unlabeled_indices]

        return labeled_instances, unlabeled_instances