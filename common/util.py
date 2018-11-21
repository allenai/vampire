import torch
from typing import Dict


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