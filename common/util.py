import torch
from typing import Dict


def compute_bow(token_indices: torch.Tensor, vocab: Dict, stopword_indicator: torch.Tensor=None) -> torch.Tensor:
    bow_vectors = []
    for document in token_indices['tokens']:
        vec = token_indices["tokens"].new_zeros(len(vocab)).float()
        for word_idx in document:
            if vocab.get(int(word_idx)):
                vec[word_idx] += 1
        if stopword_indicator is not None:
            vec = torch.masked_select(vec, 1 - stopword_indicator.to(vec).byte())
        vec = vec.view(1, -1)
        bow_vectors.append(vec)
    return torch.cat(bow_vectors, 0)