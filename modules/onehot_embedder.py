import torch
from allennlp.modules import TokenEmbedder
from common.util import compute_bow
from typing import Dict
from allennlp.data import Vocabulary
from allennlp.common import Params, Tqdm

@TokenEmbedder.register("onehot_token_embedder")
class OnehotTokenEmbedder(TokenEmbedder):
    """
    """
    def __init__(self, num_embeddings: int, idx2tok: Dict, projection_dim: int = None) -> None:
        super(OnehotTokenEmbedder, self).__init__()
        self.num_embeddings = num_embeddings
        self.idx2tok = idx2tok
        if projection_dim:
            self._projection = torch.nn.Linear(num_embeddings, projection_dim)
        else:
            self._projection = None
        self.output_dim = projection_dim or num_embeddings
    
    def get_output_dim(self):
        return self.num_embeddings

    def forward(self, # pylint: disable=arguments-differ
                inputs: torch.Tensor) -> torch.Tensor:
        bow_output = compute_bow(inputs, self.idx2tok)
        if self._projection:
            projection = self._projection
            bow_output = projection(bow_output)
        return bow_output

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'Embedding':  # type: ignore
        """
        we look for a ``vocab_namespace`` key in the parameter dictionary to know which vocabulary to use.
        """
        # pylint: disable=arguments-differ
        vocab_namespace = params.pop("vocab_namespace", "tokens")
        num_embeddings = vocab.get_vocab_size(vocab_namespace)
        idx2tok = vocab.get_index_to_token_vocabulary(vocab_namespace)
        projection_dim = params.pop_int("projection_dim", None)
        params.assert_empty(cls.__name__)

        return cls(num_embeddings=num_embeddings,
                   idx2tok=idx2tok,
                   projection_dim=projection_dim)
                   