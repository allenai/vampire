from typing import List
import torch

from allennlp.common import Params
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from modules.pretrained_vae import PretrainedVAE
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.data import Vocabulary


@TokenEmbedder.register("vae_token_embedder")
class VAETokenEmbedder(TokenEmbedder):
    """
    Compute a single layer of VAE representations.
    This class serves as a convenience when you only want to use one layer of
    VAE representations at the input of your network.  It's essentially a wrapper
    around VAE(num_output_representations=1, ...)
    Parameters
    ----------
    options_file : ``str``, required.
        An VAE JSON options file.
    weight_file : ``str``, required.
        An VAE hdf5 weight file.
    do_layer_norm : ``bool``, optional.
        Should we apply layer normalization (passed to ``ScalarMix``)?
    dropout : ``float``, optional.
        The dropout value to be applied to the VAE representations.
    requires_grad : ``bool``, optional
        If True, compute gradient of VAE parameters for fine tuning.
    projection_dim : ``int``, optional
        If given, we will project the VAE embedding down to this dimension.  We recommend that you
        try using VAE with a lot of dropout and no projection first, but we have found a few cases
        where projection helps (particularly where there is very limited training data).
    vocab_to_cache : ``List[str]``, optional, (default = 0.5).
        A list of words to pre-compute and cache character convolutions
        for. If you use this option, the VAETokenEmbedder expects that you pass word
        indices of shape (batch_size, timesteps) to forward, instead
        of character indices. If you use this option and pass a word which
        wasn't pre-cached, this will break.
    scalar_mix_parameters : ``List[int]``, optional, (default=None)
        If not ``None``, use these scalar mix parameters to weight the representations
        produced by different layers. These mixing weights are not updated during
        training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 model_archive: str,
                 dropout: float = 0.5,
                 requires_grad: bool = False,
                 projection_dim: int = None) -> None:
        super(VAETokenEmbedder, self).__init__()

        self._vae = PretrainedVAE(vocab,
                                  model_archive,
                                  requires_grad,
                                  dropout)
        if projection_dim:
            self._projection = torch.nn.Linear(self._vae.get_output_dim(), projection_dim)
            self.output_dim = projection_dim
        else:
            self._projection = None
            self.output_dim = self._vae.get_output_dim()

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                word_inputs: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``
            Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
        word_inputs : ``torch.Tensor``, optional.
            If you passed a cached vocab, you can in addition pass a tensor of shape
            ``(batch_size, timesteps)``, which represent word ids which have been pre-cached.
        Returns
        -------
        The VAE representations for the input sequence, shape
        ``(batch_size, timesteps, embedding_dim)``
        """
        vae_output = self._vae(inputs)
        vae_representations = vae_output['vae_representations']
        if self._projection:
            projection = self._projection
            # for _ in range(vae_representations.dim() - 2):
            #     projection = TimeDistributed(projection)
            vae_representations = projection(vae_representations)
        return vae_representations

    # Custom vocab_to_cache logic requires a from_params implementation.
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'VAETokenEmbedder':  # type: ignore
        # pylint: disable=arguments-differ
        params.add_file_to_archive('model_archive')
        model_archive = params.pop('model_archive')
        requires_grad = params.pop('requires_grad', False)
        dropout = params.pop_float("dropout", 0.5)        
        projection_dim = params.pop_int("projection_dim", None)
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   model_archive=model_archive,
                   dropout=dropout,
                   requires_grad=requires_grad,
                   projection_dim=projection_dim)