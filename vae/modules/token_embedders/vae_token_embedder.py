from typing import List

import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn import util
from torch.nn.functional import embedding

from vae.modules.pretrained_vae import PretrainedVAE


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
                 model_archive: str,
                 background_frequency: str,
                 dropout: float = 0.0,
                 representations: List[str] = ["encoder_output"],
                 requires_grad: bool = False,
                 projection_dim: int = None,
                 expand_dim: bool = False) -> None:
        super(VAETokenEmbedder, self).__init__()

        self._vae = PretrainedVAE(model_archive,
                                  background_frequency,
                                  representations,
                                  requires_grad,
                                  dropout)
        self._representations = representations
        self._expand_dim = expand_dim
        if projection_dim:
            self._projection = torch.nn.Linear(self._vae.get_output_dim(), projection_dim)
            self.output_dim = projection_dim
        else:
            self._projection = None
            self.output_dim = self._vae.get_output_dim()

    def get_output_dim(self) -> int:
        
        return self.output_dim

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``
            Shape ``(batch_size, timesteps)`` of character ids representing the current batch.
        Returns
        -------
        The VAE representations for the input sequence, shape
        ``(batch_size, timesteps, embedding_dim)`` or ``(batch_size, timesteps)``
        depending on the representation chosen and expansion.
        """
        vae_output = self._vae(inputs)
        embedded = vae_output['vae_representations']

        if self._expand_dim:
            embedded = (embedded.unsqueeze(0)
                        .expand(inputs.shape[1], inputs.shape[0], -1)
                        .permute(1, 0, 2).contiguous())

        elif self._representations == ['encoder_weights']:
            original_size = inputs.size()
            inputs = util.combine_initial_dims(inputs)
            embedded = embedding(inputs, embedded)
            # Now (if necessary) add back in the extra dimensions.
            embedded = util.uncombine_initial_dims(embedded, original_size)

        if self._projection:
            projection = self._projection
            for _ in range(embedded.dim() - 2):
                projection = TimeDistributed(projection)
            embedded = projection(embedded)

        return embedded

    # Custom vocab_to_cache logic requires a from_params implementation.
    @classmethod
    def from_params(cls,
                    vocab: Vocabulary,  # pylint: disable=unused-argument
                    params: Params) -> 'VAETokenEmbedder':  # type: ignore
        # pylint: disable=arguments-differ
        params.add_file_to_archive('model_archive')
        model_archive = params.pop('model_archive')
        background_frequency = params.pop('background_frequency')
        requires_grad = params.pop('requires_grad', False)
        representations = params.pop('representations', ["encoder_output"])
        dropout = params.pop_float("dropout", 0.5)
        expand_dim = params.pop_float("expand_dim", False)
        projection_dim = params.pop_int("projection_dim", None)
        params.assert_empty(cls.__name__)
        return cls(expand_dim=expand_dim,
                   representations=representations,
                   background_frequency=background_frequency,
                   model_archive=model_archive,
                   dropout=dropout,
                   requires_grad=requires_grad,
                   projection_dim=projection_dim)
