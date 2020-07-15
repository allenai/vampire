from typing import List

import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder

from vampire.vampire.modules.pretrained_vae import PretrainedVAE


@TokenEmbedder.register("vampire_token_embedder")
class VampireTokenEmbedder(TokenEmbedder):
    """
    Compute VAMPIRE representations for use with downstream models.

    Parameters
    ----------
    model_archive : ``str``, required.
        A path to the pretrained VAMPIRE model archive
    device : ``int``, required.
        The device you'd like to load the VAE on.
    background_frequency : ``str``, required.
        Path to the precomputed background frequency file used with this VAMPIRE
    scalar_mix : ``List[int]``, optional, (default=None)
        If not ``None``, use these scalar mix parameters to weight the representations
        produced by different layers. These mixing weights are not updated during
        training.
    dropout : ``float``, optional.
        The dropout value to be applied to the VAMPIRE representations.
    requires_grad : ``bool``, optional
        If True, compute gradient of VAMPIRE parameters for fine tuning.
    projection_dim : ``int``, optional
        If given, we will project the VAMPIRE embedding down to this dimension.  We recommend that you
        try using VAE with a lot of dropout and no projection first, but we have found a few cases
        where projection helps (particularly where there is very limited training data).
    expand_dim : `bool``, optional
        If True, expand the dimensions of the output to a 3-dimensional matrix that can be concatenated with
        word vectors.
    """
    def __init__(self,
                 model_archive: str,
                 device: int,
                 background_frequency: str,
                 scalar_mix: List[int] = None,
                 dropout: float = None,
                 requires_grad: bool = False,
                 projection_dim: int = None,
                 expand_dim: bool = False) -> None:
        super(VampireTokenEmbedder, self).__init__()

        self._vae = PretrainedVAE(model_archive,
                                  device,
                                  background_frequency,
                                  requires_grad,
                                  scalar_mix,
                                  dropout)
        self._expand_dim = expand_dim
        self._layers = None
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
        The VAMPIRE representations for the input sequence, shape
        ``(batch_size, timesteps, embedding_dim)`` or ``(batch_size, timesteps)``
        depending on whether expand_dim is set to True.
        """
        vae_output = self._vae(inputs)
        embedded = vae_output['vae_representation']
        self._layers = vae_output['layers']
        if self._expand_dim:
            embedded = (embedded.unsqueeze(0)
                        .expand(inputs.shape[1], inputs.shape[0], -1)
                        .permute(1, 0, 2).contiguous())
        if self._projection:
            projection = self._projection
            for _ in range(embedded.dim() - 2):
                projection = TimeDistributed(projection)
            embedded = projection(embedded)
        return embedded

    # Custom vocab_to_cache logic requires a from_params implementation.
    @classmethod
    def from_params(cls,
                    constructor_to_call,
                    constructor_to_inspect,
                    vocab: Vocabulary,  # pylint: disable=unused-argument
                    params: Params) -> 'VampireTokenEmbedder':  # type: ignore
        # pylint: disable=arguments-differ
        # params.add_file_to_archive('model_archive')
        model_archive = params.pop('model_archive')
        device = params.pop_int('device')
        background_frequency = params.pop('background_frequency')
        requires_grad = params.pop('requires_grad', False)
        scalar_mix = params.pop("scalar_mix", None)
        dropout = params.pop_float("dropout", None)
        expand_dim = params.pop_float("expand_dim", False)
        projection_dim = params.pop_int("projection_dim", None)
        params.assert_empty(cls.__name__)
        return cls(expand_dim=expand_dim,
                   scalar_mix=scalar_mix,
                   background_frequency=background_frequency,
                   device=device,
                   model_archive=model_archive,
                   dropout=dropout,
                   requires_grad=requires_grad,
                   projection_dim=projection_dim)
