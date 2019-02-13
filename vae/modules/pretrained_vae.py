import logging
from typing import Union, List, Dict
import torch
from overrides import overrides
from allennlp.common import Params
from allennlp.models.archival import load_archive
from allennlp.common.file_utils import cached_path


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class _PretrainedVAE:
    def __init__(self,
                 model_archive: str,
                 background_frequency: str,
                 representations: List[str],
                 requires_grad: bool = False) -> None:

        super(_PretrainedVAE, self).__init__()
        logger.info("Initializing pretrained VAE")
        archive = load_archive(cached_path(model_archive), cuda_device=0)
        self._representations = representations
        self.vae = archive.model
        if not requires_grad:
            self.vae.eval()
            self.vae.freeze_weights()
        self.vae.initialize_bg_from_file(cached_path(background_frequency))
        self._requires_grad = requires_grad


class PretrainedVAE(torch.nn.Module):
    def __init__(self,
                 model_archive: str,
                 background_frequency: str,
                 representations: List[str] = ["encoder_output"],
                 requires_grad: bool = False,
                 dropout: float = 0.5) -> None:

        super(PretrainedVAE, self).__init__()
        logger.info("Initializing pretrained VAE")

        self._pretrained_model = _PretrainedVAE(model_archive=model_archive,
                                                background_frequency=background_frequency,
                                                requires_grad=requires_grad,
                                                representations=representations)
        self._representations = representations
        self._requires_grad = requires_grad
        self._dropout = torch.nn.Dropout(dropout)

    def get_output_dim(self) -> int:
        output_dim = 0
        if "first_layer_output" in self._representations:
            output_dim += self._pretrained_model.vae.vae.encoder.get_output_dim()
        if "theta" in self._representations:
            output_dim += self._pretrained_model.vae.vae.mean_projection.get_output_dim()
        return output_dim

    @overrides
    def forward(self,    # pylint: disable=arguments-differ
                inputs: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``, required.
        Shape ``(batch_size, timesteps)`` of word ids representing the current batch.
        Returns
        -------
        Dict with keys:
        ``'vae_representations'``: ``List[torch.Tensor]``
            A ``num_output_representations`` list of VAE representations for the input sequence.
            Each representation is shape ``(batch_size, timesteps, embedding_dim)``
            or ``(batch_size, embedding_dim)`` depending on the VAE representation being used.
        ``'mask'``:  ``torch.Tensor``
            Shape ``(batch_size, timesteps)`` long tensor with sequence mask.
        """
        vae_output = self._pretrained_model.vae(tokens={'tokens': inputs})
        vae_representations = []
        for representation in self._representations:
            if representation == "encoder_weights":
                vae_representations.append(vae_output['activations']['encoder_weights'].t())
            elif representation == "encoder_output":
                vae_representations.append(vae_output['activations']['encoder_output'])
            elif representation == "theta":
                vae_representations.append(vae_output['activations']['theta'])
            elif representation == "first_layer_output":
                vae_representations.append(vae_output['activations']['first_layer_output'])
        vae_representations = torch.cat(vae_representations, 1)
        vae_representations = self._dropout(vae_representations)
        return {'vae_representations': vae_representations}

    @classmethod
    def from_params(cls, params: Params) -> 'PretrainedVAE':
        # Add files to archive
        params.add_file_to_archive('model_archive')
        model_archive = params.pop('model_archive')
        background_frequency = params.pop('background_frequency')
        representations = params.pop('representations', ["encoder_output"])
        requires_grad = params.pop('requires_grad', False)
        dropout = params.pop_float('dropout', 0.0)
        params.assert_empty(cls.__name__)

        return cls(model_archive=model_archive,
                   background_frequency=background_frequency,
                   representations=representations,
                   requires_grad=requires_grad,
                   dropout=dropout)
