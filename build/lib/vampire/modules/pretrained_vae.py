import logging
from typing import Union, List, Dict
import torch
from overrides import overrides
from allennlp.common import Params
from allennlp.models.archival import load_archive
from allennlp.common.file_utils import cached_path
from allennlp.modules.scalar_mix import ScalarMix


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class _PretrainedVAE:
    def __init__(self,
                 model_archive: str,
                 device: int,
                 background_frequency: str,
                 requires_grad: bool = False) -> None:

        super(_PretrainedVAE, self).__init__()
        logger.info("Initializing pretrained VAMPIRE")
        self.cuda_device = device if torch.cuda.is_available() else -1
        archive = load_archive(cached_path(model_archive), cuda_device=self.cuda_device)
        self.vae = archive.model
        if not requires_grad:
            self.vae.eval()
            self.vae.freeze_weights()
        self.vae.initialize_bg_from_file(cached_path(background_frequency))
        self._requires_grad = requires_grad


class PretrainedVAE(torch.nn.Module):
    """
    Core Pretrained VAMPIRE module
    """
    def __init__(self,
                 model_archive: str,
                 device: int,
                 background_frequency: str,
                 requires_grad: bool = False,
                 scalar_mix: List[int] = None,
                 dropout: float = None) -> None:

        super(PretrainedVAE, self).__init__()
        logger.info("Initializing pretrained VAMPIRE")
        self._pretrained_model = _PretrainedVAE(model_archive=model_archive,
                                                device=device,
                                                background_frequency=background_frequency,
                                                requires_grad=requires_grad)
        self._requires_grad = requires_grad
        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        num_layers = len(self._pretrained_model.vae.vae.encoder._linear_layers) + 1  # pylint: disable=protected-access
        if not scalar_mix:
            initial_params = [1] + [-20] * (num_layers - 2) + [1]
        else:
            initial_params = scalar_mix
        self.scalar_mix = ScalarMix(
                num_layers,
                do_layer_norm=False,
                initial_scalar_parameters=initial_params,
                trainable=not scalar_mix)
        self.add_module('scalar_mix', self.scalar_mix)

    def get_output_dim(self) -> int:
        output_dim = self._pretrained_model.vae.vae.encoder.get_output_dim()
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

        layers, layer_activations = zip(*vae_output['activations'])

        scalar_mix = getattr(self, 'scalar_mix')
        representation = scalar_mix(layer_activations)

        if self._dropout:
            representation = self._dropout(representation)

        return {'vae_representation': representation, 'layers': layers}

    @classmethod
    def from_params(cls, params: Params) -> 'PretrainedVAE':
        # Add files to archive
        params.add_file_to_archive('model_archive')
        model_archive = params.pop('model_archive')
        device = params.pop('device')
        background_frequency = params.pop('background_frequency')
        requires_grad = params.pop('requires_grad', False)
        dropout = params.pop_float('dropout', None)
        scalar_mix = params.pop('scalar_mix', None)
        params.assert_empty(cls.__name__)
        return cls(model_archive=model_archive,
                   device=device,
                   background_frequency=background_frequency,
                   requires_grad=requires_grad,
                   scalar_mix=scalar_mix,
                   dropout=dropout)
