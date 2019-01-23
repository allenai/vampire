import json
import logging
from allennlp.data import Vocabulary
from typing import Union, List, Dict, Any
import warnings

import torch
from torch.nn.modules import Dropout

import numpy
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
import os
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.common import Params
from allennlp.models.archival import load_archive, Archive

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class _PretrainedVAE(torch.nn.Module):
    def __init__(self,
                 vocab: Vocabulary,
                 model_archive: str,
                 requires_grad: bool=False) -> None:

        super(_PretrainedVAE, self).__init__()
        logger.info("Initializing pretrained VAE")
        archive = load_archive(model_archive)
        self._vae = archive.model
        if not requires_grad:
            self._vae._freeze_weights()
        dir_path = os.path.dirname(os.path.realpath(model_archive))
        self._vae._initialize_bg_from_file(os.path.join(dir_path, "vocabulary", "vae.bgfreq.json"))
        self._requires_grad = requires_grad


class PretrainedVAE(torch.nn.Module):
    def __init__(self,
                 vocab: Vocabulary,
                 model_archive: str,
                 requires_grad: bool=False,
                 dropout: float=0.5) -> None:

        super(PretrainedVAE, self).__init__()
        logger.info("Initializing pretrained VAE")

        self._pretrained_model = _PretrainedVAE(vocab=vocab,
                                                model_archive=model_archive,
                                                requires_grad=requires_grad)
        self._requires_grad = requires_grad
        self._dropout = torch.nn.Dropout(dropout)

    
    def get_output_dim(self):
        return self._pretrained_model._vae._encoder._architecture.get_output_dim()
    
    def forward(self,    # pylint: disable=arguments-differ
                inputs: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``, required.
        Shape ``(batch_size, timesteps, 50)`` of word ids representing the current batch.
        Returns
        -------
        Dict with keys:
        ``'vae_representations'``: ``List[torch.Tensor]``
            A ``num_output_representations`` list of VAE representations for the input sequence.
            Each representation is shape ``(batch_size, timesteps, embedding_dim)`` or ``(batch_size, timesteps, embedding_dim)``,
            depending on the VAE being used.
        ``'mask'``:  ``torch.Tensor``
            Shape ``(batch_size, timesteps)`` long tensor with sequence mask.
        """
        vae_output = self._pretrained_model._vae(tokens={'tokens': inputs})
        vae_representations = vae_output['activations']['encoder_output']
        vae_representations = self._dropout(vae_representations)
        return {'vae_representations': vae_representations, 'mask': vae_output['mask']}

    @classmethod
    def from_params(cls, params: Params) -> 'PretrainedVAE':
        # Add files to archive
        params.add_file_to_archive('model_archive')
        model_archive = params.pop('model_archive')
        requires_grad = params.pop('requires_grad', False)
        dropout = params.pop_float('dropout', 0.5)
        params.assert_empty(cls.__name__)

        return cls(model_archive=model_archive,
                   requires_grad=requires_grad,
                   dropout=dropout)