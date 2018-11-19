import torch
from allennlp.common import Registrable
from allennlp.models.archival import Archive
from typing import Dict, Optional, List, Any


class VAE(Registrable, torch.nn.Module):

    def get_latent_dim(self) -> int:
        return self.latent_dim

    def get_hidden_dim(self) -> int:
        return self.hidden_dim

    def _initialize_weights_from_archive(self, archive: Archive) -> None:
        raise NotImplementedError

    def _encode(self, tokens, label, n_sample):
        raise NotImplementedError

    def _decode(self, latent_code):
        raise NotImplementedError

    def _reconstruction_loss(self, **kwargs):
        raise NotImplementedError

    def _discriminator(self, tokens: Dict, label: torch.IntTensor):
        raise NotImplementedError

    def _reparameterize(self, posteriors):
        raise NotImplementedError

    def forward(self, tokens, label) -> Dict:
        raise NotImplementedError

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        raise NotImplementedError
