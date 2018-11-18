from typing import Dict

import numpy as np
import torch
from allennlp.models.model import Model
from vae.vae import VAE
from allennlp.data import Vocabulary
from allennlp.nn import InitializerApplicator
from overrides import overrides


@Model.register("vae_clf")
class VAE_CLF(Model):
    def __init__(self, 
                 vocab: Vocabulary,
                 vae: VAE, 
                 initializer: InitializerApplicator = InitializerApplicator()):
        super(VAE_CLF, self).__init__(vocab)
        self._vae = vae
        initializer(self)

    @overrides
    def forward(self, full_tokens, stopless_tokens, label):  # pylint: disable=W0221
        """
        Given an input vector, produces the latent encoding z, followed by the mean and
        log variance of the variational distribution produced.

        z is the result of the reparameterization trick (Autoencoding Variational Bayes (Kingma et al.)).
        """

        vae_output = self._vae(full_tokens, stopless_tokens, label)
        self.metrics = self._vae.metrics
        return vae_output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: float(metric.get_metric(reset)) for metric_name, metric in self.metrics.items()}
