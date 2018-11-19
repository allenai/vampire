from typing import Dict

import numpy as np
import torch
from allennlp.models.model import Model
from modules.vae import VAE
from allennlp.data import Vocabulary
from allennlp.nn import InitializerApplicator
from overrides import overrides
from allennlp.training.metrics import CategoricalAccuracy, Average


@Model.register("vae_clf")
class VAE_CLF(Model):
    def __init__(self, 
                 vocab: Vocabulary,
                 vae: VAE, 
                 initializer: InitializerApplicator = InitializerApplicator()):
        super(VAE_CLF, self).__init__(vocab)
        self.metrics = {
            'kld': Average(),
            'reconstruction': Average(),
            'nll': Average(),
            'accuracy': CategoricalAccuracy(),
            'elbo': Average()
        }
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
        logits = vae_output['logits']
        reconstruction_loss = vae_output['reconstruction']
        elbo = vae_output['elbo']
        kld = vae_output['kld']
        nll = vae_output['nll']
        
        clf_output = vae_output
        clf_output['loss'] = vae_output['elbo']
        # set metrics
        self.metrics['accuracy'](logits, label)
        self.metrics["reconstruction"](reconstruction_loss.mean())
        self.metrics["elbo"](elbo.mean())
        self.metrics["kld"](kld.mean())
        self.metrics["nll"](nll.mean())
        return clf_output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: float(metric.get_metric(reset)) for metric_name, metric in self.metrics.items()}
