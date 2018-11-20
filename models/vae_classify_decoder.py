from typing import Dict

import numpy as np
import torch
from allennlp.models.model import Model
from modules.vae import VAE
from allennlp.data import Vocabulary
from allennlp.nn import InitializerApplicator
from overrides import overrides
from allennlp.training.metrics import CategoricalAccuracy, Average
from allennlp.modules import FeedForward

@Model.register("vae_classify_decoder")
class VAE_DECODER_CLF(Model):
    def __init__(self, 
                 vocab: Vocabulary,
                 vae: VAE,
                 classifier: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super(VAE_DECODER_CLF, self).__init__(vocab)
        self.metrics = {
            'kld': Average(),
            'reconstruction': Average(),
            'nll': Average(),
            'accuracy': CategoricalAccuracy(),
            'elbo': Average()
        }
        self._num_labels = vocab.get_vocab_size("labels")
        self._vae = vae
        self._discriminator_loss = torch.nn.CrossEntropyLoss()
        self._classifier = classifier
        self._output_logits = torch.nn.Linear(self._classifier.get_output_dim(), self._num_labels)
        
        initializer(self)

    @overrides
    def forward(self, tokens, label):  # pylint: disable=W0221
        """
        Given an input vector, produces the latent encoding z, followed by the mean and
        log variance of the variational distribution produced.

        z is the result of the reparameterization trick (Autoencoding Variational Bayes (Kingma et al.)).
        """
        vae_output = self._vae(tokens, label)

        if self._vae.name == 'rnn_vae':
            clf_input = vae_output['downstream_projection'] 
        else:
            clf_input = vae_output['x_recon'].squeeze(0)

        output = self._classifier(clf_input)
        logits = self._output_logits(output)

        discriminator_loss = self._discriminator_loss(logits, label)
        
        reconstruction_loss = vae_output['reconstruction']
        elbo = vae_output['elbo']
        kld = vae_output['kld']
        nll = vae_output['nll']
        clf_output = vae_output
        clf_output['loss'] = vae_output['elbo'] + discriminator_loss
        # set metrics
        self.metrics['accuracy'](logits, label)
        self.metrics["reconstruction"](reconstruction_loss.mean())
        self.metrics["elbo"](elbo.mean())
        self.metrics["kld"](kld.mean())
        self.metrics["nll"](nll.mean())
        return clf_output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: float(metric.get_metric(reset)) for metric_name, metric in self.metrics.items()}
