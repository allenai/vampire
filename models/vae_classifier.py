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

@Model.register("vae_classifier")
class VAE_CLF(Model):
    """
    Perform text classification with a VAE

    Params
    ______

    vocab: ``Vocabulary``
        vocabulary
    vae : ``VAE``
        variational autoencoder (RNN or BOW-based)
    classifier: ``FeedForward``
        feedforward network classifying input
    """
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
        Given tokens and labels, generate document representation with
        a latent code and classify.
        """

        # run VAE to decode with a latent code
        vae_output = self._vae(tokens, label)

        if self._vae.name == 'rnn_vae':
            clf_input = vae_output['decoded_output']
            document_vectors = torch.max(x_recon, 1)[0]
        else:
            clf_input = vae_output['decoded_output'].squeeze(0)

        # classify
        output = self._classifier(clf_input)
        logits = self._output_logits(output)
        discriminator_loss = self._discriminator_loss(logits, label)


        # set metrics
        reconstruction_loss = vae_output['reconstruction']
        elbo = vae_output['elbo']
        kld = vae_output['kld']
        nll = vae_output['nll']
        self.metrics['accuracy'](logits, label)
        self.metrics["reconstruction"](reconstruction_loss.mean())
        self.metrics["elbo"](elbo.mean())
        self.metrics["kld"](kld.mean())
        self.metrics["nll"](nll.mean())

        # create clf_output
        clf_output = vae_output
        clf_output['loss'] = vae_output['elbo'] + discriminator_loss

        return clf_output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: float(metric.get_metric(reset)) for metric_name, metric in self.metrics.items()}
