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
        super(VAE_CLF, self).__init__(vocab)
        self.metrics = {
            'kld': Average(),
            'reconstruction': Average(),
            'nll': Average(),
            'accuracy': CategoricalAccuracy(),
            'elbo': Average(),
        }
        self._num_labels = vocab.get_vocab_size("labels") 
        # if vocab.get_token_to_index_vocabulary("labels").get("-1") is not None:
        #     self._num_labels = self._num_labels - 1
        self._vae = vae
        self._classifier = classifier
        self._classifier_loss = torch.nn.CrossEntropyLoss()
        self._output_logits = torch.nn.Linear(self._classifier.get_output_dim(), self._num_labels)
        initializer(self)

    @overrides
    def forward(self, tokens, label, **metadata):  # pylint: disable=W0221
        """
        Given tokens and labels, generate document representation with
        a latent code and classify.
        """

        # run VAE to decode with a latent code
        vae_output = self._vae(tokens, label, **metadata)

        if self._vae.__class__.__name__ in ('RNN_VAE', 'SCHOLAR_RNN'):
            decoded_output = vae_output['decoded_output']
            document_vectors = torch.max(decoded_output, 1)[0]
        else:
            document_vectors = vae_output['decoded_output'].squeeze(0)

        is_labeled = (label != self.vocab.get_token_to_index_vocabulary("labels")["-1"]).nonzero().squeeze()
        if is_labeled.sum() > 0:
            label = label[is_labeled]
            # classify
            output = self._classifier(document_vectors)
            logits = self._output_logits(output)
            logits = logits[is_labeled, :]
            if len(logits.shape) == 1:
                logits = logits.unsqueeze(0)
            if len(label.shape) == 0:
                label = label.unsqueeze(0)
            classifier_loss = self._classifier_loss(logits, label)
            print(logits.max(1)[1])
            self.metrics['accuracy'](logits, label)
        else:
            classifier_loss = 0


        # set metrics
        reconstruction_loss = vae_output['reconstruction']
        elbo = vae_output['elbo']
        kld = vae_output['kld']
        nll = vae_output['nll']
        self.metrics["reconstruction"](reconstruction_loss.mean())
        self.metrics["elbo"](elbo.mean())
        self.metrics["kld"](kld.mean())
        self.metrics["nll"](nll.mean())
        # create clf_output
        clf_output = vae_output
        clf_output['loss'] = vae_output['elbo'] + classifier_loss

        return clf_output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: float(metric.get_metric(reset)) for metric_name, metric in self.metrics.items()}
