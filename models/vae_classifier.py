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
from allennlp.models.archival import load_archive, Archive
from common.perplexity import Perplexity
from allennlp.nn.util import get_text_field_mask


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
                 pretrained_vae_file: str=None):
        super(VAE_CLF, self).__init__(vocab)
        self.metrics = {
            'l_kld': Average(),
            'u_kld': Average(),
            'l_recon': Average(),
            'u_recon': Average(),
            'l_nll': Average(),
            'u_nll': Average(),
            'accuracy': CategoricalAccuracy(),
            'elbo': Average(),
            'perp': Perplexity(),
        }
        self._num_labels = vocab.get_vocab_size("labels")
        if pretrained_vae_file is not None:
            archive = load_archive(pretrained_vae_file)
            self._vae = archive.model._vae
            self._vae.vocab = vocab
            self._vae._unlabel_index = None
        else:
            self._vae = vae

    @overrides
    def forward(self, tokens, label, **metadata):  # pylint: disable=W0221
        """
        Given tokens and labels, generate document representation with
        a latent code and classify.
        """

        # run VAE to decode with a latent code
        vae_output = self._vae(tokens, label, **metadata)
        mask = get_text_field_mask(tokens)
        # set metrics
        l_recon = vae_output.get('l_recon', np.zeros(1))
        u_recon = vae_output.get('u_recon', np.zeros(1))
        logits = vae_output['l_logits']
        elbo = vae_output['elbo']
        u_kld = vae_output.get('u_kld', np.zeros(1))
        l_kld = vae_output.get('l_kld', np.zeros(1))
        l_nll = vae_output.get('l_nll', np.zeros(1))
        u_nll = vae_output.get('u_nll', np.zeros(1))
        self.metrics['accuracy'](logits, label)
        self.metrics['perp'](vae_output['flattened_decoded_output'], tokens['tokens'].view(-1), mask)
        self.metrics["l_recon"](l_recon.mean())
        self.metrics["u_recon"](u_recon.mean())
        self.metrics["elbo"](elbo.mean())
        self.metrics["l_kld"](l_kld.mean())
        self.metrics["u_kld"](u_kld.mean())
        self.metrics["l_nll"](l_nll.mean())
        self.metrics["u_nll"](u_nll.mean())
        # create clf_output
        clf_output = vae_output
        clf_output['loss'] = vae_output['elbo']

        return clf_output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: float(metric.get_metric(reset))
                    for metric_name, metric in self.metrics.items()}
