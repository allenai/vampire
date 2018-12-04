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


@Model.register("semisupervised_vae")
class SemiSupervisedVAE(Model):
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
        super(SemiSupervisedVAE, self).__init__(vocab)
        self.metrics = {
            'kld': Average(),
            'nll': Average(),
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
    def forward(self, tokens: Dict[str, torch.Tensor], epoch_num: int, label: torch.IntTensor, **metadata):  # pylint: disable=W0221
        """
        Given tokens and labels, generate document representation with
        a latent code and classify.
        """

        # run VAE to decode with a latent code
        vae_output = self._vae(tokens=tokens, epoch_num=epoch_num, label=label, **metadata)
        mask = get_text_field_mask(tokens)
        # set metrics
        logits = vae_output['logits']
        self.metrics['accuracy'](logits, label)
        self.metrics['perp'](vae_output['decoder_output']['flattened_decoder_output'],
                             tokens['tokens'].view(-1), mask)

        self.metrics["elbo"](vae_output['elbo'].mean())
        self.metrics["kld"](vae_output['kld'].mean())
        self.metrics["kld_weight"] = vae_output['kld_weight']
        self.metrics["nll"](vae_output['nll'].mean())
        # create clf_output
        vae_output['loss'] = vae_output['elbo'].mean()

        return vae_output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        output = {}
        for metric_name, metric in self.metrics.items():
            if isinstance(metric, float):
                output[metric_name] = metric
            else:
                output[metric_name] = float(metric.get_metric(reset))
        return output
