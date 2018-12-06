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
from common.perplexity import Perplexity
from allennlp.nn.util import get_text_field_mask
from modules.vae import VAE
from allennlp.models.archival import load_archive, Archive

@Model.register("unsupervised_vae")
class UnSupervisedVAE(Model):
    """
    Run unsupervised VAE on text

    Params
    ______

    vocab: ``Vocabulary``
        vocabulary
    vae : ``VAE``
        variational autoencoder (RNN or BOW-based)
    """
    def __init__(self,
                 vocab: Vocabulary,
                 vae: VAE=None,
                 pretrained_file: str=None,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super(UnSupervisedVAE, self).__init__(vocab)
        self.metrics = {
            'kld': Average(),
            'nll': Average(),
            'elbo': Average(),
            'perp': Perplexity(),
        }
        self.vocab = vocab
        self._vae = vae
        self.original_word_dropout = vae.word_dropout
        initializer(self)
        if pretrained_file is not None:
            archive = load_archive(pretrained_file)
            self._vae = archive.model._vae
            self._vae.vocab = vocab
        else:
            self._vae = vae

    @overrides
    def forward(self,
                epoch_num: int,
                tokens: Dict[str, torch.Tensor],
                label: torch.IntTensor=None):  # pylint: disable=W0221
        """
        Given tokens and labels, generate document representation with
        a latent code and classify.
        """
        if not self.training and self._vae.word_dropout < 1.0:
            self._vae.word_dropout=0.0
        else:
            self._vae.word_dropout=self.original_word_dropout
        # run VAE to decode with a latent code
        vae_output = self._vae(tokens=tokens,
                               epoch_num=epoch_num)
        mask = get_text_field_mask(tokens)

        # add metrics
        self.metrics["elbo"](vae_output['elbo'])
        self.metrics['perp'](vae_output['decoder_output']['flattened_decoder_output'],
                             tokens['tokens'].view(-1), mask)
        self.metrics["kld"](vae_output['kld'])
        self.metrics["kld_weight"] = vae_output['kld_weight']
        self.metrics["nll"](vae_output['nll'])
        vae_output['loss'] = vae_output['elbo']
        return vae_output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        output = {}
        for metric_name, metric in self.metrics.items():
            if isinstance(metric, float):
                output[metric_name] = metric
            else:
                output[metric_name] = float(metric.get_metric(reset))
        return output