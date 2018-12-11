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
from allennlp.nn.util import get_text_field_mask
from modules.vae import VAE
from allennlp.models.archival import load_archive, Archive
from common.util import schedule, extract_topics
from tabulate import tabulate


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
        }
        self.vocab = vocab
        self._vae = vae
        self.step = 0
        initializer(self)
        if pretrained_file is not None:
            archive = load_archive(pretrained_file)
            self._vae = archive.model._vae
            self._vae.vocab = vocab
        else:
            self._vae = vae
        self.original_word_dropout = vae.word_dropout
        self.apply_batchnorm = self._vae._decoder._apply_batchnorm

    

    @overrides
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]=None,
                label: torch.IntTensor=None):  # pylint: disable=W0221
        """
        Given tokens and labels, generate document representation with
        a latent code and classify.
        """
        if not self.training:
            self._vae.weight_scheduler = lambda x: schedule(x, "constant")
            if self._vae.word_dropout < 1.0:
                self._vae.word_dropout=0.0
            self._vae.kl_weight_annealing="constant"
            self._vae._decoder._apply_batchnorm = False
        else:
            self._vae.weight_scheduler = lambda x: schedule(x, "sigmoid")
            self._vae.word_dropout=self.original_word_dropout
            self._vae.kl_weight_annealing="sigmoid"
            self._vae._decoder._apply_batchnorm = self.apply_batchnorm
        # run VAE to decode with a latent code
        vae_output = self._vae(tokens=tokens,
                               targets=targets)
        if targets is not None:
            # add metrics
            self.metrics["elbo"](vae_output['elbo'])
            self.metrics["kld"](vae_output['kld'])
            self.metrics["kld_weight"] = vae_output['kld_weight']
            self.metrics["nll"](vae_output['nll'])
            self.metrics["cos"] = vae_output['avg_cos']
            vae_output['loss'] = vae_output['elbo']
        
        if type(self._vae._decoder).__name__ == 'Bow':
            if self.step == 100:
                print(tabulate(extract_topics(self.vocab, self._vae._decoder._decoder_out.weight.data.transpose(0, 1), self._vae.bg), headers=["Topic #", "Words"]))
                self.step = 0
            else:
                self.step += 1

        return vae_output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        output = {}
        for metric_name, metric in self.metrics.items():
            if isinstance(metric, float) or isinstance(metric, np.float32):
                output[metric_name] = metric
            else:
                output[metric_name] = float(metric.get_metric(reset))
        return output