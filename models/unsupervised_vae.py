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

@Model.register("unsupervised_vae")
class VAE(Model):
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
                 vae: VAE,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super(VAE, self).__init__(vocab)
        self.metrics = {
            'kld': Average(),
            'nll': Average(),
            'elbo': Average(),
            'perp': Perplexity(),
        }
        self.vocab = vocab
        self._vae = vae
        initializer(self)

    @overrides
    def forward(self,
                epoch_num: int,
                tokens: Dict[str, torch.Tensor],
                label: torch.IntTensor=None):  # pylint: disable=W0221
        """
        Given tokens and labels, generate document representation with
        a latent code and classify.
        """

        # run VAE to decode with a latent code
        vae_output = self._vae(tokens=tokens,
                               epoch_num=epoch_num,
                               )
        mask = get_text_field_mask(tokens)
        elbo = vae_output['elbo']
        self.metrics["elbo"](elbo.mean())
        self.metrics['perp'](vae_output['decoder_output']['flattened_decoder_output'],
                             tokens['tokens'].view(-1), mask)
        self.metrics["kld"](vae_output['kld'].mean())
        self.metrics["nll"](vae_output['nll'].mean())
        vae_output['loss'] = vae_output['elbo']
        return vae_output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: float(metric.get_metric(reset)) for metric_name, metric in self.metrics.items()}
