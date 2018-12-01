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
            'recon': Average(),
            'nll': Average(),
            'elbo': Average(),
            'perp': Perplexity(),
        }
        self._num_labels = vocab.get_vocab_size("labels")
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
        vae_output = self._vae(tokens=tokens, epoch_num=epoch_num)
        mask = get_text_field_mask(tokens)
        u_recon = vae_output.get('u_recon', np.zeros(1))
        elbo = vae_output['elbo']
        generative_clf_loss = vae_output.get('generative_clf_loss',  np.zeros(1))
        u_kld = vae_output.get('u_kld', np.zeros(1))
        u_nll = vae_output.get('u_nll', np.zeros(1))
        self.metrics["recon"](u_recon.mean())
        self.metrics["elbo"](elbo.mean())
        self.metrics['perp'](vae_output['flattened_decoded_output'], tokens['tokens'].view(-1), mask)
        self.metrics["kld"](u_kld.mean())
        self.metrics["nll"](u_nll.mean())

        vae_output['loss'] = vae_output['elbo']
        
        return vae_output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: float(metric.get_metric(reset)) for metric_name, metric in self.metrics.items()}
