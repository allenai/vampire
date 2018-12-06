import torch
from allennlp.common import Registrable
from overrides import overrides
from scipy import special as sp
import numpy as np
from allennlp.modules import FeedForward
from typing import Dict, Tuple
from modules.distribution.distribution import Distribution


@Distribution.register("gaussian")
class Gaussian(Distribution):

    def __init__(self) -> None:
        """
        Normal distribution prior

        Params
        ______
        hidden_dim : ``int``
            hidden dimension of VAE
        latent_dim : ``int``
            latent dimension of VAE
        """
        super(Gaussian, self).__init__()
        
    @overrides
    def _initialize_params(self, hidden_dim, latent_dim):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.func_mean = FeedForward(input_dim=hidden_dim,
                                     num_layers=1,
                                     hidden_dims=latent_dim,
                                     activations=lambda x: x)
        self.func_logvar = FeedForward(input_dim=hidden_dim,
                                       num_layers=1,
                                       hidden_dims=latent_dim,
                                       activations=lambda x: x)

    @overrides
    def estimate_param(self, input_repr: torch.FloatTensor):
        """
        estimate the parameters of distribution given an input representation

        Params
        ______

        input_repr: ``torch.FloatTensor``
            input representation

        Returns
        _______
        params : ``Dict[str, torch.Tensor]``
            estimated parameters after feedforward projection
        """
        mean = self.func_mean(input_repr)
        logvar = self.func_logvar(input_repr)
        params = {'mean': mean, 'logvar': logvar}
        return params

    @overrides
    def compute_KLD(self, params: Dict[str, torch.Tensor]) -> float:
        """
        Compute the KL Divergence of Normal distribution given estimated parameters

        Params
        ______

        params : ``Dict[str, torch.Tensor]``
            estimated parameters after feedforward projection

        Returns
        _______
        kld : ``float``
            KL divergence
        """
        mean = params['mean']
        logvar = params['logvar']
        var = torch.exp(logvar)
        kld = 1 + logvar - mean ** 2 - var
        kld = -0.5 * torch.sum(kld)
        return kld

    @overrides
    def generate_latent_code(self,
                             input_repr: torch.FloatTensor,
                             n_sample: int) -> Tuple[Dict[str, torch.FloatTensor],
                                                     float,
                                                     torch.FloatTensor]:
        """
        Generate latent code from input representation

        Params
        ______

        input_repr : ``torch.Tensor``
            input representation

        n_sample: ``int``
            number of times to sample noise

        Returns
        _______
        params : ``Dict[str, torch.Tensor]``
            estimated parameters after feedforward projection

        kld : ``float``
            KL divergence

        theta : ``Dict[str, torch.Tensor]``
            latent code
        """
        batch_size = input_repr.size()[0]
        params = self.estimate_param(input_repr=input_repr)
        mean = params['mean']
        logvar = params['logvar']
        kld = self.compute_KLD(params)
        eps = torch.randn([batch_size, self.latent_dim])
        sigma = torch.exp(0.5 * logvar)
        theta = torch.mul(sigma, eps.to(logvar.device)) + mean.to(logvar.device)
        return params, kld, theta