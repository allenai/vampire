from typing import Dict, Tuple
import torch
from overrides import overrides
from allennlp.modules import FeedForward
from vae.modules.distribution.distribution import Distribution


@Distribution.register("logistic_normal")
class LogisticNormal(Distribution):

    def __init__(self,
                 alpha: float = 1.0,
                 apply_batchnorm: bool = False,
                 theta_dropout: float = 0.0,
                 theta_softmax: bool = False) -> None:
        """
        Logistic Normal distribution prior

        Params
        ______
        hidden_dim : ``int``
            hidden dimension of VAE
        latent_dim : ``int``
            latent dimension of VAE
        func_mean: ``FeedForward``
            Network parameterizing mean of normal distribution
        func_logvar: ``FeedForward``
            Network parameterizing log variance of normal distribution
        """
        super(LogisticNormal, self).__init__()
        self.alpha = alpha
        self._apply_batchnorm = apply_batchnorm
        self._theta_dropout = torch.nn.Dropout(theta_dropout)
        self._theta_softmax = theta_softmax

    @overrides
    # pylint: disable=arguments-differ, attribute-defined-outside-init
    def initialize_params(self, hidden_dim, latent_dim):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.alpha = torch.ones((self.latent_dim, 1)) * self.alpha
        softplus = torch.nn.Softplus()
        self.func_mean = FeedForward(input_dim=hidden_dim,
                                     num_layers=1,
                                     hidden_dims=latent_dim,
                                     activations=softplus)
        self.func_logvar = FeedForward(input_dim=hidden_dim,
                                       num_layers=1,
                                       hidden_dims=latent_dim,
                                       activations=softplus)
        if self._apply_batchnorm:
            self.mean_bn = torch.nn.BatchNorm1d(latent_dim, eps=0.001, momentum=0.001, affine=True)
            self.mean_bn.weight.data.copy_(torch.ones(latent_dim))
            self.mean_bn.weight.requires_grad = False
            self.logvar_bn = torch.nn.BatchNorm1d(latent_dim, eps=0.001, momentum=0.001, affine=True)
            self.logvar_bn.weight.data.copy_(torch.ones(latent_dim))
            self.logvar_bn.weight.requires_grad = False
        log_alpha = self.alpha.log()
        self.prior_mean = (log_alpha.transpose(0, 1) - torch.mean(log_alpha, 1)).to(self.alpha.device)
        self.prior_var = (((1 / self.alpha) * (1 - (2.0 / self.latent_dim))).transpose(0, 1) +
                          (1.0 / (self.latent_dim**2) * torch.sum(1 / self.alpha, 1))).to(self.alpha.device)

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
        if self._apply_batchnorm:
            mean = self.mean_bn(mean)
            logvar = self.logvar_bn(logvar)
        return {'mean': mean, 'logvar': logvar}

    @overrides
    def compute_kld(self, params: Dict[str, torch.Tensor]) -> float:
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
        var_div = logvar.exp() / self.prior_var.to(logvar.device)
        diff = mean - self.prior_mean.to(mean.device)
        diff_term = diff * diff / self.prior_var.to(logvar.device)
        logvar_div = self.prior_var.log().to(logvar.device) - logvar
        kld = 0.5 * (var_div + diff_term + logvar_div).sum() - self.latent_dim
        return kld

    @overrides
    def generate_latent_code(self,
                             input_repr: torch.FloatTensor,
                             n_sample: int,
                             training: bool) -> Tuple[Dict[str, torch.FloatTensor],
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
        params = self.estimate_param(input_repr=input_repr)
        mean = params['mean']
        logvar = params['logvar']
        kld = self.compute_kld(params)
        eps = torch.randn(mean.shape)
        if training:
            theta = mean.to(logvar.device) + logvar.exp().sqrt() * eps.to(logvar.device)
        else:
            theta = mean.to(logvar.device)
        theta = self._theta_dropout(theta)
        if self._theta_softmax:
            theta = torch.nn.functional.softmax(theta, dim=-1)
        return params, kld, theta

    def forward(self, *inputs):
        """
        generate latent code from input representation
        """
