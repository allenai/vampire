import torch
from allennlp.common import Registrable
from overrides import overrides
from scipy import special as sp
import numpy as np
from allennlp.modules import FeedForward
from typing import Dict, Tuple
from vae.modules.distribution.distribution import Distribution



@Distribution.register("vmf")
class VMF(Distribution):
    """
    von Mises-Fisher distribution class with batch support and manual tuning kappa value.
    Implementation is derived from https://github.com/jiacheng-xu/vmf_vae_nlp.
    """
    def __init__(self, kappa: int=80, apply_batchnorm: bool=False, theta_dropout: float=0.0, theta_softmax: bool=False) -> None:
        super(VMF, self).__init__()
        self.kappa = kappa
        self._apply_batchnorm = apply_batchnorm
        self._theta_dropout = torch.nn.Dropout(theta_dropout)
        self._theta_softmax = theta_softmax
        
    @overrides
    def _initialize_params(self, hidden_dim, latent_dim):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        softplus = torch.nn.Softplus()
        self.func_mean = FeedForward(input_dim=self.hidden_dim,
                                     num_layers=1,
                                     hidden_dims=self.latent_dim,
                                     activations=softplus)
        if self._apply_batchnorm:
            self.mean_bn = torch.nn.BatchNorm1d(latent_dim, eps=0.001, momentum=0.001, affine=True)
            self.mean_bn.weight.data.copy_(torch.ones(latent_dim))
            self.mean_bn.weight.requires_grad = False
        self.kld = torch.from_numpy(self._vmf_kld(self.kappa, latent_dim))


    @overrides
    def estimate_param(self, input_repr):
        ret_dict = {}
        ret_dict['kappa'] = self.kappa

        # Only compute mu, use mu/mu_norm as mu,
        #  use 1 as norm, use diff(mu_norm, 1) as redundant_norm
        mu = self.func_mean(input_repr)

        norm = torch.norm(mu, 2, 1, keepdim=True)
        mu_norm_sq_diff_from_one = torch.pow(torch.add(norm, -1), 2)
        redundant_norm = torch.sum(mu_norm_sq_diff_from_one, dim=1, keepdim=True)
        ret_dict['norm'] = torch.ones_like(mu)
        ret_dict['redundant_norm'] = redundant_norm

        mu = mu / torch.norm(mu, p=2, dim=1, keepdim=True)
        if self._apply_batchnorm:
            mu = self.mean_bn(mu)
        ret_dict['mu'] = mu

        return ret_dict

    @overrides
    def compute_KLD(self, params, batch_sz):
        return self.kld.float()

    @staticmethod
    def _vmf_kld(k, d):
        tmp = (k * ((sp.iv(d / 2.0 + 1.0, k) + sp.iv(d / 2.0, k) * d / (2.0 * k)) / sp.iv(d / 2.0, k) - d / (2.0 * k)) \
               + d * np.log(k) / 2.0 - np.log(sp.iv(d / 2.0, k)) \
               - sp.loggamma(d / 2 + 1) - d * np.log(2) / 2).real
        if tmp != tmp:
            exit()
        return np.array([tmp])
    
    @staticmethod
    def _vmf_kld_davidson(k, d):
        """
        This should be the correct KLD.
        Empirically we find that _vmf_kld (as in the Guu paper)
        only deviates a little (<2%) in most cases we use.
        """
        tmp = k * sp.iv(d / 2, k) / sp.iv(d / 2 - 1, k) + (d / 2 - 1) * torch.log(k) - torch.log(
            sp.iv(d / 2 - 1, k)) + np.log(np.pi) * d / 2 + np.log(2) - sp.loggamma(d / 2).real - (d / 2) * np.log(
            2 * np.pi)
        if tmp != tmp:
            exit()
        return np.array([tmp])

    @overrides
    def generate_latent_code(self, input_repr, n_sample):
        batch_sz = input_repr.size()[0]
        params = self.estimate_param(input_repr=input_repr)
        mu = params['mu']
        norm = params['norm']
        kappa = params['kappa']
        kld = self.compute_KLD(params, batch_sz)
        theta = self.sample_cell(mu, norm, kappa)
        theta = self._theta_dropout(theta)
        if self._theta_softmax:
            theta = torch.nn.functional.softmax(theta, dim=-1)
        return params, kld, theta

    @overrides
    def sample_cell(self, mu, norm, kappa):
        batch_sz, latent_dim = mu.size()
        mu = mu / torch.norm(mu, p=2, dim=1, keepdim=True)
        w = self._sample_weight_batch(kappa, latent_dim, batch_sz)
        w = w.unsqueeze(1)
        
        w_var = (w * torch.ones(batch_sz, latent_dim)).to(mu.device)
        v = self._sample_ortho_batch(mu, latent_dim)
        scale_factr = torch.sqrt(
            torch.ones(batch_sz, latent_dim).to(mu.device) - torch.pow(w_var, 2))
        orth_term = v * scale_factr
        muscale = mu * w_var
        sampled_vec = orth_term + muscale
        return sampled_vec

    def _sample_weight_batch(self, kappa, dim, batch_sz=1):
        result = torch.FloatTensor((batch_sz))
        for b in range(batch_sz):
            result[b] = self._sample_weight(kappa, dim)
        return result

    def _sample_weight(self, kappa, dim):
        """Rejection sampling scheme for sampling distance from center on
        surface of the sphere.
        """
        dim = dim - 1  # since S^{n-1}
        b = dim / (np.sqrt(4. * kappa ** 2 + dim ** 2) + 2 * kappa)  # b= 1/(sqrt(4.* kdiv**2 + 1) + 2 * kdiv)
        x = (1. - b) / (1. + b)
        c = kappa * x + dim * np.log(1 - x ** 2)  # dim * (kdiv *x + np.log(1-x**2))

        while True:
            z = np.random.beta(dim / 2., dim / 2.)  # concentrates towards 0.5 as d-> inf
            w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
            u = np.random.uniform(low=0, high=1)
            if kappa * w + dim * np.log(1. - x * w) - c >= np.log(
                    u):  # thresh is dim *(kdiv * (w-x) + log(1-x*w) -log(1-x**2))
                return w

    def _sample_ortho_batch(self, mu, dim):
        """
        :param mu: Variable, [batch size, latent dim]
        :param dim: scala. =latent dim
        :return:
        """
        _batch_sz, _latent_dim = mu.size()
        assert _latent_dim == dim
        squeezed_mu = mu.unsqueeze(1)

        v = torch.randn(_batch_sz, dim, 1).to(mu.device)  # TODO random
        rescale_val = torch.bmm(squeezed_mu, v).squeeze(2)
        proj_mu_v = mu * rescale_val
        ortho = v.squeeze() - proj_mu_v
        ortho_norm = torch.norm(ortho, p=2, dim=1, keepdim=True)
        y = ortho / ortho_norm
        return y

    def _sample_orthonormal_to(self, mu, dim):
        """Sample point on sphere orthogonal to mu.
        """
        v = torch.randn(dim).to(mu.device)  # TODO random
        rescale_value = mu.dot(v) / mu.norm()
        proj_mu_v = mu * rescale_value.expand(dim)
        ortho = v - proj_mu_v
        ortho_norm = torch.norm(ortho)
        return ortho / ortho_norm.expand_as(ortho)