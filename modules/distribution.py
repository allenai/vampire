import torch
from allennlp.common import Registrable
from overrides import overrides
from scipy import special as sp
import numpy as np
from allennlp.modules import FeedForward
from typing import Dict, Tuple


class Distribution(Registrable, torch.nn.Module):
    """
    Standard implementations of VAE distributions
    Mostly taken from https://github.com/jiacheng-xu/vmf_vae_nlp
    """
    default_implementation = 'normal'

    def _initialize_params(self, input_dim, latent_dim):
        raise NotImplementedError

    def estimate_param(self, input_repr):
        """
        estimate the parameters of distribution given an input representation
        """
        raise NotImplementedError

    def compute_KLD(self, params):
        """
        compute the KL divergence given posteriors
        """
        raise NotImplementedError

    def sample_cell(self, batch_size):
        """
        sample noise for reparameterization
        """
        raise NotImplementedError

    def generate_latent_code(self, input_repr, n_sample):
        """
        generate latent code from input representation
        """
        raise NotImplementedError


@Distribution.register("normal")
class Normal(Distribution):

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
        super(Normal, self).__init__()
        
    @overrides
    def _initialize_params(self, hidden_dim, latent_dim):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        softplus = torch.nn.Softplus()
        self.func_mean = FeedForward(input_dim=hidden_dim,
                                     num_layers=1,
                                     hidden_dims=latent_dim,
                                     activations=softplus)
        self.func_logvar = FeedForward(input_dim=hidden_dim,
                                       num_layers=1,
                                       hidden_dims=latent_dim,
                                       activations=softplus)

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
        kld = -0.5 * torch.sum(1 - torch.mul(mean, mean) +
                               2 * logvar - torch.exp(2 * logvar), dim=1)
        return kld

    @overrides
    def sample_cell(self, batch_size: int) -> torch.FloatTensor:
        """
        sample noise for reparameterization
        """
        eps = torch.autograd.Variable(torch.normal(torch.zeros((batch_size, self.latent_dim))))
        if torch.cuda.is_available():
            eps = eps.cuda()
        return eps.unsqueeze(0)

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
        batch_sz = input_repr.size()[0]
        params = self.estimate_param(input_repr=input_repr)
        mean = params['mean']
        logvar = params['logvar']

        kld = self.compute_KLD(params)
        if n_sample == 1:
            eps = self.sample_cell(batch_size=batch_sz)
            theta = torch.mul(torch.exp(logvar), eps) + mean
            return params, kld, theta

        theta = []
        for ns in range(n_sample):
            eps = self.sample_cell(batch_size=batch_sz)
            vec = torch.mul(torch.exp(logvar), eps) + mean
            theta.append(vec)
        theta = torch.cat(theta, dim=0)
        return params, kld, theta


@Distribution.register("logistic_normal")
class LogisticNormal(Distribution):

    def __init__(self, alpha: float=1.0) -> None:
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
        

    @overrides
    def _initialize_params(self, hidden_dim, latent_dim):
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
        log_alpha = self.alpha.log()
        self.prior_mean = (log_alpha.transpose(0, 1) - torch.mean(log_alpha, 1)).cuda()
        self.prior_var = (((1 / self.alpha) * (1 - (2.0 / self.latent_dim))).transpose(0, 1) +
                          (1.0 / (self.latent_dim**2) * torch.sum(1 / self.alpha, 1))).cuda()

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
        return {'mean': mean, 'logvar': logvar}

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
        var_div = logvar.exp() / self.prior_var.to(logvar.device)
        diff = mean - self.prior_mean.to(mean.device)
        diff_term = diff * diff / self.prior_var.to(logvar.device)
        logvar_div = self.prior_var.log().to(logvar.device) - logvar
        kld = 0.5 * (var_div + diff_term + logvar_div).sum(dim=1) - self.latent_dim
        return kld

    @overrides
    def sample_cell(self, batch_size: int) -> torch.FloatTensor:
        """
        sample noise for reparameterization
        """
        eps = torch.autograd.Variable(torch.normal(torch.zeros((batch_size, self.latent_dim))))
        if torch.cuda.is_available():
             eps = eps.cuda()
        return eps.unsqueeze(0)

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
        batch_sz = input_repr.size()[0]
        params = self.estimate_param(input_repr=input_repr)
        mean = params['mean']
        logvar = params['logvar']

        kld = self.compute_KLD(params)
        if n_sample == 1:
            eps = self.sample_cell(batch_size=batch_sz)
            theta = torch.mul(torch.exp(logvar), eps.to(logvar.device)) + mean.to(logvar.device)
            return params, kld, theta

        theta = []
        for ns in range(n_sample):
            eps = self.sample_cell(batch_size=batch_sz)
            vec = torch.mul(torch.exp(logvar), eps) + mean
            theta.append(vec)
        theta = torch.cat(theta, dim=0)
        return params, kld, theta


@Distribution.register("vmf")
class VMF(Distribution):
    """
    von Mises-Fisher distribution class with batch support and manual tuning kappa value.
    Implementation is copy and pasted from https://github.com/jiacheng-xu/vmf_vae_nlp.
    """
    def __init__(self, kappa: int=80):
        super(VMF, self).__init__()
        self.kappa = kappa
        
    @overrides
    def _initialize_params(self, hidden_dim, latent_dim):
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        softplus = torch.nn.Softplus()
        self.func_mean = FeedForward(input_dim=self.hidden_dim,
                                     num_layers=1,
                                     hidden_dims=self.latent_dim,
                                     activations=softplus)
        self.kld = torch.from_numpy(VMF._vmf_kld(self.kappa, latent_dim))


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
        vecs = []
        if n_sample == 1:
            return params, kld, self.sample_cell(mu, norm, kappa)
        for n in range(n_sample):
            sample = self.sample_cell(mu, norm, kappa)
            vecs.append(sample)
        vecs = torch.cat(vecs, dim=0)
        return params, kld, vecs

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
        return sampled_vec.unsqueeze(0)

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

        # v = GVar(torch.linspace(-1, 1, steps=dim))
        # v = v.expand(_batch_sz, dim).unsqueeze(2)

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

        # v = GVar(torch.linspace(-1,1,steps=dim))

        rescale_value = mu.dot(v) / mu.norm()
        proj_mu_v = mu * rescale_value.expand(dim)
        ortho = v - proj_mu_v
        ortho_norm = torch.norm(ortho)
        return ortho / ortho_norm.expand_as(ortho)
