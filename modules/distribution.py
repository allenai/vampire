import torch
from allennlp.common import Registrable
from overrides import overrides
from scipy import special as sp
import numpy as np
from allennlp.modules import FeedForward


class Distribution(Registrable, torch.nn.Module):
    default_implementation = 'normal'

    def estimate_param(self, input_repr):
        raise NotImplementedError

    def compute_KLD(self, tup):
        raise NotImplementedError

    def sample_cell(self, batch_size):
        raise NotImplementedError

    def generate_latent_repr(self, input_repr, n_sample):
        raise NotImplementedError

@Distribution.register("normal")
class Normal(Distribution):
    # __slots__ = ['latent_dim', 'logvar', 'mean']

    def __init__(self, hidden_dim, latent_dim, func_mean: FeedForward, func_logvar: FeedForward):
        super(Normal, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.func_mean = func_mean
        self.func_logvar = func_logvar

    @overrides
    def estimate_param(self, input_repr):
        mean = self.func_mean(input_repr)
        logvar = self.func_logvar(input_repr)
        return {'mean': mean, 'logvar': logvar}

    @overrides
    def compute_KLD(self, tup):
        mean = tup['mean']
        logvar = tup['logvar']
        kld = -0.5 * torch.sum(1 - torch.mul(mean, mean) +
                               2 * logvar - torch.exp(2 * logvar), dim=1)
        return kld

    @overrides
    def sample_cell(self, batch_size):
        eps = torch.autograd.Variable(torch.normal(torch.zeros((batch_size, self.latent_dim))))
        if torch.cuda.is_available():
             eps = eps.cuda()
        return eps.unsqueeze(0)

    @overrides
    def generate_latent_repr(self, input_repr, n_sample):
        batch_sz = input_repr.size()[0]
        tup = self.estimate_param(input_repr=input_repr)
        mean = tup['mean']
        logvar = tup['logvar']

        kld = self.compute_KLD(tup)
        if n_sample == 1:
            eps = self.sample_cell(batch_size=batch_sz)
            vec = torch.mul(torch.exp(logvar), eps) + mean
            return tup, kld, vec

        vecs = []
        for ns in range(n_sample):
            eps = self.sample_cell(batch_size=batch_sz)
            vec = torch.mul(torch.exp(logvar), eps) + mean
            vecs.append(vec)
        vecs = torch.cat(vecs, dim=0)
        return tup, kld, vecs


@Distribution.register("logistic_normal")
class LogisticNormal(Distribution):
    # __slots__ = ['latent_dim', 'logvar', 'mean']

    def __init__(self, hidden_dim, latent_dim, func_mean: FeedForward, func_logvar: FeedForward, alpha=1.0):
        super(LogisticNormal, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.alpha = torch.ones((self.latent_dim, 1)) * alpha
        self.func_mean = func_mean
        self.func_logvar = func_logvar
        log_alpha = self.alpha.log()
        self.prior_mean = (log_alpha.transpose(0, 1) - torch.mean(log_alpha, 1)).cuda()
        self.prior_var = (((1 / self.alpha) * (1 - (2.0 / self.latent_dim))).transpose(0, 1) +
                         (1.0 / (self.latent_dim**2) * torch.sum(1 / self.alpha, 1))).cuda()
    @overrides
    def estimate_param(self, input_repr):
        mean = self.func_mean(input_repr)
        logvar = self.func_logvar(input_repr)
        return {'mean': mean, 'logvar': logvar}

    @overrides
    def compute_KLD(self, tup):
        mean = tup['mean']
        logvar = tup['logvar']
        var_div = logvar.exp() / self.prior_var.to(logvar.device)
        diff = mean - self.prior_mean.to(mean.device)
        diff_term = diff * diff / self.prior_var.to(logvar.device)
        logvar_div = self.prior_var.log().to(logvar.device) - logvar
        kld = 0.5 * (var_div + diff_term + logvar_div).sum(dim=1) - self.latent_dim
        return kld

    @overrides
    def sample_cell(self, batch_size):
        eps = torch.autograd.Variable(torch.normal(torch.zeros((batch_size, self.latent_dim))))
        if torch.cuda.is_available():
             eps = eps.cuda()
        return eps.unsqueeze(0)

    @overrides
    def generate_latent_repr(self, input_repr, n_sample):
        batch_sz = input_repr.size()[0]
        tup = self.estimate_param(input_repr=input_repr)
        mean = tup['mean']
        logvar = tup['logvar']

        kld = self.compute_KLD(tup)
        if n_sample == 1:
            eps = self.sample_cell(batch_size=batch_sz)
            vec = torch.mul(torch.exp(logvar), eps.to(logvar.device)) + mean.to(logvar.device)
            return tup, kld, vec

        vecs = []
        for ns in range(n_sample):
            eps = self.sample_cell(batch_size=batch_sz)
            vec = torch.mul(torch.exp(logvar), eps) + mean
            vecs.append(vec)
        vecs = torch.cat(vecs, dim=0)
        return tup, kld, vecs

@Distribution.register("vmf")
class VMF(Distribution):
    def __init__(self, hidden_dim, latent_dim, func_mean: FeedForward, kappa=80):
        """
        von Mises-Fisher distribution class with batch support and manual tuning kappa value.
        Implementation follows description of my paper and Guu's.
        """

        super(vMF, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.kappa = kappa
        # self.func_kappa = torch.nn.Linear(hidden_dim, latent_dim)
        self.func_mean = func_mean
        self.kld = torch.from_numpy(vMF._vmf_kld(kappa, latent_dim))
        
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
    def compute_KLD(self, tup, batch_sz):
        return self.kld.expand(batch_sz).float().cuda()

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
        Empirically we find that _vmf_kld (as in the Guu paper) only deviates a little (<2%) in most cases we use.
        """
        tmp = k * sp.iv(d / 2, k) / sp.iv(d / 2 - 1, k) + (d / 2 - 1) * torch.log(k) - torch.log(
            sp.iv(d / 2 - 1, k)) + np.log(np.pi) * d / 2 + np.log(2) - sp.loggamma(d / 2).real - (d / 2) * np.log(
            2 * np.pi)
        if tmp != tmp:
            exit()
        return np.array([tmp])

    @overrides
    def generate_latent_repr(self, input_repr, n_sample):
        batch_sz = input_repr.size()[0]
        tup = self.estimate_param(input_repr=input_repr)
        mu = tup['mu']
        norm = tup['norm']
        kappa = tup['kappa']
        kld = self.compute_KLD(tup, batch_sz)
        vecs = []
        if n_sample == 1:
            return tup, kld, self.sample_cell(mu, norm, kappa)
        for n in range(n_sample):
            sample = self.sample_cell(mu, norm, kappa)
            vecs.append(sample)
        vecs = torch.cat(vecs, dim=0)
        return tup, kld, vecs

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