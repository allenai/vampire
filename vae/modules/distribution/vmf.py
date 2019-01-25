import torch
from overrides import overrides
from scipy import special as sp
import numpy as np
from allennlp.modules import FeedForward
from vae.modules.distribution.distribution import Distribution


@Distribution.register("vmf")
class VMF(Distribution):
    """
    von Mises-Fisher distribution class with batch support and manual tuning kappa value.
    Implementation is derived from https://github.com/jiacheng-xu/vmf_vae_nlp.
    """
    def __init__(self,
                 kappa: int = 80,
                 apply_batchnorm: bool = False,
                 theta_dropout: float = 0.0,
                 theta_softmax: bool = False) -> None:
        super(VMF, self).__init__()
        self.kappa = kappa
        self._apply_batchnorm = apply_batchnorm
        self._theta_dropout = torch.nn.Dropout(theta_dropout)
        self._theta_softmax = theta_softmax

    @overrides
    # pylint: disable=arguments-differ, attribute-defined-outside-init
    def initialize_params(self, hidden_dim, latent_dim):
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
        mean = self.func_mean(input_repr)
        norm = torch.norm(mean, 2, 1, keepdim=True)
        mu_norm_sq_diff_from_one = torch.pow(torch.add(norm, -1), 2)
        redundant_norm = torch.sum(mu_norm_sq_diff_from_one, dim=1, keepdim=True)
        ret_dict['norm'] = torch.ones_like(mean)
        ret_dict['redundant_norm'] = redundant_norm

        mean = mean / torch.norm(mean, p=2, dim=1, keepdim=True)
        if self._apply_batchnorm:
            mean = self.mean_bn(mean)
        ret_dict['mu'] = mean

        return ret_dict

    @overrides
    # pylint: disable=unused-argument
    def compute_kld(self, params):
        return self.kld.float()

    @staticmethod
    # pylint: disable=no-member
    def _vmf_kld(k_term, d_term):
        term1 = sp.iv(d_term / 2.0 + 1.0, k_term) + sp.iv(d_term / 2.0, k_term) * d_term / (2.0 * k_term)
        term2 = d_term / (2.0 * k_term)
        term3 = sp.iv(d_term / 2.0, k_term)
        term4 = d_term * np.log(k_term) / 2.0
        term5 = np.log(sp.iv(d_term / 2.0, k_term))  # pylint: disable=assignment-from-no-return
        term6 = sp.loggamma(d_term / 2 + 1)
        term7 = d_term * np.log(2) / 2
        tmp = (k_term * (term1 / term3 - term2) + term4 - term5 - term6 - term7).real
        return np.array([tmp])

    @staticmethod
    # pylint: disable=no-member
    def _vmf_kld_davidson(k_term, d_term):
        """
        This should be the correct KLD.
        Empirically we find that _vmf_kld (as in the Guu paper)
        only deviates a little (<2%) in most cases we use.
        """
        term1 = k_term * sp.iv(d_term / 2, k_term) / sp.iv(d_term / 2 - 1, k_term)
        term2 = (d_term / 2 - 1) * torch.log(k_term)
        term3 = torch.log(sp.iv(d_term / 2 - 1, k_term))
        term4 = np.log(np.pi) * d_term / 2
        term5 = np.log(2) - sp.loggamma(d_term / 2).real
        term6 = (d_term / 2) * np.log(2 * np.pi)
        tmp = term1 + term2 - term3 + term4 + term5 - term6
        return np.array([tmp])

    @overrides
    def generate_latent_code(self, input_repr, n_sample, training):
        params = self.estimate_param(input_repr=input_repr)
        mean = params['mu']
        kappa = params['kappa']
        kld = self.compute_kld(params)
        theta = self.sample_cell(mean, kappa)
        theta = self._theta_dropout(theta)
        if self._theta_softmax:
            theta = torch.nn.functional.softmax(theta, dim=-1)
        return params, kld, theta

    def sample_cell(self, mean, kappa):
        batch_sz, latent_dim = mean.size()
        mean = mean / torch.norm(mean, p=2, dim=1, keepdim=True)
        weight_batch = self._sample_weight_batch(kappa, latent_dim, batch_sz)
        weight_batch = weight_batch.unsqueeze(1)
        weight_batch_var = (weight_batch * torch.ones(batch_sz, latent_dim)).to(mean.device)
        ortho_batch = self._sample_ortho_batch(mean, latent_dim)
        scale_factr = torch.sqrt(
                torch.ones(batch_sz, latent_dim).to(mean.device) - torch.pow(weight_batch_var, 2))
        orth_term = ortho_batch * scale_factr
        muscale = mean * weight_batch_var
        sampled_vec = orth_term + muscale
        return sampled_vec

    def _sample_weight_batch(self, kappa, dim, batch_sz=1):
        result = torch.FloatTensor((batch_sz))
        for batch in range(batch_sz):
            result[batch] = self._sample_weight(kappa, dim)
        return result

    # pylint: disable=no-self-use
    def _sample_weight(self, kappa, dim):
        """Rejection sampling scheme for sampling distance from center on
        surface of the sphere.
        """
        dim = dim - 1  # since S^{n-1}
        term1 = dim / (np.sqrt(4. * kappa ** 2 + dim ** 2) + 2 * kappa)  # b= 1/(sqrt(4.* kdiv**2 + 1) + 2 * kdiv)
        term2 = (1. - term1) / (1. + term1)
        term3 = kappa * term2 + dim * np.log(1 - term2 ** 2)  # dim * (kdiv *x + np.log(1-x**2))

        while True:
            term4 = np.random.beta(dim / 2., dim / 2.)  # concentrates towards 0.5 as d-> inf
            term5 = (1. - (1. + term1) * term4) / (1. - (1. - term1) * term4)
            term6 = np.random.uniform(low=0, high=1)
            if kappa * term5 + dim * np.log(1. - term2 * term5) - term3 >= np.log(term6):
                # thresh is dim *(kdiv * (w-x) + log(1-x*w) -log(1-x**2))
                return term5
        return None

    # pylint: disable=no-self-use
    def _sample_ortho_batch(self, mean, dim):
        """
        :param mu: Variable, [batch size, latent dim]
        :param dim: scala. =latent dim
        :return:
        """
        _batch_sz, _latent_dim = mean.size()
        assert _latent_dim == dim
        squeezed_mean = mean.unsqueeze(1)
        term1 = torch.randn(_batch_sz, dim, 1).to(mean.device)  # TODO random
        rescale_val = torch.bmm(squeezed_mean, term1).squeeze(2)
        proj_mu_v = mean * rescale_val
        ortho = term1.squeeze() - proj_mu_v
        ortho_norm = torch.norm(ortho, p=2, dim=1, keepdim=True)
        term2 = ortho / ortho_norm
        return term2

    # pylint: disable=no-self-use
    def _sample_orthonormal_to(self, mean, dim):
        """Sample point on sphere orthogonal to mu.
        """
        term1 = torch.randn(dim).to(mean.device)  # TODO random
        rescale_value = mean.dot(term1) / mean.norm()
        proj_mu_v = mean * rescale_value.expand(dim)
        ortho = term1 - proj_mu_v
        ortho_norm = torch.norm(ortho)
        return ortho / ortho_norm.expand_as(ortho)
