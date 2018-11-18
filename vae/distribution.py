import torch
from allennlp.common import Registrable
from overrides import overrides


class Distribution(Registrable, torch.nn.Module):
    default_implementation = 'normal'

    def set_priors(self):
        raise NotImplementedError
    
    def set_param_nets(self):
        raise NotImplementedError
        
    def estimate_parameters(self, input_repr):
        output = {}
        for param, net in self.param_nets.items():
            output[param] = net(input_repr)
        return output

    def compute_KLD(self, posteriors, priors=None, kl_weight=1.0):
        raise NotImplementedError


@Distribution.register("normal")
class Normal(Distribution):

    def __init__(self, hidden_dim, latent_dim):
        super(Normal, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.param_nets = self.set_param_nets()
    
    @overrides
    def set_priors(self):
        return {"mean": 0, "var": 0}

    @overrides
    def set_param_nets(self):
        return {"mean": torch.nn.Linear(self.hidden_dim, self.latent_dim).cuda(),
                "logvar": torch.nn.Linear(self.hidden_dim, self.latent_dim).cuda()}

    @overrides
    def compute_KLD(self, posteriors, priors=None, kl_weight=1.0):
        mean = posteriors['mean']
        logvar = posteriors['logvar']

        kld = -0.5 * torch.sum(1 - torch.mul(mean, mean) +
                               2 * logvar - torch.exp(2 * logvar), dim=1)
        return kl_weight * kld


@Distribution.register("logistic_normal")
class LogisticNormal(Distribution):

    def __init__(self, alpha, hidden_dim, latent_dim):
        super(LogisticNormal, self).__init__()
        self._alpha = alpha
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.param_nets = self.set_param_nets()

    @overrides
    def set_priors(self):
        alpha = self._alpha * torch.ones((self.lat_dim, 1))
        log_alpha = alpha.log()

        prior_mean = (log_alpha.transpose(0, 1) - torch.mean(log_alpha, 1)).transpose(0, 1)
        prior_var = (((1 / alpha) * (1 - (2.0 / self.lat_dim))).transpose(0, 1) +
                    (1.0 / (self.lat_dim**2) * torch.sum(1 / alpha, 1))).transpose(0, 1)
        prior_mean = prior_mean.transpose(0, 1)
        prior_var = prior_var.transpose(0, 1)
        return {"mean": prior_mean, "var": prior_var}

    @overrides
    def set_param_nets(self):
        return {"mean": torch.nn.Linear(self.hidden_dim, self.latent_dim).cuda(),
                "logvar": torch.nn.Linear(self.hidden_dim, self.latent_dim).cuda()}

    @overrides
    def compute_KLD(self, posteriors, priors, kl_weight=1.0):
        posterior_mean = posteriors['mean']
        posterior_logvar = posteriors['logvar']
        prior_var = priors['var']
        prior_mean = priors['mean']
        var_div = posterior_logvar.exp() / prior_var
        diff = posterior_mean - prior_mean
        diff_term = diff * diff / prior_var
        logvar_div = prior_var.log() - posterior_logvar
        neg_kl_divergence = (var_div + diff_term + logvar_div).sum(dim=1) - self.lat_dim
        neg_kl_divergence = 0.5 * kl_weight * neg_kl_divergence 
        return neg_kl_divergence