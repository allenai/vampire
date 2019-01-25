import torch
from allennlp.common import Registrable


class Distribution(Registrable, torch.nn.Module):
    """
    Standard implementations of VAE distributions
    Mostly taken from https://github.com/jiacheng-xu/vmf_vae_nlp
    """
    default_implementation = 'gaussian'

    def initialize_params(self, input_dim, latent_dim):
        raise NotImplementedError

    def estimate_param(self, input_repr):
        """
        estimate the parameters of distribution given an input representation
        """
        raise NotImplementedError

    def compute_kld(self, params):
        """
        compute the KL divergence given posteriors
        """
        raise NotImplementedError

    def generate_latent_code(self, input_repr, n_sample, training):
        """
        generate latent code from input representation
        """
        raise NotImplementedError

    def forward(self, *inputs):
        """
        generate latent code from input representation
        """
