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
    default_implementation = 'gaussian'

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
