import torch
from allennlp.models import Model


class VAE(Model):

    def __init__(self, vocab):
        super(VAE, self).__init__(vocab)

    def estimate_params(self, input_repr):
        """
        Estimate the parameters of distribution given an input representation

        Parameters
        ----------
        input_repr: ``torch.FloatTensor``
            input representation

        Returns
        -------
        params : ``Dict[str, torch.Tensor]``
            estimated parameters after feedforward projection
        """
        raise NotImplementedError

    def compute_negative_kld(self, params):
        """
        Compute the KL divergence given posteriors.
        """
        raise NotImplementedError

    def generate_latent_code(self, input_repr: torch.Tensor):  # pylint: disable=W0221
        """
        Given an input representation, produces the latent variables from the VAE.

        Parameters
        ----------
        input_repr : ``torch.Tensor``
            Input in which the VAE will use to re-create the original text.
            This can either be x_enc (the latent representation of x after
            being encoded) or x_bow: the Bag-of-Word-Counts representation of x.

        Returns
        -------
        A ``Dict[str, torch.Tensor]`` containing
            theta:
                the latent variable produced by the VAE
            parameters:
                A dictionary containing the parameters produces by the
                distribution
            negative_kl_divergence:
                The negative KL=divergence specific to the distribution this
                VAE implements
        """
        raise NotImplementedError

    def get_beta(self):
        """
        Returns
        -------
        The topics x vocabulary tensor representing word strengths for each topic.
        """
        raise NotImplementedError

    def encode(self, input_vector: torch.Tensor):
        """
        Encode the input_vector to the VAE's internal representation.
        """
        raise NotImplementedError
