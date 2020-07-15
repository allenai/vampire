from typing import Dict, Optional, List, Tuple, Union
import os
import torch
from allennlp.modules import FeedForward
from overrides import overrides

from vampire.vampire.modules.vae.vae import VAE


@VAE.register("logistic_normal")
class LogisticNormal(VAE):
    """
    A Variational Autoencoder with a Logistic Normal prior
    """
    def __init__(self,
                 vocab,
                 encoder: FeedForward,
                 mean_projection: FeedForward,
                 log_variance_projection: FeedForward,
                 decoder: FeedForward,
                 kld_clamp: Union[str, float] = None,
                 z_dropout: Union[str, float] = 0.2) -> None:
        super(LogisticNormal, self).__init__(vocab)
        self.encoder = encoder
        self.mean_projection = mean_projection
        self.log_variance_projection = log_variance_projection
        self._kld_clamp = float(kld_clamp)
        self._decoder = torch.nn.Linear(decoder.get_input_dim(), decoder.get_output_dim(),
                                        bias=False)
        self._z_dropout = torch.nn.Dropout(float(z_dropout))

        self.latent_dim = mean_projection.get_output_dim()

    @overrides
    def forward(self, input_repr: torch.FloatTensor):  # pylint: disable = W0221
        """
        Given the input representation, produces the reconstruction from theta
        as well as the negative KL-divergence, theta itself, and the parameters
        of the distribution.
        """
        activations: List[Tuple[str, torch.FloatTensor]] = []
        intermediate_input = input_repr
        for layer_index, layer in enumerate(self.encoder._linear_layers):  # pylint: disable=protected-access
            intermediate_input = layer(intermediate_input)
            activations.append((f"encoder_layer_{layer_index}", intermediate_input))
        output = self.generate_latent_code(intermediate_input)
        theta = output["theta"]
        activations.append(('theta', theta))
        reconstruction = self._decoder(theta)
        output["reconstruction"] = reconstruction
        output['activations'] = activations

        return output

    @overrides
    def estimate_params(self, input_repr: torch.FloatTensor):
        """
        Estimate the parameters for the logistic normal.
        """
        mean = self.mean_projection(input_repr)  # pylint: disable=C0103
        log_var = self.log_variance_projection(input_repr)
        sigma = torch.sqrt(torch.exp(log_var)).clamp(max=10)  # log_var is actually log (variance^2).
        return {
                "mean": mean,
                "variance": sigma,
                "log_variance": log_var
                }

    @overrides
    def compute_negative_kld(self, params: Dict):
        """
        Compute the closed-form solution for negative KL-divergence for Gaussians.
        """
        mu, sigma = params["mean"], params["variance"]  # pylint: disable=C0103
        negative_kl_divergence = 1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2
        if self._kld_clamp:
            negative_kl_divergence = torch.clamp(negative_kl_divergence,
                                                 min=-1 * self._kld_clamp,
                                                 max=self._kld_clamp)
        negative_kl_divergence = 0.5 * negative_kl_divergence.sum(dim=-1)  # Shape: (batch, )
        return negative_kl_divergence

    @overrides
    def generate_latent_code(self, input_repr: torch.Tensor):
        """
        Given an input vector, produces the latent encoding z, followed by the
        mean and log variance of the variational distribution produced.

        z is the result of the reparameterization trick.
        (https://arxiv.org/abs/1312.6114)
        """
        params = self.estimate_params(input_repr)
        negative_kl_divergence = self.compute_negative_kld(params)
        mu, sigma = params["mean"], params["variance"]  # pylint: disable=C0103

        # Generate random noise and sample theta.
        # Shape: (batch, latent_dim)
        batch_size = params["mean"].size(0)

        # Enable reparameterization for training only.
        if self.training:
            seed = os.environ['SEED']
            torch.manual_seed(seed)
            # Seed all GPUs with the same seed if available.
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            epsilon = torch.randn(batch_size, self.latent_dim).to(device=mu.device)
            z = mu + sigma * epsilon  # pylint: disable=C0103
        else:
            z = mu  # pylint: disable=C0103

        # Apply dropout to theta.
        theta = self._z_dropout(z)

        # Normalize theta.
        theta = torch.softmax(theta, dim=-1)

        return {
                "theta": theta,
                "params": params,
                "negative_kl_divergence": negative_kl_divergence
                }

    @overrides
    def encode(self, input_vector: torch.Tensor):
        return self.encoder(input_vector)

    @overrides
    def get_beta(self):
        return self._decoder._parameters['weight'].data.transpose(0, 1)  # pylint: disable=W0212
