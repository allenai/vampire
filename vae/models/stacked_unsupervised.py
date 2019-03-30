from typing import Any, Dict, List, Optional, Tuple
import math

import torch
from overrides import overrides
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask

from vae.modules.semi_supervised_base import SemiSupervisedBOW
from vae.modules.vae.vae import VAE


@Model.register("stacked_unsupervised")
class StackedUnsupervisedNVDM(SemiSupervisedBOW):
    """
    VAE topic model trained in a semi-supervised environment
    (https://arxiv.org/abs/1406.5298).

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    input_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    bow_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model
        into a bag-of-word-counts.
    classification_layer : ``Feedfoward``
        Projection from latent topics to classification logits.
    vae : ``VAE``
        The variational autoencoder used to project the BoW into a latent space.
    alpha: ``float``
        Scales the importance of classification.
    background_data_path: ``str``
        Path to a JSON file containing word frequencies accumulated over the training corpus.
    update_background_freq: ``bool``:
        Whether to allow the background frequency to be learnable.
    track_topics: ``bool``:
        Whether to periodically print the learned topics.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 vae_embedder_1: TextFieldEmbedder,
                 vae_embedder_2: TextFieldEmbedder,
                 # --- parameters specific to this model ---
                 vae: VAE,
                 # -----------------------------------------
                 background_data_path: str = None,
                 reference_counts: str = None,
                 reference_vocabulary: str = None,
                 update_background_freq: bool = True,
                 track_topics: bool = False,
                 track_npmi: bool = False,
                 apply_batchnorm: bool = True,
                 kl_weight_annealing: str = "sigmoid",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(StackedUnsupervisedNVDM, self).__init__(
                vocab, vae, kl_weight_annealing=kl_weight_annealing,
                reference_counts=reference_counts, reference_vocabulary=reference_vocabulary,
                background_data_path=background_data_path, update_background_freq=update_background_freq,
                track_topics=track_topics, track_npmi=track_npmi, apply_batchnorm=apply_batchnorm,
                initializer=initializer, regularizer=regularizer)
        self.vae_embedder_1 = vae_embedder_1
        self.vae_embedder_2 = vae_embedder_2
        self.kl_weight_annealing = kl_weight_annealing
        self.batch_num = 0
        self._vae = vae

    def freeze_weights(self) -> None:
        """
        Freeze the weights of the VAE.
        """
        model_parameters = dict(self.vae.named_parameters())
        for item in model_parameters:
            model_parameters[item].requires_grad = False

    def _stacked_reconstruction_loss(self, input_embedding: torch.Tensor, variational_output: Dict):
        # Encode the input text and perform variational inference as usual.
        # Concatenate the label before encoding.

        # Extract mean and variance.
        mean = variational_output['params']['mean']
        log_variance = variational_output['params']['log_variance']

        precision = torch.exp(-log_variance)
        power = ((input_embedding - mean) ** 2) * precision * 0.5

        latent_dim = input_embedding.size(-1)

        log_likelihood = (-(torch.sum(log_variance * 0.5 + power, dim=-1) +
                            latent_dim * (math.log(2 * math.pi))))

        # Here, we return log likelihood, as elbo itself will return a negative value.
        return log_likelihood

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                tokens: Dict[str, torch.LongTensor],
                label: torch.Tensor = None,  # pylint: disable=unused-argument
                metadata: List[Dict[str, Any]] = None,  # pylint: disable=unused-argument
                epoch_num=None):

        # TODO: Verify that this works on a GPU.
        # For easy tranfer to the GPU.
        self.device = self.vae.get_beta().device  # pylint: disable=W0201

        # TODO: Port the rest of the metrics that `nvdm.py` is using.
        output_dict = {}

        if not self.training:
            self._kld_weight = 1.0  # pylint: disable=W0201
        else:
            self.update_kld_weight(epoch_num, self.kl_weight_annealing)
        embedded_tokens_1 = self.vae_embedder_1({"vae_tokens_1": tokens['vae_tokens_1']})
        embedded_tokens_2 = self.vae_embedder_2({"vae_tokens_2": tokens['vae_tokens_2']})
        # embedded_tokens = self.dropout(embedded_tokens)

        # Encode the text into a shared representation for both the VAE
        # and downstream classifiers to use.

        embedding = torch.cat([embedded_tokens_1, embedded_tokens_2], dim=1)
        variational_input = self.vae.encode(embedding)

        variational_output = self.vae.generate_latent_code(variational_input)

        # Reconstruction log likelihood: log P(x | z) = log softmax(z beta + b)
        reconstruction_loss = self._stacked_reconstruction_loss(embedding, variational_output)

        # KL-divergence that is returned is the mean of the batch by default.
        negative_kl_divergence = variational_output['negative_kl_divergence']

        elbo = negative_kl_divergence * self._kld_weight + reconstruction_loss

        loss = -torch.mean(elbo)

        output_dict['loss'] = loss
        theta = variational_output['theta']

        activations: List[Tuple[str, torch.FloatTensor]] = []
        intermediate_input = embedding
        for layer_index, layer in enumerate(self.vae.encoder._linear_layers):  # pylint: disable=protected-access
            intermediate_input = layer(intermediate_input)
            activations.append((f"encoder_layer_{layer_index}", intermediate_input))

        # activations.append(('theta', variational_output['params']['mean']))
        activations.append(('theta', theta))

        output_dict['activations'] = activations

        output_dict['mask'] = get_text_field_mask(tokens)
        # Update metrics
        self.metrics['kld_weight'] = float(self._kld_weight)
        self.metrics['nkld'](-torch.mean(negative_kl_divergence))
        self.metrics['nll'](-torch.mean(reconstruction_loss))
        self.metrics['elbo'](loss)
        self.metrics['z_entropy'](self.theta_entropy(theta))

        theta_max, theta_min = self.theta_extremes(theta)
        self.metrics['z_max'](theta_max)
        self.metrics['z_min'](theta_min)

        # batch_num is tracked for kl weight annealing
        self.batch_num += 1

        self.compute_custom_metrics_once_per_epoch(epoch_num)

        self.metrics['npmi'] = self._cur_npmi

        return output_dict
