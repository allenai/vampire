from typing import Any, Dict, List, Optional, Tuple

import torch
from overrides import overrides
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask

from vae.modules.semi_supervised_base import SemiSupervisedBOW
from vae.modules.vae.vae import VAE


@Model.register("nvdm")
class UnsupervisedNVDM(SemiSupervisedBOW):
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
                 bow_embedder: TokenEmbedder,
                 vae: VAE,
                 # --- parameters specific to this model ---
                 kl_weight_annealing: str = None,
                 dropout: float = 0.2,
                 # -----------------------------------------
                 background_data_path: str = None,
                 reference_counts: str = None,
                 reference_vocabulary: str = None,
                 update_background_freq: bool = True,
                 track_topics: bool = True,
                 track_npmi: bool = True,
                 apply_batchnorm: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(UnsupervisedNVDM, self).__init__(
                vocab, bow_embedder, vae, kl_weight_annealing=kl_weight_annealing,
                reference_counts=reference_counts, reference_vocabulary=reference_vocabulary,
                background_data_path=background_data_path, update_background_freq=update_background_freq,
                track_topics=track_topics, track_npmi=track_npmi, apply_batchnorm=apply_batchnorm,
                initializer=initializer, regularizer=regularizer)
        self.kl_weight_annealing = kl_weight_annealing
        self.batch_num = 0
        self.dropout = torch.nn.Dropout(dropout)

    def _bow_embedding(self, bow: torch.Tensor):
        """
        For convenience, moves them to the GPU.
        """
        bow = self.bow_embedder(bow)
        bow = bow.to(device=self.device)
        return bow

    def freeze_weights(self) -> None:
        """
        Freeze the weights of the VAE.
        """
        model_parameters = dict(self.vae.named_parameters())
        for item in model_parameters:
            model_parameters[item].requires_grad = False

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

        embedded_tokens = self._bow_embedding(tokens['tokens'])
        # embedded_tokens = self.dropout(embedded_tokens)

        # Encode the text into a shared representation for both the VAE
        # and downstream classifiers to use.
        encoder_output = self.vae.encoder(embedded_tokens)

        # Perform variational inference.
        variational_output = self.vae(encoder_output)

        # Reconstructed bag-of-words from the VAE with background bias.
        # Variational reconstruction is not the same order of magnitude...
        # Should consider log softmax before adding
        reconstructed_bow = variational_output['reconstruction'] + self._background_freq

        if self._apply_batchnorm:
            reconstructed_bow = self.bow_bn(reconstructed_bow)

        # Reconstruction log likelihood: log P(x | z) = log softmax(z beta + b)
        reconstruction_loss = SemiSupervisedBOW.bow_reconstruction_loss(reconstructed_bow, embedded_tokens)

        # KL-divergence that is returned is the mean of the batch by default.
        negative_kl_divergence = variational_output['negative_kl_divergence']

        elbo = negative_kl_divergence * self._kld_weight + reconstruction_loss

        loss = -torch.mean(elbo)

        output_dict['loss'] = loss
        theta = variational_output['theta']

        activations: List[Tuple(str, torch.FloatTensor)] = []
        intermediate_input = embedded_tokens
        for layer_index, layer in enumerate(self.vae.encoder._linear_layers):
            intermediate_input = layer(intermediate_input)
            activations.append((f"encoder_layer_{layer_index}", intermediate_input))

        activations.append(('theta', variational_output['params']['mean']))
        # activations.append(('theta', theta))

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
