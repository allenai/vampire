from typing import Any, Dict, List, Optional

import torch
from allennlp.data.vocabulary import (DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN,
                                      Vocabulary)
from allennlp.models.model import Model
from allennlp.modules import TokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from numpy import nan
from overrides import overrides

from vae.modules.semi_supervised_base import SemiSupervisedBOW
from vae.modules.vae.logistic_normal import LogisticNormal


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
                 vae: LogisticNormal,
                 # --- parameters specific to this model ---
                 kl_weight_annealing: str = None,
                 dropout: float = 0.2,
                 # -----------------------------------------
                 background_data_path: str = None,
                 ref_directory: str = None,
                 update_background_freq: bool = True,
                 track_topics: bool = True,
                 apply_batchnorm: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(UnsupervisedNVDM, self).__init__(
                vocab, bow_embedder, vae, kl_weight_annealing=kl_weight_annealing, ref_directory=ref_directory,
                background_data_path=background_data_path, update_background_freq=update_background_freq,
                track_topics=track_topics, apply_batchnorm=apply_batchnorm, initializer=initializer,
                regularizer=regularizer)
        self.kl_weight_annealing = kl_weight_annealing
        self.batch_num = 0
        self.dropout = torch.nn.Dropout(dropout)

        self._cur_npmi = nan

    def _bow_embedding(self, bow: torch.Tensor):
        """
        In practice, excluding the OOV explicitly helps topic coherence.
        Clearing padding is a precautionary measure.

        For convenience, moves them to the GPU.
        """
        bow = self.bow_embedder(bow)
        bow[:, self.vocab.get_token_index(DEFAULT_OOV_TOKEN, "vae")] = 0
        bow[:, self.vocab.get_token_index(DEFAULT_PADDING_TOKEN, "vae")] = 0
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
        reconstructed_bow = variational_output['reconstruction'] + self._background_freq

        if self._apply_batchnorm:
            reconstructed_bow_bn = self.bow_bn(reconstructed_bow)

        # Reconstruction log likelihood: log P(x | z) = log softmax(z beta + b)
        reconstruction_loss = SemiSupervisedBOW.bow_reconstruction_loss(reconstructed_bow_bn, embedded_tokens)

        # KL-divergence that is returned is the mean of the batch by default.
        negative_kl_divergence = variational_output['negative_kl_divergence']

        elbo = negative_kl_divergence * self._kld_weight + reconstruction_loss

        loss = -torch.sum(elbo)

        output_dict['loss'] = loss
        output_dict['activations'] = {
                'encoder_output': encoder_output,
                'theta': variational_output['theta'],
                'encoder_weights': self.vae.encoder._linear_layers[-1].weight  # pylint: disable=protected-access
        }

        # Update metrics
        self.metrics['kld_weight'] = float(self._kld_weight)
        self.metrics['nkld'](-torch.mean(negative_kl_divergence))
        self.metrics['nll'](-torch.mean(reconstruction_loss))
        self.metrics['perp'](float((-torch.mean(reconstruction_loss / embedded_tokens.sum(1))).exp()))
        self.metrics['elbo'](loss)

        # batch_num is tracked for kl weight annealing
        self.batch_num += 1

        self.compute_custom_metrics_once_per_epoch(epoch_num)

        self.metrics['npmi'] = float(self._cur_npmi)

        return output_dict
