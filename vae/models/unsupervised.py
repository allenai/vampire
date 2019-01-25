from typing import Dict, Optional

import torch
from allennlp.data.vocabulary import (DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN,
                                      Vocabulary)
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, TokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from overrides import overrides

from common.util import schedule
from vae.modules.semi_supervised_base import SemiSupervisedBOW
from vae.modules.vae import VAE


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
    update_bg: ``bool``:
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
                 input_embedder: TextFieldEmbedder,
                 bow_embedder: TokenEmbedder,
                 vae: VAE,
                 # --- parameters specific to this model ---
                 kl_weight_annealing: str = None,
                 # -----------------------------------------
                 background_data_path: str = None,
                 update_bg: bool = True,
                 track_topics: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(UnsupervisedNVDM, self).__init__(
                vocab, input_embedder, bow_embedder, vae,
                background_data_path=background_data_path, update_bg=update_bg,
                track_topics=track_topics, initializer=initializer,
                regularizer=regularizer)

        self.kl_weight_annealing = kl_weight_annealing
        self.batch_num = 0

        # Batchnorm to be applied throughout inference.
        vae_vocab_size = self.vocab.get_vocab_size("vae")
        self.bow_bn = torch.nn.BatchNorm1d(vae_vocab_size, eps=0.001, momentum=0.001, affine=True)
        self.bow_bn.weight.data.copy_(torch.ones(vae_vocab_size, dtype=torch.float64))
        self.bow_bn.weight.requires_grad = False

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

    def _freeze_vae(self) -> None:
        """
        Freeze the weights of the VAE.
        """
        model_parameters = dict(self.vae.named_parameters())
        for item in model_parameters:
            model_parameters[item].requires_grad = False

    @overrides
    def forward(self, # pylint: disable=arguments-differ
                tokens: Dict[str, torch.LongTensor],
                labels: torch.Tensor = None,
                epoch_num=None):
        # TODO: Port the rest of the metrics that `nvdm.py` is using.
        output_dict = {}

        if not self.training:
            self.weight_scheduler = lambda x: 1.0  # pylint: disable=W0201
        else:
            self.weight_scheduler = lambda x: schedule(x, self.kl_weight_annealing)  # pylint: disable=W0201

        target_bow = self._bow_embedding(tokens)
        target_bow = self.dropout(target_bow)
        num_tokens = target_bow.sum()

        # Encode the text into a shared representation for both the VAE
        # and downstream classifiers to use.
        mask = get_text_field_mask(tokens)
        embedded_tokens = self.input_embedder(tokens)
        encoder_output = self.vae.encoder(embedded_tokens, mask)

        # Perform variational inference.
        variational_output = self.vae(encoder_output)

        # Reconstructed bag-of-words from the VAE with background bias.
        reconstructed_bow = variational_output['reconstruction'] + self.background
        reconstructed_bow_bn = self.bow_bn(reconstructed_bow)

        # Reconstruction log likelihood: log P(x | z) = log softmax(z beta + b)
        reconstruction_loss = SemiSupervisedBOW.bow_reconstruction_loss(reconstructed_bow_bn, target_bow)
        reconstruction_loss /= num_tokens

        # KL-divergence that is returned is the mean of the batch by default.
        negative_kl_divergence = variational_output['negative_kl_divergence']
        kld_weight = self.weight_scheduler(self.batch_num)
        elbo = negative_kl_divergence * kld_weight + reconstruction_loss

        output_dict['loss'] = elbo
        output_dict['activations'] = {
                'encoder_output': encoder_output,
                'theta': variational_output['theta'],
                'encoder_weights': self.encoder.architecture._linear_layers[0].weight  # pylint: disable=protected-access
        }

        # Update metrics
        self.metrics['nkld'](-torch.mean(negative_kl_divergence))
        self.metrics['nll'](-torch.mean(reconstruction_loss))
        self.metrics['elbo'](elbo.item())

        # batch_num is tracked for kl weight annealing
        self.batch_num += 1

        if self.track_topics:
            self.print_topics_once_per_epoch(epoch_num)

        return output_dict
