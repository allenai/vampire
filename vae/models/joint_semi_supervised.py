from typing import Any, Dict, List, Optional

import torch
from allennlp.data.vocabulary import (DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN,
                                      Vocabulary)
from allennlp.models.model import Model
from allennlp.modules import (FeedForward, Seq2VecEncoder, TextFieldEmbedder,
                              TokenEmbedder)
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Average, CategoricalAccuracy
from overrides import overrides

from vae.common.util import (log_standard_categorical,
                             separate_labeled_unlabeled_instances)
from vae.modules.semi_supervised_base import SemiSupervisedBOW
from vae.modules.vae.logistic_normal import LogisticNormal


@Model.register("joint_m2_classifier")
class JointSemiSupervisedClassifier(SemiSupervisedBOW):
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
    encoder: ``Seq2VecEncoder``
        The encoder that is learned jointly with (and is separate from) the VAE
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
                 input_embedder: TextFieldEmbedder,
                 bow_embedder: TokenEmbedder,
                 vae: LogisticNormal,
                 # --- parameters specific to this model ---
                 classification_layer: FeedForward,
                 encoder: Seq2VecEncoder,
                 alpha: float = 0.1,
                 # -----------------------------------------
                 ref_directory: str = None,
                 background_data_path: str = None,
                 update_background_freq: bool = True,
                 track_topics: bool = True,
                 apply_batchnorm: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(JointSemiSupervisedClassifier, self).__init__(
                vocab, bow_embedder, vae, ref_directory=ref_directory,
                background_data_path=background_data_path, update_background_freq=update_background_freq,
                track_topics=track_topics, apply_batchnorm=apply_batchnorm, initializer=initializer,
                regularizer=regularizer
        )
        self.input_embedder = input_embedder
        self.classification_layer = classification_layer
        self.num_classes = classification_layer.get_output_dim()
        self.encoder = encoder
        self.alpha = alpha

        # Learnable covariates to relate latent topics and labels.
        covariates = torch.FloatTensor(self.num_classes, self.vocab.get_vocab_size("vae"))
        self.covariates = torch.nn.Parameter(covariates)
        torch.nn.init.uniform_(self.covariates)

        # Loss functions.
        self.classification_criterion = torch.nn.CrossEntropyLoss(reduction='sum')

        # Additional classification metrics.
        self.metrics['accuracy'] = CategoricalAccuracy()
        self.metrics['cross_entropy'] = Average()

    def _classify(self, instances: Dict):
        """
        Given the instances, labeled or unlabeled, selects the correct input
        to use and classifies it.
        """
        token_mask = get_text_field_mask({"tokens": instances['tokens']})
        embedded_tokens = self.input_embedder({"tokens": instances['tokens']})
        encoded_input = self.encoder(embedded_tokens, token_mask)
        logits = self.classification_layer(encoded_input)

        return logits

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

    def _classification_loss(self, logits: torch.tensor, labels: torch.Tensor):
        return self.classification_criterion(logits, labels)

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                tokens: Dict[str, torch.LongTensor],
                filtered_tokens: Dict[str, torch.LongTensor],
                label: torch.Tensor,
                metadata: List[Dict[str, Any]],
                epoch_num=None):

        self.device = self.vae.get_beta().device  # pylint: disable=W0201

        output_dict = {}

        # Sort instances into labeled and unlabeled portions.
        labeled_instances, unlabeled_instances = separate_labeled_unlabeled_instances(
                tokens['tokens'], filtered_tokens['filtered_tokens'], label, metadata)

        labeled_loss = None
        if labeled_instances['tokens'].size(0) > 0:

            # Stopless Bag-of-Words to be reconstructed.
            labeled_bow = self._bow_embedding(labeled_instances['filtered_tokens'])

            # Logits for labeled data.
            labeled_logits = self._classify(labeled_instances)

            # Continue the labeled objective with only the true labels.
            label = labeled_instances['label']

            # Compute supervised reconstruction objective.
            labeled_loss = self.elbo(labeled_bow, label)

        # When provided, use the unlabeled data.
        unlabeled_loss = None
        if unlabeled_instances['tokens'].size(0) > 0:
            unlabeled_bow = self._bow_embedding(unlabeled_instances['filtered_tokens'])

            # Logits for unlabeled data where the label is a latent variable.
            unlabeled_logits = self._classify(unlabeled_instances)
            unlabeled_logits = torch.softmax(unlabeled_logits, dim=-1)

            unlabeled_loss = self.unlabeled_objective(unlabeled_bow, unlabeled_logits)

        # Classification loss and metrics.
        classification_loss = self._classification_loss(labeled_logits, label)

        # ELBO loss.
        labeled_loss = -torch.sum(labeled_loss if labeled_loss is not None else torch.FloatTensor([0])
                                  .to(self.device))
        unlabeled_loss = -torch.sum(unlabeled_loss
                                    if unlabeled_loss is not None else torch.FloatTensor([0])
                                    .to(self.device))

        # Joint supervised and unsupervised learning.
        J_alpha = (labeled_loss + unlabeled_loss) + (self.alpha * classification_loss)  # pylint: disable=C0103
        output_dict['loss'] = J_alpha

        self.metrics['accuracy'](labeled_logits, label)
        self.metrics['elbo'](labeled_loss.item() + unlabeled_loss.item())
        self.metrics['cross_entropy'](self.alpha * classification_loss)

        if self.track_topics:
            self.print_topics_once_per_epoch(epoch_num)

        return output_dict

    def elbo(self,
             target_bow: torch.Tensor,
             label: torch.Tensor):  # TODO: Make this an optional parameter.
        """
        Computes ELBO loss. For convenience, also returns classification loss
        given the label.

        Parameters:
        ----------
        target_bow: ``torch.Tensor``
            The bag-of-words representation of the input excluding stopwords.
        label: ``torch.Tensor``
            The target class labels, expexted as (batch,). Used only for
            computing the unlabeled objective; this label is treated
            as a latent variable unlike the label provided in labeled
            versions of `instances`.
        """
        batch_size = target_bow.size(0)

        # One-hot the label vector before concatenation.
        label_one_hot = torch.FloatTensor(batch_size, self.num_classes).to(device=self.device)
        label_one_hot.zero_()
        label_one_hot = label_one_hot.scatter_(1, label.reshape(-1, 1), 1)

        # Variational inference, where Z ~ q(z | x, y) OR Z ~ q(z | h(x), y)
        variational_input = torch.cat((target_bow, label_one_hot), dim=-1)

        # Encode the input text and perform variational inference.
        variational_input = self.vae.encode(variational_input)
        variational_output = self.vae(variational_input)

        # Reconstructed bag-of-words from the VAE.
        reconstructed_bow = variational_output['reconstruction']

        # Introduce background and label-specific bias.
        reconstructed_bow = self._background_freq + reconstructed_bow + self.covariates[label]

        if self._apply_batchnorm:
            reconstructed_bow_bn = self.bow_bn(reconstructed_bow)

        # Reconstruction log likelihood: log P(x | y, z) = log softmax(b + z beta + y C)
        reconstruction_loss = SemiSupervisedBOW.bow_reconstruction_loss(reconstructed_bow_bn, target_bow)

        negative_kl_divergence = variational_output['negative_kl_divergence']

        # Uniform prior.
        prior = -log_standard_categorical(label_one_hot)

        # ELBO = - KL-Div(q(z | x, y), P(z)) +  E[ log P(x | z, y) + log p(y) ]
        elbo = negative_kl_divergence + reconstruction_loss + prior

        # Update metrics
        self.metrics['nkld'](-torch.mean(negative_kl_divergence))
        self.metrics['nll'](-torch.mean(reconstruction_loss))

        return elbo

    def unlabeled_objective(self,
                            target_bow: torch.Tensor,
                            logits: torch.Tensor):
        """
        Computes loss for unlabeled data.

        Parameters
        ----------
        target_bow: ``torch.Tensor``
            The bag-of-words representation of the input excluding stopwords.
        logits: ``torch.Tensor``
            The classification logits produced after applying the classifier to
            the input_representation.

        Returns
        -------
        The ELBO objective and the entropy of the predicted
        classification logits for each example in the batch.
        """
        batch_size = target_bow.size(0)

        # No work to be done on zero instances.
        if batch_size == 0:
            return None

        elbos = torch.zeros(self.num_classes, batch_size).to(device=self.device)
        for i in range(self.num_classes):
            # Instantiate an artifical labelling of each class.
            # Labels are treated as a latent variable that we marginalize over.
            label = (torch.ones(batch_size).long() * i).to(device=self.device)
            elbos[i] = self.elbo(target_bow, label)

        # Compute q(y | x)(-ELBO) and entropy H(q(y|x)), sum over all labels.
        # Reshape elbos to be (batch, num_classes) before the per-class weighting.
        L_weighted = torch.sum(logits * elbos.t(), dim=-1)  # pylint: disable=C0103
        H = -torch.sum(logits * torch.log(logits + 1e-8), dim=-1)  # pylint: disable=C0103

        return L_weighted + H
