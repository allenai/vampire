from typing import Any, Dict, List, Optional

import torch
from overrides import overrides
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator

from allennlp.training.metrics import Average, CategoricalAccuracy

from vae.common.util import (log_standard_categorical,
                             separate_labeled_unlabeled_instances)
from vae.modules.semi_supervised_base import SemiSupervisedBOW
from vae.modules.vae.vae import VAE

from vae.models.classifier import Classifier


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
                 bow_embedder: TokenEmbedder,
                 vae: VAE,
                 # --- parameters specific to this model ---
                 classifier: Classifier,
                 kl_weight_annealing: str = None,
                 alpha: float = 0.1,
                 # -----------------------------------------
                 reference_counts: str = None,
                 reference_vocabulary: str = None,
                 background_data_path: str = None,
                 update_background_freq: bool = True,
                 track_topics: bool = True,
                 track_npmi: bool = True,
                 apply_batchnorm: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(JointSemiSupervisedClassifier, self).__init__(
                vocab, bow_embedder, vae, reference_counts=reference_counts,
                kl_weight_annealing=kl_weight_annealing, reference_vocabulary=reference_vocabulary,
                background_data_path=background_data_path, update_background_freq=update_background_freq,
                track_topics=track_topics, track_npmi=track_npmi,
                apply_batchnorm=apply_batchnorm, initializer=initializer,
                regularizer=regularizer
        )
        self.kl_weight_annealing = kl_weight_annealing
        self.classifier = classifier
        self.num_classes = self.vocab.get_vocab_size(namespace='labels')  # pylint: disable=protected-access

        self.alpha = alpha

        # Learnable covariates to relate latent topics and labels.
        covariates = torch.FloatTensor(self.num_classes, self.vocab.get_vocab_size("vae"))
        self.covariates = torch.nn.Parameter(covariates)
        torch.nn.init.uniform_(self.covariates)

        # Additional classification metrics.
        self.metrics['accuracy'] = CategoricalAccuracy()
        self.metrics['cross_entropy'] = Average()

    def _bow_embedding(self, bow: torch.Tensor):
        """
        For convenience, moves them to the GPU.
        """
        bow = self.bow_embedder(bow)
        bow = bow.to(device=self.device)
        return bow

    def _classify(self, instances: Dict):
        """
        Given the instances, labeled or unlabeled, selects the correct input
        to use and classifies it.
        """
        return self.classifier({"tokens": instances['classifier_tokens']}, instances.get('label'))


    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                tokens: Dict[str, torch.LongTensor],
                classifier_tokens: Dict[str, torch.LongTensor],
                label: torch.Tensor,
                metadata: List[Dict[str, Any]],
                epoch_num=None):

        self.device = self.vae.get_beta().device  # pylint: disable=W0201

        output_dict = {}

        if not self.training:
            self._kld_weight = 1.0  # pylint: disable=W0201
        else:
            self.update_kld_weight(epoch_num, self.kl_weight_annealing)

        # Sort instances into labeled and unlabeled portions.
        labeled_instances, unlabeled_instances = separate_labeled_unlabeled_instances(
                tokens['tokens'], classifier_tokens['classifier_tokens'], label, metadata)

        labeled_elbo, unlabeled_elbo = None, None
        classification_loss = 0
        labeled_batch_size = labeled_instances['tokens'].size(0)
        unlabeled_batch_size = unlabeled_instances['tokens'].size(0)
        target_bow = []
        target_label = []

        # We compute ELBO across labeled and unlabelled instances in parallel to allow batchnorm
        # to operate over all reconstructions at once.
        #
        # First, include the unlabeled examples, enumerated over classes.
        # Here, we are hallucinating a label with which to marginalize.
        if unlabeled_batch_size > 0:
            unlabeled_bow = self._bow_embedding(unlabeled_instances['tokens'])
            target_bow += [unlabeled_bow for _ in range(self.num_classes)]

            # Instantiate an artifical labelling of each class.
            # Labels are treated as a latent variable that we marginalize over.
            unlabeled_label = []
            for i in range(self.num_classes):
                unlabeled_label += [(torch.ones(unlabeled_batch_size).long() * i).to(device=self.device)]

            target_label += unlabeled_label

        # Next, concatenate the labeled target bow.
        if labeled_batch_size > 0:
            label_for_labeled_instances = labeled_instances['label']
            labeled_bow = self._bow_embedding(labeled_instances['tokens'])
            target_bow += [labeled_bow]
            target_label += [label_for_labeled_instances]

            # We can also compute classification loss here.
            # Logits for labeled data.
            labeled_classifier_output = self._classify(labeled_instances)
            labeled_logits = labeled_classifier_output['label_logits']
            self.metrics['accuracy'](labeled_logits, label_for_labeled_instances)
            classification_loss = labeled_classifier_output['loss']

        # Compute ELBO across all examples. Consolidate the list of tensors
        # into a single tensor first.
        target_bow = torch.cat(target_bow, dim=0)
        target_label = torch.cat(target_label, dim=0)
        assert target_bow.size(0) == target_label.size(0)
        elbo = self.elbo(target_bow, target_label)

        # Extract unlabeled elbo and perform the unlabeled objective.
        # Reshape to (num_classes, unlabelled_batch_size) for easy marginalization.
        if unlabeled_batch_size > 0:
            initial_unlabeled_elbo = elbo[:unlabeled_batch_size * self.num_classes]
            initial_unlabeled_elbo = initial_unlabeled_elbo.view(self.num_classes, unlabeled_batch_size)

            # Compute unlabeled logits.
            unlabeled_classifier_output = self._classify(unlabeled_instances)
            unlabeled_probs = unlabeled_classifier_output['label_probs']

            # Marginalize for the unlabeled objective.
            unlabeled_elbo = self.unlabeled_objective(initial_unlabeled_elbo, unlabeled_probs)

        # Labeled loss is contained in the remaining elbos.
        if labeled_batch_size > 0:
            labeled_elbo = elbo[unlabeled_batch_size * self.num_classes:]

        # ELBO loss.
        labeled_loss = -torch.mean(labeled_elbo if labeled_elbo is not None else torch.FloatTensor([0])
                                  .to(self.device))
        unlabeled_loss = -torch.mean(unlabeled_elbo
                                    if unlabeled_elbo is not None else torch.FloatTensor([0])
                                    .to(self.device))

        # Joint supervised and unsupervised learning.
        J_alpha = (labeled_loss + unlabeled_loss) + (self.alpha * classification_loss)  # pylint: disable=C0103
        output_dict['loss'] = J_alpha

        self.metrics['elbo'](labeled_loss.item() + unlabeled_loss.item())
        self.metrics['cross_entropy'](self.alpha * classification_loss)

        self.compute_custom_metrics_once_per_epoch(epoch_num)

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
            reconstructed_bow = self.bow_bn(reconstructed_bow)

        # Reconstruction log likelihood: log P(x | y, z) = log softmax(b + z beta + y C)
        reconstruction_loss = SemiSupervisedBOW.bow_reconstruction_loss(reconstructed_bow, target_bow)

        negative_kl_divergence = variational_output['negative_kl_divergence']

        # Uniform prior.
        prior = -log_standard_categorical(label_one_hot)

        # ELBO = - KL-Div(q(z | x, y), P(z)) +  E[ log P(x | z, y) + log p(y) ]
        elbo = negative_kl_divergence * self._kld_weight + reconstruction_loss + prior

        # Update metrics
        self.metrics['kld_weight'] = float(self._kld_weight)
        self.metrics['nkld'](-torch.mean(negative_kl_divergence))
        self.metrics['nll'](-torch.mean(reconstruction_loss))

        theta = variational_output['theta']
        self.metrics['z_entropy'](self.theta_entropy(theta))

        theta_max, theta_min = self.theta_extremes(theta)
        self.metrics['z_max'](theta_max)
        self.metrics['z_min'](theta_min)
        self.metrics['npmi'] = self._cur_npmi
        return elbo

    def unlabeled_objective(self,
                            elbos: torch.Tensor,
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

        batch_size = elbos.size(0)
        if batch_size == 0:
            return 0

        # Compute q(y | x)(-ELBO) and entropy H(q(y|x)), sum over all labels.
        # Reshape elbos to be (batch, num_classes) before the per-class weighting.
        L_weighted = torch.sum(logits * elbos.t(), dim=-1)  # pylint: disable=C0103
        H = -torch.sum(logits * torch.log(logits + 1e-8), dim=-1)  # pylint: disable=C0103

        return L_weighted + H
