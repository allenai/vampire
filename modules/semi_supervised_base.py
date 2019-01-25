from typing import Dict, Optional

from tabulate import tabulate
import torch
from torch.nn.functional import log_softmax

from allennlp.data.vocabulary import Vocabulary

from allennlp.models.model import Model
from allennlp.modules import FeedForward, TextFieldEmbedder, TokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import Average, CategoricalAccuracy
from overrides import overrides

from modules.vae import VAE

from common.util import compute_background_log_frequency


@Model.register("SemiSupervisedBOW")  # pylint: disable=W0223
class SemiSupervisedBOW(Model):
    """
    Neural variational document-level topic model.
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
                 classification_layer: FeedForward,
                 vae: VAE,
                 background_data_path: str = None,
                 update_bg: bool = True,
                 track_topics: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SemiSupervisedClassifier, self).__init__(vocab, regularizer)

        self.metrics = {
            'nkld': Average(),
            'nll': Average(),
            'elbo': Average()
        }

        if classification_layer is not None:
            self.metrics['accuracy'] = CategoricalAccuracy()
            self.metrics['cross_entropy'] = Average()

        self.vocab = vocab
        self.input_embedder = input_embedder
        self.bow_embedder = bow_embedder
        self.classifier = classification_layer
        self.vae = vae
        self.track_topics = track_topics
        self.num_classes = classification_layer.get_output_dim()

        # Loss functions.
        self.classification_criterion = torch.nn.CrossEntropyLoss(reduction='sum')

        # Learnable bias.
        if background_data_path is not None:
            background = compute_background_log_frequency(
                background_data_path, vocab, "vae")
            self.background = torch.nn.Parameter(
                background, requires_grad=update_bg)
        else:
            background = torch.FloatTensor(self.vocab.get_vocab_size("vae"))
            self.background = torch.nn.Parameter(background)
            torch.nn.init.uniform_(self.background)

        # For computing metrics and printing topics.
        self.step = 0

        # TODO: Verify that this works on a GPU.
        # For easy tranfer to the GPU.
        self.device = self.vae.get_beta().device

        self._epoch = 0

        initializer(self)

    def bow_reconstruction_loss(self,
                                reconstructed_bow: torch.Tensor,
                                target_bow: torch.Tensor):
        # Final shape: (batch, )
        log_reconstructed_bow = log_softmax(reconstructed_bow + 1e-10, dim=-1)
        reconstruction_loss = torch.sum(target_bow * log_reconstructed_bow, dim=-1)
        return reconstruction_loss

    def classification_loss(self, logits: torch.tensor, labels: torch.Tensor):
        return self.classification_criterion(logits, labels)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: float(metric.get_metric(reset)) for metric_name, metric in self.metrics.items()}

    def print_topics_once_per_epoch(self, epoch_num):
        if epoch_num[0] != self._epoch:
            print(tabulate(self.extract_topics(self.beta), headers=["Topic #", "Words"]))
            print(tabulate(self.extract_topics(self.covariates), headers=["Covariate #", "Words"]))
            self._epoch = epoch_num[0]

    def extract_topics(self, k: int = 20):
        """
        Given the learned (K, vocabulary size) weights, print the
        top k words from each row as a topic.

        :param weights: ``torch.Tensor``
            The weight matrix whose second dimension equals the vocabulary size.
        :param k: ``int``
            The number of words per topic to display.
        """

        weights = self.vae.get_beta()  # TODO: Transpose?
        words = list(range(weights.size(1)))
        words = [self.vocab.get_token_from_index(i, "stopless") for i in words]

        topics = []

        word_strengths = list(zip(words, self.background.tolist()))
        sorted_by_strength = sorted(word_strengths,
                                    key=lambda x: x[1],
                                    reverse=True)
        background = [x[0] for x in sorted_by_strength][:k]
        topics.append(('bg', background))

        for i, topic in enumerate(weights):
            word_strengths = list(zip(words, topic.tolist()))
            sorted_by_strength = sorted(word_strengths,
                                        key=lambda x: x[1],
                                        reverse=True)
            topic = [x[0] for x in sorted_by_strength][:k]
            topics.append((i, topic))

        return topics
