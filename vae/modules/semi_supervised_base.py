from typing import Dict, Optional
import os
from tabulate import tabulate
import numpy as np
import torch
from torch.nn.functional import log_softmax
from overrides import overrides
from tqdm import tqdm
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import Average
from allennlp.common.checks import ConfigurationError
from vae.modules.vae.logistic_normal import LogisticNormal
from vae.common.util import compute_background_log_frequency


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
                 background_data_path: str = None,
                 kl_weight_annealing: str = None,
                 update_background_freq: bool = True,
                 track_topics: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SemiSupervisedBOW, self).__init__(vocab, regularizer)

        self.metrics = {
                'nkld': Average(),
                'nll': Average(),
                'elbo': Average()
                }

        self.vocab = vocab
        self.bow_embedder = bow_embedder
        self.vae = vae
        self.track_topics = track_topics
        self.vocab_namespace = "vae"
        self._update_background_freq = update_background_freq
        self._background_freq = self.initialize_bg_from_file(background_data_path)
        self._covariates = None

        if kl_weight_annealing == "linear":
            self._kld_weight = min(1, 1 / 50)
        elif kl_weight_annealing == "sigmoid":
            self._kld_weight = float(1/(1 + np.exp(-0.25 * (1 - 15))))
        elif kl_weight_annealing == "constant":
            self._kld_weight = 1.0
        else:
            raise ConfigurationError("anneal type {} not found")

        # Maintain these states for periodically printing topics and updating KLD
        self._topic_epoch_tracker = 0
        self._kl_epoch_tracker = 0
        self._cur_epoch = 0

        initializer(self)

    def initialize_bg_from_file(self, file) -> None:
        background_freq = compute_background_log_frequency(self.vocab, self.vocab_namespace, file)
        return torch.nn.Parameter(background_freq, requires_grad=self._update_background_freq)

    @staticmethod
    def bow_reconstruction_loss(reconstructed_bow: torch.Tensor,
                                target_bow: torch.Tensor):
        # Final shape: (batch, )
        log_reconstructed_bow = log_softmax(reconstructed_bow + 1e-10, dim=-1)
        reconstruction_loss = torch.sum(target_bow * log_reconstructed_bow, dim=-1)
        return reconstruction_loss

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        output = {}
        for metric_name, metric in self.metrics.items():
            if isinstance(metric, float):
                output[metric_name] = metric
            else:
                output[metric_name] = float(metric.get_metric(reset))
        return output

    def update_kld_weight(self, epoch_num, kl_weight_annealing='constant'):
        """
        weight annealing scheduler
        """
        epoch_num = epoch_num[0]
        if epoch_num != self._kl_epoch_tracker:
            print(self._kld_weight)
            self._kl_epoch_tracker = epoch_num
            self._cur_epoch += 1
            if kl_weight_annealing == "linear":
                self._kld_weight = min(1, self._cur_epoch / 50)
            elif kl_weight_annealing == "sigmoid":
                self._kld_weight = float(1 / (1 + np.exp(-0.25 * (self._cur_epoch - 15))))
            elif kl_weight_annealing == "constant":
                self._kld_weight = 1.0
            else:
                raise ConfigurationError("anneal type {} not found")

    def print_topics_once_per_epoch(self, epoch_num):
        if epoch_num[0] != self._topic_epoch_tracker:
            tqdm.write(tabulate(self.extract_topics(self.vae.get_beta()), headers=["Topic #", "Words"]))
            topic_dir = os.path.join(os.path.dirname(self.vocab.serialization_dir), "topics")
            if not os.path.exists(topic_dir):
                os.mkdir(topic_dir)
            ser_dir = os.path.dirname(self.vocab.serialization_dir)
            topic_filepath = os.path.join(ser_dir, "topics", "topics_{}.txt".format(epoch_num[0]))
            with open(topic_filepath, 'w+') as file_:
                file_.write(tabulate(self.extract_topics(self.vae.get_beta()), headers=["Topic #", "Words"]))
            if self._covariates:
                tqdm.write(tabulate(self.extract_topics(self.covariates), headers=["Covariate #", "Words"]))
            self._topic_epoch_tracker = epoch_num[0]

    def extract_topics(self, weights, k: int = 20):
        """
        Given the learned (K, vocabulary size) weights, print the
        top k words from each row as a topic.

        :param weights: ``torch.Tensor``
            The weight matrix whose second dimension equals the vocabulary size.
        :param k: ``int``
            The number of words per topic to display.
        """

        words = list(range(weights.size(1)))
        words = [self.vocab.get_token_from_index(i, "vae") for i in words]

        topics = []

        word_strengths = list(zip(words, self._background_freq.tolist()))
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
