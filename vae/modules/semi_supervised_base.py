import os
from typing import Dict, List, Optional, Tuple
import logging
from itertools import combinations

import numpy as np
from scipy import sparse
from allennlp.common.checks import ConfigurationError
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import Average
from allennlp.common.file_utils import cached_path
from overrides import overrides
from tabulate import tabulate
import torch
from torch.nn.functional import log_softmax
from tqdm import tqdm

from vae.common.util import compute_background_log_frequency, load_sparse, read_json
from vae.modules.vae.logistic_normal import LogisticNormal
from scripts.compute_npmi import compute_npmi_during_train, get_files

logger = logging.getLogger(__name__)

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
                 reference_counts: str = None,
                 reference_vocabulary: str = None,
                 kl_weight_annealing: str = None,
                 update_background_freq: bool = True,
                 track_topics: bool = True,
                 apply_batchnorm: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SemiSupervisedBOW, self).__init__(vocab, regularizer)
        self.metrics = {
                'nkld': Average(),
                'nll': Average(),
                'elbo': Average(),
                'perp': Average(),
                'z_entropy': Average(),
                'z_max': Average(),
                'z_min': Average(),
                'npmi': Average()
                }

        self.vocab = vocab
        self.bow_embedder = bow_embedder
        self.vae = vae
        self.track_topics = track_topics
        self.vocab_namespace = "vae"
        self._update_background_freq = update_background_freq
        self._background_freq = self.initialize_bg_from_file(background_data_path)
        self._ref_vocab = reference_vocabulary
        self._ref_counts = reference_counts
        if self._ref_vocab is not None:
            logger.info("Loading reference vocabulary.")
            self._ref_vocab = read_json(cached_path(self._ref_vocab))
            self._ref_vocab_index = dict(zip(self._ref_vocab, range(len(self._ref_vocab))))
            logger.info("Loading reference count matrix.")
            self._ref_counts = load_sparse(cached_path(self._ref_counts))
            logger.info("Computing word interaction matrix.")
            self._ref_doc_counts = (self._ref_counts > 0).astype(float)
            self._ref_interaction = (self._ref_doc_counts).T.dot(self._ref_doc_counts)
            self._ref_doc_sum = np.array(self._ref_doc_counts.sum(0).tolist()[0])
            logger.info("Generating npmi matrices.")
            self._npmi_numerator, self._npmi_denominator = self.generate_npmi_vals(self._ref_vocab, self._ref_counts, self._ref_interaction, self._ref_doc_sum)
            self.n_docs = self._ref_counts.shape[0]
        self._covariates = None

        # Batchnorm to be applied throughout inference.
        self._apply_batchnorm = apply_batchnorm
        vae_vocab_size = self.vocab.get_vocab_size("vae")
        self.bow_bn = torch.nn.BatchNorm1d(vae_vocab_size, eps=0.001, momentum=0.001, affine=True)
        self.bow_bn.weight.data.copy_(torch.ones(vae_vocab_size, dtype=torch.float64))
        self.bow_bn.weight.requires_grad = False

        if kl_weight_annealing == "linear":
            self._kld_weight = min(1, 1 / 1000)
        elif kl_weight_annealing == "sigmoid":
            self._kld_weight = float(1/(1 + np.exp(-0.25 * (1 - 15))))
        elif kl_weight_annealing == "constant":
            self._kld_weight = 1.0
        elif kl_weight_annealing is None:
            self._kld_weight = 1.0
        else:
            raise ConfigurationError("anneal type {} not found".format(kl_weight_annealing))

        # Maintain these states for periodically printing topics and updating KLD
        self._metric_epoch_tracker = 0
        self._kl_epoch_tracker = 0
        self._cur_epoch = 0
        self._cur_npmi = np.nan
        initializer(self)

    def initialize_bg_from_file(self, file: str) -> torch.Tensor:
        background_freq = compute_background_log_frequency(self.vocab, self.vocab_namespace, file)
        return torch.nn.Parameter(background_freq, requires_grad=self._update_background_freq)

    @staticmethod
    def bow_reconstruction_loss(reconstructed_bow: torch.Tensor,
                                target_bow: torch.Tensor) -> torch.Tensor:
        # Final shape: (batch, )
        log_reconstructed_bow = log_softmax(reconstructed_bow + 1e-10, dim=-1)
        reconstruction_loss = torch.sum(target_bow * log_reconstructed_bow, dim=-1)
        return reconstruction_loss

    def theta_entropy(self, theta):
        normalizer = torch.log(torch.Tensor([theta.size(-1)])).to(theta.device)
        log_theta = torch.log(theta)
        normalized_entropy = -torch.sum((theta * log_theta), dim=-1) / normalizer
        return torch.mean(normalized_entropy)

    def theta_extremes(self, theta):
        maxes = torch.max(theta, dim=-1)[0]
        mins = torch.min(theta, dim=-1)[0]
        return torch.mean(maxes), torch.mean(mins)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        output = {}
        for metric_name, metric in self.metrics.items():
            if isinstance(metric, float):
                output[metric_name] = metric
            else:
                output[metric_name] = float(metric.get_metric(reset))
        return output

    def update_kld_weight(self, epoch_num: List[int], kl_weight_annealing: str = 'constant') -> None:
        """
        weight annealing scheduler
        """
        _epoch_num = epoch_num[0]
        if _epoch_num != self._kl_epoch_tracker:
            print(self._kld_weight)
            self._kl_epoch_tracker = _epoch_num
            self._cur_epoch += 1
            if kl_weight_annealing == "linear":
                self._kld_weight = min(1, self._cur_epoch / 50)
            elif kl_weight_annealing == "sigmoid":
                self._kld_weight = float(1 / (1 + np.exp(-0.25 * (self._cur_epoch - 15))))
            elif kl_weight_annealing == "constant":
                self._kld_weight = 1.0
            elif kl_weight_annealing is None:
                self._kld_weight = 1.0
            else:
                raise ConfigurationError("anneal type {} not found".format(kl_weight_annealing))

    def compute_custom_metrics_once_per_epoch(self, epoch_num: List[int]) -> None:
        if epoch_num and epoch_num[0] != self._metric_epoch_tracker:

            # Logs the newest set of topics.
            if self.track_topics:
                self.update_topics(epoch_num)

            self._metric_epoch_tracker = epoch_num[0]

    def update_npmi(self) -> float:
        topics = self.extract_topics(self.vae.get_beta())
        mean_npmi = self.compute_npmi(topics)
        return mean_npmi

    def update_topics(self, epoch_num):
        topic_table = tabulate(self.extract_topics(self.vae.get_beta()), headers=["Topic #", "Words"])
        # tqdm.write(topic_table)
        topic_dir = os.path.join(os.path.dirname(self.vocab.serialization_dir), "topics")
        if not os.path.exists(topic_dir):
            os.mkdir(topic_dir)
        ser_dir = os.path.dirname(self.vocab.serialization_dir)
        topic_filepath = os.path.join(ser_dir, "topics", "topics_{}.txt".format(epoch_num[0]))
        with open(topic_filepath, 'w+') as file_:
            file_.write(topic_table)
        if self._covariates:
            cov_table = tabulate(self.extract_topics(self.covariates), headers=["Covariate #", "Words"])
            # tqdm.write(table)
            covariate_dir = os.path.join(os.path.dirname(self.vocab.serialization_dir), "covariates")
            if not os.path.exists(covariate_dir):
                os.mkdir(covariate_dir)
            covariate_filepath = os.path.join(ser_dir, "covariate", "covariate_{}.txt".format(epoch_num[0]))
            with open(covariate_filepath, 'w+') as file_:
                file_.write(cov_table)

    def extract_topics(self, weights: torch.Tensor, k: int = 20) -> List[Tuple[str, List[int]]]:
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
            top_k = [x[0] for x in sorted_by_strength][:k]
            topics.append((str(i), top_k))

        return topics

    def generate_npmi_vals(self, vocab, counts, interactions, sums):
        r, c = np.triu_indices(sums.size, 1)
        interaction_rows, interaction_cols = interactions.nonzero()
        logger.info("generating doc sums...")
        doc_sums = sparse.csr_matrix((np.log10(sums[interaction_rows]) + np.log10(sums[interaction_cols]),
                                      (interaction_rows, interaction_cols)),
                                     shape=interactions.shape)
        logger.info("generating numerator...")
        interactions.data = np.log10(interactions.data)
        numerator = interactions - doc_sums
        logger.info("generating denominator...")
        denominator = interactions
        return numerator, denominator

    def compute_npmi(self, topics, num_words=10):
        n_docs, _ = self._ref_counts.shape
        npmi_means = []
        topics_idx = [[self._ref_vocab_index.get(word) for word in topic[1][:num_words]] for topic in topics]
        npmis = []
        for topic in topics_idx:
            indices = list(combinations(topic, 2))
            indices = [x for x in indices if None not in x]
            rows = [x[0] for x in indices]
            cols = [x[1] for x in indices]
            npmi = (np.log10(self.n_docs) + self._npmi_numerator[rows, cols]) / (np.log10(self.n_docs) - self._npmi_denominator[rows, cols])
            npmi[npmi == 1.0] = 0.0
            npmis.append(np.mean(npmi))
        return np.mean(npmis)
