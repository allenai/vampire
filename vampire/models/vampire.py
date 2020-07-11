import logging
import os
from functools import partial
from itertools import combinations
from operator import is_not
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TokenEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import Average
from overrides import overrides
from scipy import sparse
from tabulate import tabulate

from vampire.common.util import (compute_background_log_frequency, load_sparse,
                                 read_json)
from vampire.modules import VAE
from allennlp.training.trainer import EpochCallback, BatchCallback
from allennlp.data.dataloader import TensorDict


logger = logging.getLogger(__name__)

@BatchCallback.register("track_learning_rate")
class TrackLearningRate(BatchCallback):
    def __call__(
        self,
        trainer: "TrackLearningRate",
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_master: bool,
    ) -> None:
        trainer.model.metrics['lr'] = trainer.optimizer.param_groups[0]['lr']

@EpochCallback.register("kl_anneal")
class KLAnneal(EpochCallback):
    def __init__(self,
                 kl_weight_annealing: str,
                 linear_scaling: Union[str, float],
                 sigmoid_weight_1: Union[str, float],
                 sigmoid_weight_2: Union[str, float]) -> None:
        self._kl_weight_annealing = kl_weight_annealing
        self._linear_scaling = float(linear_scaling)
        self._sigmoid_weight_1 = float(sigmoid_weight_1)
        self._sigmoid_weight_2 = float(sigmoid_weight_2)
        super().__init__()

    def update_kld_weight(self, model: Model, epoch: int) -> float:
        """
        KL weight annealing scheduler

        Parameters
        ----------
        ``epoch_num`` : List[int]
            epoch tracker output (containing current epoch number)
        """
        if self._kl_weight_annealing == "linear":
            kld_weight = min(1, epoch / self._linear_scaling)
        elif self._kl_weight_annealing == "sigmoid":
            exp = np.exp(- self._sigmoid_weight_1 * (epoch - self._sigmoid_weight_2))
            kld_weight = float(1 /(1 + exp))
        elif self._kl_weight_annealing == "constant":
            kld_weight = 1.0
        else:
            raise ConfigurationError("anneal type {} not found".format(self._kl_weight_annealing))
        return kld_weight
    
    def __call__(
        self, trainer: "KLAnneal", metrics: Dict[str, Any], epoch: int, is_master: bool
    ) -> None:
        if epoch > -1:
            kld_weight = self.update_kld_weight(trainer.model, epoch)
            trainer.model._kld_weight = kld_weight
            trainer.model.metrics['kld_weight'] = kld_weight
        else:
            kld_weight = 1.0
            trainer.model._kld_weight = kld_weight
            trainer.model.metrics['kld_weight'] = kld_weight

@EpochCallback.register("compute_topics")
class ComputeTopics(EpochCallback):

    def extract_topics(self, model: Model, k: int = 20) -> List[Tuple[str, List[int]]]:
        """
        Given the learned (K, vocabulary size) weights, print the
        top k words from each row as a topic.

        Parameters
        ----------
        weights: ``torch.Tensor``
            The weight matrix whose second dimension equals the vocabulary size.
        k: ``int``
            The number of words per topic to display.

        Returns
        -------
        topics: ``List[Tuple[str, List[int]]]``
            collection of learned topics
        """
        weights = model.vae.get_beta()
        words = list(range(weights.size(1)))
        words = [model.vocab.get_token_from_index(i, model.vocab_namespace) for i in words]

        topics = []

        word_strengths = list(zip(words, model._background_freq.tolist()))
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

    def update_topics(self, model, serialization_dir, epoch) -> List[Tuple[str, List[int]]]:
        """
        Update topics and NPMI once per epoch

        Parameters
        ----------
        ``epoch_num`` : List[int]
            epoch tracker output (containing current epoch number)
        """
        # Logs the newest set of topics.
        topics = self.extract_topics(model)
        topic_table = tabulate(topics, headers=["Topic #", "Words"])
        topic_dir = os.path.join(serialization_dir, "topics")
        if not os.path.exists(topic_dir):
            os.mkdir(topic_dir)
        # Topics are saved for the previous epoch.
        topic_filepath = os.path.join(serialization_dir, "topics", "topics_{}.txt".format(epoch))
        with open(topic_filepath, 'w+') as file_:
            file_.write(topic_table)     
        return topics

    def update_npmi(self) -> None:
        """
        Update topics and NPMI at the beginning of validation.

        Parameters
        ----------
        ``epoch_num`` : List[int]
            epoch tracker output (containing current epoch number)
        """

        topics = self.extract_topics(self.vae.get_beta())
        self.compute_npmi(topics[1:])

    def compute_npmi(self, model, topics, num_words=10):
        """
        Compute global NPMI across topics

        Parameters
        ----------
        topics: ``List[Tuple[str, List[int]]]``
            list of learned topics
        num_words: ``int``
            number of words to compute npmi over
        """
        topics_idx = [[model._ref_vocab_index.get(word)
                       for word in topic[1][:num_words]] for topic in topics]
        rows = []
        cols = []
        res_rows = []
        res_cols = []
        max_seq_len = max([len(topic) for topic in topics_idx])

        for index, topic in enumerate(topics_idx):
            topic = list(filter(partial(is_not, None), topic))
            if len(topic) > 1:
                _rows, _cols = zip(*combinations(topic, 2))
                res_rows.extend([index] * len(_rows))
                res_cols.extend(range(len(_rows)))
                rows.extend(_rows)
                cols.extend(_cols)

        npmi_data = ((np.log10(model.n_docs) + model._npmi_numerator[rows, cols])
                     / (np.log10(model.n_docs) - model._npmi_denominator[rows, cols]))
        npmi_data[npmi_data == 1.0] = 0.0
        npmi_shape = (len(topics), len(list(combinations(range(max_seq_len), 2))))
        npmi = sparse.csr_matrix((npmi_data.tolist()[0], (res_rows, res_cols)), shape=npmi_shape)
        return npmi.mean()
 
    def __call__(
        self, trainer: "ComputeTopics", metrics: Dict[str, Any], epoch: int, is_master: bool
    ) -> None:
        if epoch > -1:
            topics = self.update_topics(trainer.model, trainer._serialization_dir, epoch)
            npmi = self.compute_npmi(trainer.model, topics)
            trainer.model.metrics['npmi'] = npmi


@Model.register("vampire")
class VAMPIRE(Model):
    """
    VAMPIRE is a variational document model for pretraining under low
    resource environments.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    bow_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model
        into a bag-of-word-counts.
    vae : ``VAE``, required
        The variational autoencoder used to project the BoW into a latent space.
    kl_weight_annealing : ``string``, required
        Annealing weight on the KL divergence of ELBO.
        Choice between `sigmoid`, `linear` and `constant` annealing.
    linear_scaling: ``float``
        scaling applied ot KL weight annealing
    sigmoid_weight_1: ``float``
        first weight applied to sigmoid KL annealing
    sigmoid_weight_2: ``float``
        second weight applied to sigmoid KL annealing
    background_data_path: ``str``
        Path to a JSON file containing word frequencies accumulated over the training corpus.
    reference_counts: ``str``
        Path to reference counts for NPMI calculation
    reference_vocabulary: ``str``
        Path to reference vocabulary for NPMI calculation
    update_background_freq: ``bool``:
        Whether to allow the background frequency to be learnable.
    track_topics: ``bool``:
        Whether to periodically print the learned topics.
    track_npmi: ``bool``:
        Whether to track NPMI every epoch.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 bow_embedder: TokenEmbedder,
                 vae: VAE,
                 reference_counts: str = None,
                 reference_vocabulary: str = None,
                 background_data_path: str = None,
                 update_background_freq: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.metrics = {'nkld': Average(), 'nll': Average(), 'npmi': 0.0}

        self.vocab = vocab
        self.vae = vae
        self.vocab_namespace = "vampire"
        self._update_background_freq = update_background_freq
        self._background_freq = self.initialize_bg_from_file(file_=background_data_path)
        self._ref_counts = reference_counts

        if reference_vocabulary is not None:
            # Compute data necessary to compute NPMI every epoch
            logger.info("Loading reference vocabulary.")
            self._ref_vocab = read_json(cached_path(reference_vocabulary))
            self._ref_vocab_index = dict(zip(self._ref_vocab, range(len(self._ref_vocab))))
            logger.info("Loading reference count matrix.")
            self._ref_count_mat = load_sparse(cached_path(self._ref_counts))
            logger.info("Computing word interaction matrix.")
            self._ref_doc_counts = (self._ref_count_mat > 0).astype(float)
            self._ref_interaction = (self._ref_doc_counts).T.dot(self._ref_doc_counts)
            self._ref_doc_sum = np.array(self._ref_doc_counts.sum(0).tolist()[0])
            logger.info("Generating npmi matrices.")
            (self._npmi_numerator,
             self._npmi_denominator) = self.generate_npmi_vals(self._ref_interaction,
                                                               self._ref_doc_sum)
            self.n_docs = self._ref_count_mat.shape[0]

        vampire_vocab_size = self.vocab.get_vocab_size(self.vocab_namespace)
        self._bag_of_words_embedder = bow_embedder
        # setup batchnorm
        self.bow_bn = torch.nn.BatchNorm1d(vampire_vocab_size, eps=0.001, momentum=0.001, affine=True)
        self.bow_bn.weight.data.copy_(torch.ones(vampire_vocab_size, dtype=torch.float64))
        self.bow_bn.weight.requires_grad = False

        # Maintain these states for periodically printing topics and updating KLD

        initializer(self)

    def initialize_bg_from_file(self, file_: Optional[str] = None) -> torch.Tensor:
        """
        Initialize the background frequency parameter from a file

        Parameters
        ----------
        ``file`` : str
            path to background frequency file
        """
        background_freq = compute_background_log_frequency(self.vocab, self.vocab_namespace, file_)
        return torch.nn.Parameter(background_freq, requires_grad=self._update_background_freq)

    @staticmethod
    def bow_reconstruction_loss(reconstructed_bow: torch.Tensor,
                                target_bow: torch.Tensor) -> torch.Tensor:
        """
        Initialize the background frequency parameter from a file

        Parameters
        ----------
        ``reconstructed_bow`` : torch.Tensor
            reconstructed bag of words from VAE
        ``target_bow`` : torch.Tensor
            target bag of words tensor

        Returns
        -------
        ``reconstruction_loss``
            Cross entropy loss between reconstruction and target
        """
        log_reconstructed_bow = torch.nn.functional.log_softmax(reconstructed_bow + 1e-10, dim=-1)
        reconstruction_loss = torch.sum(target_bow * log_reconstructed_bow, dim=-1)
        return reconstruction_loss
    

    @staticmethod
    def generate_npmi_vals(interactions, document_sums):
        """
        Compute npmi values from interaction matrix and document sums

        Parameters
        ----------
        interactions: ``np.ndarray``
            Interaction matrix of size reference vocab size x reference vocab size,
            where cell [i][j] indicates how many times word i and word j co-occur
            in the corpus.
        document_sums: ``np.ndarray``
            Matrix of size number of docs x reference vocab size, where
            cell [i][j] indicates how many times word i occur in documents
            in the corpus
        TODO(suchin): update this documentation
        """
        interaction_rows, interaction_cols = interactions.nonzero()
        logger.info("generating doc sums...")
        doc_sums = sparse.csr_matrix((np.log10(document_sums[interaction_rows])
                                      + np.log10(document_sums[interaction_cols]),
                                      (interaction_rows, interaction_cols)),
                                     shape=interactions.shape)
        logger.info("generating numerator...")
        interactions.data = np.log10(interactions.data)
        numerator = interactions - doc_sums
        logger.info("generating denominator...")
        denominator = interactions
        return numerator, denominator

    def freeze_weights(self) -> None:
        """
        Freeze the weights of the VAE.
        """
        model_parameters = dict(self.vae.named_parameters())
        for item in model_parameters:
            model_parameters[item].requires_grad = False

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                tokens: Union[Dict[str, torch.IntTensor], torch.IntTensor]):
        """
        Parameters
        ----------
        tokens: ``Union[Dict[str, torch.IntTensor], torch.IntTensor]``
            A batch of tokens. We expect tokens to be represented in one of two ways:
                1. As token IDs. This representation will be used with downstream models, where bag-of-word count embedding
                must be done on the fly. If token IDs are provided, we use the bag-of-word-counts embedder to embed these
                tokens during training.
                2. As pre-computed bag of words vectors. This representation will be used during pretraining, where we can
                precompute bag-of-word counts and train much faster.
        epoch_num: ``List[int]``
            Output of epoch tracker
        """
        # For easy transfer to the GPU.
        self.device = self.vae.get_beta().device  # pylint: disable=W0201

        output_dict = {}

        if not self.training:
            self._kld_weight = 1.0  # pylint: disable=W0201


        # if you supply input as token IDs, embed them into bag-of-word-counts with a token embedder
        if isinstance(tokens, dict):
            if isinstance(tokens['tokens'], dict):
                tokens = tokens['tokens']['tokens']
            else:
                tokens = tokens['tokens']
            embedded_tokens = (self._bag_of_words_embedder(tokens)
                               .to(device=self.device))
        else:
            embedded_tokens = tokens

        # Perform variational inference.
        variational_output = self.vae(embedded_tokens)

        # Reconstructed bag-of-words from the VAE with background bias.
        reconstructed_bow = variational_output['reconstruction'] + self._background_freq

        # Apply batchnorm to the reconstructed bag of words.
        # Helps with word variety in topic space.
        reconstructed_bow = self.bow_bn(reconstructed_bow)

        # Reconstruction log likelihood: log P(x | z) = log softmax(z beta + b)
        reconstruction_loss = self.bow_reconstruction_loss(reconstructed_bow, embedded_tokens)

        # KL-divergence that is returned is the mean of the batch by default.
        negative_kl_divergence = variational_output['negative_kl_divergence']

        # Compute ELBO
        elbo = negative_kl_divergence * self._kld_weight + reconstruction_loss

        loss = -torch.mean(elbo)

        output_dict['loss'] = loss

        output_dict['activations'] = variational_output['activations']

        # Update metrics
        self.metrics['nkld'](-torch.mean(negative_kl_divergence))
        self.metrics['nll'](-torch.mean(reconstruction_loss))


        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        output = {}
        for metric_name, metric in self.metrics.items():
            if isinstance(metric, float):
                output[metric_name] = metric
            else:
                output[metric_name] = float(metric.get_metric(reset))
        return output
