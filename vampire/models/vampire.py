import logging
import os
from functools import partial
from itertools import combinations
from operator import is_not
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TokenEmbedder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Average
from overrides import overrides
from scipy import sparse
from tabulate import tabulate
from torch.nn.functional import log_softmax
from allennlp.training.metrics import CategoricalAccuracy


from vampire.common.util import (compute_background_log_frequency, load_sparse,
                                 read_json)
from vampire.modules import VAE
from vampire.modules.encoder import Encoder

logger = logging.getLogger(__name__)


@Model.register("vampire")
class VAMPIRE(Model):
    """
    VAMPIRE is a variational model for pretraining under low resource environments.

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
                #  additional_input_embedder: TextFieldEmbedder,
                #  additional_input_encoder: Encoder,
                 vae: VAE,
                 kl_weight_annealing: str = None,
                 linear_scaling: float = 1000.0,
                 sigmoid_weight_1: float = 0.25,
                 sigmoid_weight_2: float = 15,
                 background_data_path: str = None,
                 reference_counts: str = None,
                 reference_vocabulary: str = None,
                 update_background_freq: bool = True,
                 track_topics: bool = True,
                 track_npmi: bool = True,
                 apply_batchnorm: bool = True,
                 num_sources: int = 1,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.metrics = {
                'nkld': Average(),
                'nll': Average(),
                # 'acc': CategoricalAccuracy()
                }

        self.vocab = vocab
        self.vae = vae
        self.track_topics = track_topics
        self.track_npmi = track_npmi
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
        # Batchnorm to be applied throughout inference.
        self._apply_batchnorm = apply_batchnorm
        vae_vocab_size = self.vocab.get_vocab_size("vae")
        self.num_covariates = self.vocab.get_vocab_size("covariate")
        self.bow_bn = torch.nn.BatchNorm1d(vae_vocab_size, eps=0.001, momentum=0.001, affine=True)
        self.bow_bn.weight.data.copy_(torch.ones(vae_vocab_size, dtype=torch.float64))
        self.bow_bn.weight.requires_grad = False

        self._linear_scaling = float(linear_scaling)
        self._sigmoid_weight_1 = float(sigmoid_weight_1)
        self._sigmoid_weight_2 = float(sigmoid_weight_2)

        if kl_weight_annealing == "linear":
            self._kld_weight = min(1, 1 / self._linear_scaling)
        elif kl_weight_annealing == "sigmoid":
            self._kld_weight = float(1/(1 + np.exp(-self._sigmoid_weight_1 * (1 - self._sigmoid_weight_2))))
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
        self._cur_npmi = 0.0
        initializer(self)
        self.bow_embedder = bow_embedder
        # self._additional_input_embedder = additional_input_embedder
        # self._additional_input_encoder = additional_input_encoder
        self._num_sources = num_sources
        # self._covariate_prediction_layer = torch.nn.Linear(self.vae.encoder.get_output_dim(),
                                                        #    self._num_sources)
        self.kl_weight_annealing = kl_weight_annealing
        self.batch_num = 0        
        # self._cross_entropy = torch.nn.CrossEntropyLoss()

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

    def update_kld_weight(self, epoch_num: List[int], kl_weight_annealing: str = 'constant', linear_scaling: float = 1000.0, sigmoid_weight_1: float=0.25, sigmoid_weight_2: int = 15) -> None:
        """
        weight annealing scheduler
        """
        if not epoch_num:
            self._kld_weight = 1.0
        else:
            _epoch_num = epoch_num[0]
            if _epoch_num != self._kl_epoch_tracker:
                print(self._kld_weight)
                self._kl_epoch_tracker = _epoch_num
                self._cur_epoch += 1
                if kl_weight_annealing == "linear":
                    self._kld_weight = min(1, self._cur_epoch / linear_scaling)
                elif kl_weight_annealing == "sigmoid":
                    self._kld_weight = float(1 / (1 + np.exp(-sigmoid_weight_1 * (self._cur_epoch - sigmoid_weight_2))))
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

            if self.track_npmi:
                if self._ref_vocab:
                    self._cur_npmi = self.update_npmi()
            self._metric_epoch_tracker = epoch_num[0]

    def update_npmi(self) -> float:
        topics = self.extract_topics(self.vae.get_beta())
        mean_npmi = self.compute_npmi(topics[1:])
        return mean_npmi

    def update_topics(self, epoch_num):
        topic_table = tabulate(self.extract_topics(self.vae.get_beta()), headers=["Topic #", "Words"])
        topic_dir = os.path.join(os.path.dirname(self.vocab.serialization_dir), "topics")
        if not os.path.exists(topic_dir):
            os.mkdir(topic_dir)
        ser_dir = os.path.dirname(self.vocab.serialization_dir)
        topic_filepath = os.path.join(ser_dir, "topics", "topics_{}.txt".format(epoch_num[0]))
        with open(topic_filepath, 'w+') as file_:
            file_.write(topic_table)

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

    @staticmethod
    def generate_npmi_vals(interactions, sums):
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
        topics_idx = [[self._ref_vocab_index.get(word) for word in topic[1][:num_words]] for topic in topics]
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
        npmi_data = ((np.log10(self.n_docs) + self._npmi_numerator[rows, cols])
                     / (np.log10(self.n_docs) - self._npmi_denominator[rows, cols]))
        npmi_data[npmi_data == 1.0] = 0.0
        npmi_shape = (len(topics), len(list(combinations(range(max_seq_len), 2))))
        npmi = sparse.csr_matrix((npmi_data.tolist()[0], (res_rows, res_cols)), shape=npmi_shape)
        return npmi.mean()

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
                covariate: torch.Tensor = None,
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
            self.update_kld_weight(epoch_num,
                                   self.kl_weight_annealing,
                                   linear_scaling=self._linear_scaling,
                                   sigmoid_weight_1=self._sigmoid_weight_1,
                                   sigmoid_weight_2=self._sigmoid_weight_2)

        embedded_tokens = self._bow_embedding(tokens['tokens'])

        # additional_embeddings = self._additional_input_embedder(tokens)
    
        # mask = get_text_field_mask(tokens)
        # additional_encoding = self._additional_input_encoder(embedded_text = additional_embeddings, mask=mask)
        
        # embeddings = [additional_encoding]
        # if metadata:
        #     covariate_embedding = torch.FloatTensor(embedded_tokens.shape[0], self._num_sources).to(device=self.device)
        #     covariate_embedding.zero_()
        #     covariate_embedding.scatter_(1, covariate.unsqueeze(-1), 1)
        #     embeddings.append(covariate_embedding)

        # input_embedding = torch.cat(embeddings, 1)
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
        reconstruction_loss = self.bow_reconstruction_loss(reconstructed_bow, embedded_tokens)

        # KL-divergence that is returned is the mean of the batch by default.
        negative_kl_divergence = variational_output['negative_kl_divergence']

        # logits = self._covariate_prediction_layer(variational_output['theta'])

        # covariate_prediction_loss = self._cross_entropy(logits, covariate.long().view(-1))

        # covariate_prediction_acc = self.metrics['acc'](logits, covariate)
        elbo = negative_kl_divergence * self._kld_weight + reconstruction_loss

        loss = -torch.mean(elbo) 
        # + covariate_prediction_loss

        output_dict['loss'] = loss
        # output_dict['cov acc'] = covariate_prediction_acc
        theta = variational_output['theta']

        activations: List[Tuple[str, torch.FloatTensor]] = []
        intermediate_input = embedded_tokens
        for layer_index, layer in enumerate(self.vae.encoder._linear_layers):  # pylint: disable=protected-access
            intermediate_input = layer(intermediate_input)
            activations.append((f"encoder_layer_{layer_index}", intermediate_input))

        # activations.append(('theta', variational_output['params']['mean']))
        activations.append(('theta', theta))

        output_dict['activations'] = activations

        output_dict['mask'] = get_text_field_mask(tokens)
        # Update metrics
        # self.metrics['kld_weight'] = float(self._kld_weight)
        self.metrics['nkld'](-torch.mean(negative_kl_divergence))
        self.metrics['nll'](-torch.mean(reconstruction_loss))
        # self.metrics['elbo'](loss)

        # batch_num is tracked for kl weight annealing
        self.batch_num += 1

        self.compute_custom_metrics_once_per_epoch(epoch_num)

        self.metrics['npmi'] = self._cur_npmi

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
