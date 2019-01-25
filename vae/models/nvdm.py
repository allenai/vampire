from typing import Dict
import torch
import numpy as np
from allennlp.models.model import Model
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy, Average
from tabulate import tabulate
from tqdm import tqdm
from vae.modules.distribution import Distribution
from vae.modules.encoder import Encoder
from vae.modules.decoder import Decoder
from vae.common.util import (schedule, compute_background_log_frequency)
from vae.modules import Classifier


@Model.register("nvdm")
class NVDM(Model):
    """
    This is the neural variational document model
    (NVDM; https://arxiv.org/abs/1511.06038)

    This VAE represents documents as a bag of words, encodes them into a latent representation
    (with a series of feedforward networks), and then decodes the latent representation into a bag of words.

    With this code, you can optionally include a classifier to predict labels from either
    the encoder or the latent representation (``theta``).

    Additionally, you can apply different priors to the latent distribution:
        - Gaussian (https://arxiv.org/abs/1312.6114)
        - logistic normal (https://arxiv.org/abs/1703.01488)
        - vMF (https://arxiv.org/abs/1808.10805)

    The logistic normal prior tends to result in reasonable topics from the latent space.

    With the logistic normal prior and classifier on the latent representation,
    you effectively are using SCHOLAR (https://arxiv.org/abs/1705.09296)

    With a gaussian prior and a classifier on the encoder, you effectively are using M2
    (https://arxiv.org/abs/1406.5298)

    By default, during training this model tracks:
        - ``kld``: KL-divergence
        - ``nll``: negative log likelihood
        - ``elbo``: evidence lower bound
        - ``accuracy``: accuracy of classifier (if specified)
        - ``kld weight``: KL-divergence weight
        - ``perp``: perplexity

    Parameters
    ----------
    vocab : ``Vocabulary``
        vocabulary to use
    latent_dim : ``int``
        Dimension of the latent space. Note that different latent dimensions will
        affect your reconstruction perplexity.
    encoder : ``Encoder``
        VAE encoder to use. check ``modules.encoder`` for available encoders.
        For NVDM you should use a feedforward encoder.
    decoder : ``Decoder``
        VAE decoder to use. check ``modules.decoder`` for available decoders.
        For NVDM you should use a feedforward decoder.
    distribution : ``Distribution``
        VAE prior distribution to use. check ``modules.distribution`` for available distributions.
    embedder: ``TextFieldEmbedder``
        text field embedder to use. For NVDM, you should use a bag-of-word-counts embedder.
    classifier: ``Classifier``, optional (default = ``None``)
        if specified, will apply the corresponding classifier in the VAE to guide
        representation learning.
    background_data_path: ``str``, optional (default = ``None``)
        Path to a file containing log frequency statistics on the vocabulary,
        which is used to bias the decoder. This is not necessary to specify during training,
        as AllenNLP will generate the bg frequency by indexing the vocabulary.
    update_bg: ``bool``, optional (default = ``False``)
        Whether or not to update the background frequency bias during training.
    kl_weight_annealing: ``str``, optional (default = ``None``)
        If specified, will anneal kl divergence overtime, which was shown to be helpful to prevent KL collapse
        (https://arxiv.org/abs/1511.06349)
        Options for weight annealing: ``constant``, ``sigmoid``, ``linear``
    dropout: ``float``, optional (default = ``0.5``)
        dropout applied to the input
    track_topics: ``bool``, optional (default = ``False``)
        if True, we will display topics learned from decoder during training
    topic_log_interval: ``int``, optional (default = ``100``)
        batch interval to display topics.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 latent_dim: int,
                 encoder: Encoder,
                 decoder: Decoder,
                 distribution: Distribution,
                 embedder: TextFieldEmbedder,
                 classifier: Classifier = None,
                 update_background_freq: bool = False,
                 kl_weight_annealing: str = None,
                 dropout: float = 0.5,
                 track_topics: bool = False,
                 topic_log_interval: int = 100,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(NVDM, self).__init__(vocab)

        self.metrics = {
                'kld': Average(),
                'nll': Average(),
                'elbo': Average(),
        }

        self.vocab_namespace = "vae"

        if classifier is not None:
            self.metrics['accuracy'] = CategoricalAccuracy()

        self.track_topics = track_topics

        background_freq = compute_background_log_frequency(vocab, self.vocab_namespace)
        self._update_background_freq = update_background_freq

        if self._update_background_freq:
            self._background_freq = torch.nn.Parameter(background_freq, requires_grad=True)
        else:
            self._background_freq = torch.nn.Parameter(background_freq, requires_grad=False)

        self.vocab = vocab
        self._embedder = embedder
        self.dist = distribution
        self.latent_dim = latent_dim
        self.topic_log_interval = topic_log_interval
        self.embedding_dim = embedder.token_embedder_tokens.get_output_dim()
        self.encoder = encoder
        self.decoder = decoder
        self.dist_apply_batchnorm = self.dist.apply_batchnorm
        self.decoder_apply_batchnorm = self.decoder.apply_batchnorm
        self.step = 0
        self.batch_num = 0
        self.dropout = torch.nn.Dropout(dropout)
        self.kl_weight_annealing = kl_weight_annealing
        self.weight_scheduler = lambda x: schedule(x, self.kl_weight_annealing)
        self.classifier = classifier

        # we initialize parts of the decoder, classifier, and distribution here
        # so we don't have to repeat
        # dimensions in the config, which can be cumbersome.

        self.encoder.initialize_encoder_architecture(self.embedding_dim)

        param_input_dim = self.encoder.architecture.get_output_dim()
        if self.classifier is not None:
            if self.classifier.input == 'theta':
                self.classifier.initialize_classifier_hidden(latent_dim)
            elif self.classifier.input == 'encoder_output':
                self.classifier.initialize_classifier_hidden(self.encoder.architecture.get_output_dim())
            self.classifier.initialize_classifier_out(vocab.get_vocab_size("labels"))

        if self.classifier is not None:
            if self.classifier.input == 'encoder_output':
                param_input_dim += vocab.get_vocab_size("labels")

        self.dist.initialize_params(param_input_dim, latent_dim)

        self.decoder.initialize_decoder_out(latent_dim, vocab.get_vocab_size(self.vocab_namespace))

        initializer(self)

    def initialize_bg_from_file(self, file) -> None:
        background_freq = compute_background_log_frequency(self.vocab, self.vocab_namespace, file)
        if self._update_background_freq:
            self._background_freq = torch.nn.Parameter(background_freq, requires_grad=True)
        else:
            self._background_freq = torch.nn.Parameter(background_freq, requires_grad=False)

    def freeze_weights(self) -> None:
        """
        Freeze the weights of the model
        """

        model_parameters = dict(self.named_parameters())
        for item in model_parameters:
            model_parameters[item].requires_grad = False

    def extract_topics(self, k=20):
        """
        Given the learned (topic, vocabulary size) beta, print the
        top k words from each topic.

        Parameters
        __________

        k: ``int``, optional (default = 20)
            number of words to print per topic
        """
        decoder_weights = self.decoder.decoder_out.weight.data.transpose(0, 1)
        words = list(range(decoder_weights.size(1)))
        words = [self.vocab.get_token_from_index(i, self.vocab_namespace) for i in words]
        topics = []

        if self._background_freq is not None:
            word_strengths = list(zip(words, self._background_freq.tolist()))
            sorted_by_strength = sorted(word_strengths,
                                        key=lambda x: x[1],
                                        reverse=True)
            background = [x[0] for x in sorted_by_strength][:k]
            topics.append(('bg', background))
        for i, topic in enumerate(decoder_weights):
            word_strengths = list(zip(words, topic.tolist()))
            sorted_by_strength = sorted(word_strengths,
                                        key=lambda x: x[1],
                                        reverse=True)
            topic = [x[0] for x in sorted_by_strength][:k]
            topics.append((i, topic))
        return topics

    # pylint: disable=arguments-differ
    def forward(self,
                tokens: Dict[str, torch.IntTensor],
                label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:

        if not self.training:
            self.weight_scheduler = lambda x: 1.0
        else:
            self.weight_scheduler = lambda x: schedule(x, self.kl_weight_annealing)

        mask = get_text_field_mask(tokens)

        onehot_repr = self._embedder(tokens)
        onehot_repr[:, self.vocab.get_token_index("@@UNKNOWN@@", "vae")] = 0
        onehot_repr[:, self.vocab.get_token_index("@@PADDING@@", "vae")] = 0
        onehot_repr = self.dropout(onehot_repr)

        num_tokens = onehot_repr.sum()

        encoder_output = self.encoder(embedded_text=onehot_repr, mask=mask)
        input_repr = [encoder_output['encoder_output']]

        if self.classifier is not None and label is not None:
            if self.classifier.input == 'encoder_output':
                clf_output = self.classifier(encoder_output['encoder_output'], label)
                input_repr.append(clf_output['label_repr'])

        input_repr = torch.cat(input_repr, 1)

        # use parameterized distribution to compute latent code and KL divergence
        _, kld, theta = self.dist.generate_latent_code(input_repr, n_sample=1, training=self.training)

        if self.classifier is not None and label is not None:
            if self.classifier.input == 'theta':
                clf_output = self.classifier(theta, label)

        # decode using the latent code and background frequency.
        decoder_output = self.decoder(theta=theta,
                                      bg=self._background_freq)

        decoder_probs = torch.nn.functional.log_softmax(decoder_output['decoder_output'], dim=1)
        error = torch.mul(onehot_repr, decoder_probs)
        nll_loss = -torch.sum(error)

        nll = nll_loss / num_tokens

        kld_weight = self.weight_scheduler(self.batch_num)

        kld = kld.to(nll.device) / num_tokens

        # compute the ELBO
        elbo = (nll + kld * kld_weight).mean()

        if self.classifier is not None and label is not None:
            elbo += clf_output['loss']

        output = {
                'loss': elbo,
                'elbo': elbo,
                'nll': nll,
                'kld': kld,
                'kld_weight': kld_weight,
                }

        if self.classifier is not None and label is not None:
            output['clf_loss'] = clf_output['loss']
            output['logits'] = clf_output['logits']

        self.metrics["elbo"](output['elbo'])
        self.metrics["kld"](output['kld'])
        self.metrics["kld_weight"] = output['kld_weight']
        self.metrics["nll"](output['nll'])
        self.metrics["perp"] = float(np.exp(self.metrics['nll'].get_metric()))

        if self.classifier is not None and label is not None:
            self.metrics['accuracy'](output['logits'], label)

        # to use the VAE as a feature embedder we also output the learned representations
        # of the input text from various layers
        output['activations'] = {
                'encoder_output': encoder_output['encoder_output'],
                'theta': theta,
                'encoder_weights': self.encoder.architecture._linear_layers[0].weight  # pylint: disable=protected-access
        }
        output['mask'] = mask

        if self.track_topics and self.training:
            if self.step == self.topic_log_interval:
                topics = self.extract_topics()
                tqdm.write(tabulate(topics))
                self.step = 0
            else:
                self.step += 1

        # batch_num is tracked for kl weight annealing
        self.batch_num += 1

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        output = {}
        for metric_name, metric in self.metrics.items():
            if isinstance(metric, float):
                output[metric_name] = metric
            else:
                output[metric_name] = float(metric.get_metric(reset))
        return output
