import torch
import numpy as np
import os
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask
from typing import Dict, Optional, List, Any, Tuple
from allennlp.data import Vocabulary
from allennlp.modules import TextFieldEmbedder
from allennlp.modules import Seq2SeqEncoder, FeedForward
from allennlp.nn.util import get_text_field_mask, get_device_of, get_lengths_from_binary_sequence_mask
from allennlp.models.archival import load_archive, Archive
from allennlp.nn import InitializerApplicator
from overrides import overrides
from modules.vae import VAE
from modules.distribution import Distribution
from modules.encoder import Encoder
from modules.decoder import Decoder
from common.util import (schedule, compute_bow, log_standard_categorical, 
                         check_dispersion, compute_background_log_frequency)
from typing import Dict
from allennlp.training.metrics import CategoricalAccuracy, Average
from modules import Classifier
from tabulate import tabulate
from common.file_handling import load_sparse, read_text
from tqdm import tqdm
from collections import defaultdict

@Model.register("nvrnn")
class NVRNN(Model):
    """
    This is the neural variational recurrent neural network (NVRNN; https://www.semanticscholar.org/paper/Generating-Sentences-from-a-Continuous-Space-Bowman-Vilnis/3d1427961edccf8940a360d203e44539db58a60f)

    This VAE represents documents as a sequence of word vectors, encodes them into a latent representation (with a series of recurrent neural networks),
    and then decodes the latent representation into a sequence of word vectors.

    With this code, you can optionally include a classifier to predict labels from either the encoder or the latent representation (``theta``).

    Additionally, you can apply different priors to the latent distribution:
        - Gaussian (https://www.semanticscholar.org/paper/Auto-Encoding-Variational-Bayes-Kingma-Welling/0f88de2ae3dc2ec1371d1e9f675b9670902b289f)
        - logistic normal (https://www.semanticscholar.org/paper/Autoencoding-Variational-Inference-for-Topic-Models-Srivastava-Sutton/b2f7e9d7fb42254223029f7f874831cea3ad0556)
        - vMF (https://www.semanticscholar.org/paper/Spherical-Latent-Spaces-for-Stable-Variational-Xu-Durrett/94c83cecc357a45846d203c6401ede792e14bb0f)

    With the logistic normal prior and classifier on the latent representation, you effectively are using SCHOLAR with RNNs (https://www.semanticscholar.org/paper/Neural-Models-for-Documents-with-Metadata-Smith-Card/0373c63e291f83b22ea2836d876f94462e72e726)

    With a gaussian prior and a classifier on the encoder, you effectively are using M2 with RNNs (https://www.semanticscholar.org/paper/Semi-Supervised-Learning-with-Deep-Generative-Kingma-Rezende/58513e5043c8a8fb61dbe83ab58225e7f60575af)

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
        Dimension of the latent space. Note that different latent dimensions will affect your reconstruction perplexity.
    encoder : ``Encoder``
        VAE encoder to use. check ``modules.encoder`` for available encoders. For NVDM you should use a feedforward encoder.
    decoder : ``Decoder``
        VAE decoder to use. check ``modules.decoder`` for available decoders. For NVDM you should use a feedforward decoder.
    distribution : ``Distribution``
        VAE prior distribution to use. check ``modules.distribution`` for available distributions.
    embedder: ``TextFieldEmbedder``
        text field embedder to use. For NVDM, you should use a bag-of-word-counts embedder.
    classifier: ``Classifier``, optional (default = ``None``)
        if specified, will apply the corresponding classifier in the VAE to guide representation learning.
    nll_objective: ``str``, optional (default = ``lm``)
        One of ``lm`` or ``reconstruction``. This will set the objective of the decoder to either predict the next word (``lm``) or reconstruct the input (``reconstruction``)
    background_data_path: ``str``, optional (default = ``None``)
        Path to a file containing log frequency statistics on the vocabulary, which is used to bias the decoder.
        Important to get quality topics! This file is generated via ``bin.preprocess_data``, and will look something like
        ``/path/to/train.bgfreq.json``
    update_bg: ``bool``, optional (default = ``False``)
        Whether or not to update the background frequency bias during training.
    kl_weight_annealing: ``str``, optional (default = ``None``)
        If specified, will anneal kl divergence overtime, which was shown to be helpful to prevent KL collapse
        (https://www.semanticscholar.org/paper/Generating-Sentences-from-a-Continuous-Space-Bowman-Vilnis/3d1427961edccf8940a360d203e44539db58a60f)
        Options for weight annealing: ``constant``, ``sigmoid``, ``linear``
    encoder_input_dropout: ``float``, optional (default = ``0.5``)
        dropout applied to the input to the encoder
    decoder_input_dropout: ``float``, optional (default = ``0.5``)
        dropout applied to the input to the decoder (also called word dropout)
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
                 nll_objective: str = "lm",
                 background_data_path: str = None,
                 update_bg: bool = False,
                 kl_weight_annealing: str = None,
                 decoder_input_dropout: float = 0.5,
                 encoder_input_dropout: float = 0.5,
                 track_topics: bool = False,
                 topic_log_interval: int = 1000,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(NVRNN, self).__init__(vocab)

        self.vocab_namespace = "vae"
        tok2idx = vocab.get_token_to_index_vocabulary(self.vocab_namespace)
        self.pad_idx = tok2idx["@@PADDING@@"]
        self.unk_idx = tok2idx["@@UNKNOWN@@"]

        self.metrics = {
            'kld': Average(),
            'nll': Average(),
            'elbo': Average(),
        }
        
        if classifier is not None:
            self.metrics['accuracy'] = CategoricalAccuracy()

        

        if background_data_path is not None:
            bg = compute_background_log_frequency(background_data_path, vocab, self.vocab_namespace)
            if update_bg:
                self.bg = torch.nn.Parameter(bg, requires_grad=True)
            else:
                self.bg = torch.nn.Parameter(bg, requires_grad=False)
        else:
            bg = torch.FloatTensor(vocab.get_vocab_size(self.vocab_namespace))
            self.bg = torch.nn.Parameter(bg)
            torch.nn.init.uniform_(self.bg)

        self.track_topics = track_topics
        self.topic_log_interval = topic_log_interval
        self.step = 0
        self.batch_num = 0
        self.vocab = vocab
        self._embedder = embedder
        self._dist = distribution
        self.latent_dim = latent_dim
        self.embedding_dim = embedder.token_embedder_tokens.get_output_dim()
        self._encoder = encoder
        self._decoder = decoder
        self._encoder_input_dropout = torch.nn.Dropout(encoder_input_dropout)
        self._decoder_input_dropout = torch.nn.Dropout(decoder_input_dropout)
        self.kl_weight_annealing = kl_weight_annealing
        self._classifier = classifier
        self.nll_objective = nll_objective
        self._num_labels = vocab.get_vocab_size("labels")

        # we initialize parts of the decoder, classifier, and distribution here so we don't have to repeat
        # dimensions in the config, which can be cumbersome.

        self._encoder._initialize_encoder_architecture(self.embedding_dim)
        param_input_dim = self._encoder._architecture.get_output_dim()

        if self._classifier is not None:
            if self._classifier.input == 'encoder_output':
                param_input_dim += self._num_labels

        self._dist._initialize_params(param_input_dim, latent_dim)

        self._decoder._initialize_decoder_out(vocab.get_vocab_size(self.vocab_namespace))

        if self._classifier is not None:
            if self._classifier.input == 'theta':
                self._classifier._initialize_classifier_hidden(latent_dim)
            elif self._classifier.input == 'encoder_output':
                self._classifier._initialize_classifier_hidden(self._encoder._architecture.get_output_dim())
            self._classifier._initialize_classifier_out(self._num_labels)

        if kl_weight_annealing is not None:
            self.weight_scheduler = lambda x: schedule(x, kl_weight_annealing)
        else:
            self.weight_scheduler = lambda x: 1

        self._nll_loss = torch.nn.CrossEntropyLoss(ignore_index=self.pad_idx,
                                                   reduction="none")

        initializer(self)
    
    def _freeze_weights(self) -> None:
        """
        Freeze the weights of the model
        """

        model_parameters = dict(self.named_parameters())
        for item in model_parameters:
            model_parameters[item].requires_grad = False

    def drop_words(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        randomly replace tokens with <unk>. This is used in the decoder input
        to coax the model to use the latent representation.
        """
        tokens = tokens['tokens']
        prob = torch.rand(tokens.size()).to(tokens.device)
        prob[(tokens.data - self.pad_idx) == 0] = 1
        tokens_with_unks = tokens.clone()
        tokens_with_unks[prob < self._decoder_input_dropout.p] = self.unk_idx
        return {"tokens": tokens_with_unks}

    def extract_topics(self, k=20):
        """
        Given the learned (topic, vocabulary size) beta, print the
        top k words from each topic.

        Parameters
        __________

        k: ``int``, optional (default = 20)
            number of words to print per topic
        """
        decoder_weights = self._decoder._decoder_out.weight.data.transpose(0, 1)
        words = list(range(decoder_weights.size(1)))
        words = [self.vocab.get_token_from_index(i, self.vocab_namespace) for i in words]
        topics = []

        if self.bg is not None:
            word_strengths = list(zip(words, self.bg.tolist()))
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
            topics.append((i,  topic))
        return topics

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]=None,
                label: torch.IntTensor=None) -> Dict[str, torch.Tensor]:

        if not self.training:
            self.weight_scheduler = lambda x: 1.0
        else:
            self.weight_scheduler = lambda x: schedule(x, self.kl_weight_annealing)
        
        mask = get_text_field_mask(tokens)

        cont_repr = self._embedder(tokens)

        cont_repr = self._encoder_input_dropout(cont_repr)

        num_tokens = mask.sum().float()

        encoder_output = self._encoder(embedded_text=cont_repr, mask=mask)

        input_repr = [encoder_output['encoder_output']]

        if self._classifier is not None and label is not None:
            if self._classifier.input == 'encoder_output':
                clf_output = self._classifier(encoder_output['encoder_output'], label)
                input_repr.append(clf_output['label_repr'])

        input_repr = torch.cat(input_repr, 1)

        # use parameterized distribution to compute latent code and KL divergence
        _, kld, theta = self._dist.generate_latent_code(input_repr, n_sample=1)

        if self._classifier is not None and label is not None:
            if self._classifier.input == 'theta':
                clf_output = self._classifier(theta, label)

        decoder_input = self.drop_words(tokens)
        decoder_input = self._embedder(decoder_input)
        decoder_input = self._encoder_input_dropout(decoder_input)

        # decode using the latent code, background frequency, and embedded text.
        decoder_output = self._decoder(embedded_text=decoder_input,
                                       mask=mask,
                                       theta=theta,
                                       bg=self.bg)

        # to use the VAE as a feature embedder we also output the learned representations 
        # of the input text from various layers
        output = {
                    'activations': {
                        'encoder_output': encoder_output['encoded_docs'],
                        'theta': theta
                    },
                    'mask': mask
                 }

        if targets is not None:

            if self.nll_objective == 'lm':
                nll_loss = self._nll_loss(decoder_output['flattened_decoder_output'],
                                          targets['tokens'].view(-1))
            elif self.nll_objective == 'reconstruction':
                nll_loss = self._nll_loss(decoder_output['flattened_decoder_output'],
                                          tokens['tokens'].view(-1))
            # compute marginal likelihood
            nll = nll_loss.sum() / num_tokens
            
            kld_weight = self.weight_scheduler(self.batch_num)

            # add in the KLD to compute the ELBO
            kld = kld.to(nll.device) / num_tokens
            
            elbo = (nll + kld * kld_weight).mean()

            if self._classifier is not None and label is not None:
                elbo += clf_output['loss']

            avg_cos = check_dispersion(theta)

            output = {
                    'loss': elbo,
                    'elbo': elbo,
                    'nll': nll,
                    'kld': kld,
                    'kld_weight': kld_weight,
                    'avg_cos': float(avg_cos.mean()),
                    }

            if label is not None and self._classifier is not None:
                output['clf_loss'] = clf_output['loss']
                output['logits'] = clf_output['logits']

            self.metrics["elbo"](output['elbo'])
            self.metrics["kld"](output['kld'])
            self.metrics["kld_weight"] = output['kld_weight']
            self.metrics["nll"](output['nll'])
            self.metrics["perp"] = float(np.exp(self.metrics['nll'].get_metric()))
            self.metrics["cos"] = output['avg_cos']
            if self._classifier is not None:
                self.metrics['accuracy'](output['logits'], label)
            
        if self.track_topics:
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
