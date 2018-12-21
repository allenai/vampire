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
from common.util import (schedule, compute_bow, split_instances, log_standard_categorical, 
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

    def __init__(self,
                 vocab: Vocabulary,
                 latent_dim: int,
                 hidden_dim: int,
                 encoder: Encoder,
                 decoder: Decoder,
                 distribution: Distribution,
                 continuous_embedder: TextFieldEmbedder,
                 reference_vocabulary: str=None,
                 reference_count_matrix: str=None,
                 pretrained_file: str=None,
                 freeze_weights: bool=False,
                 use_stopless_vocab: bool = False,
                 background_data_path: str = None,
                 update_bg : bool = False,
                 kl_weight_annealing: str = None,
                 word_dropout: float = 0.5,
                 dropout: float = 0.5,
                 tie_weights: bool = False,
                 track_topics: bool = False,
                 nll_objective: str = "lm",
                 classifier: Classifier = None,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(NVRNN, self).__init__(vocab)

        self.metrics = {
            'kld': Average(),
            'nll': Average(),
            'elbo': Average(),
            'npmi': Average()
        }
        
        if classifier is not None:
            self.metrics['accuracy'] = CategoricalAccuracy()

        if use_stopless_vocab:
            self.vocab_namespace = "stopless"
        else:
            self.vocab_namespace = "full"

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
        self.step = 0
        self.pad_idx = vocab.get_token_to_index_vocabulary(self.vocab_namespace)["@@PADDING@@"]
        self.unk_idx = vocab.get_token_to_index_vocabulary(self.vocab_namespace)["@@UNKNOWN@@"]
        self.sos_idx = vocab.get_token_to_index_vocabulary(self.vocab_namespace)["@@start@@"]
        self.eos_idx = vocab.get_token_to_index_vocabulary(self.vocab_namespace)["@@end@@"]
        self.vocab = vocab
        self._continuous_embedder = continuous_embedder
        self._masker = get_text_field_mask
        self._dist = distribution
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.embedding_dim = continuous_embedder.token_embedder_tokens.get_output_dim()
        self._encoder = encoder
        self._decoder = decoder
        self.dist_apply_batchnorm = self._dist._apply_batchnorm
        self.decoder_apply_batchnorm = self._decoder._apply_batchnorm
        self.batch_num = 0
        self.dropout = torch.nn.Dropout(dropout)
        self.word_dropout_ = word_dropout
        self.word_dropout = word_dropout
        self.kl_weight_annealing = kl_weight_annealing
        self._classifier = classifier
        self.tie_weights = tie_weights
        self.nll_objective = nll_objective
        if reference_vocabulary is not None and refrence_count_matrix is not None:
            self.ref_vocab = read_text(reference_vocabulary)
            self.ref_counts = load_sparse(reference_count_matrix).todense()
        self._num_labels = vocab.get_vocab_size("labels")
        if self.tie_weights:
            if self.hidden_dim != self.embedding_dim:
                raise ConfigurationError('When using the tied flag, '
                                         'hidden_dim must be equal to embedding_dim')
            self.decoder.weight = self.encoder.weight

        self._encoder._initialize_encoder_architecture(self.embedding_dim)

        if self._classifier is not None:
            if self._classifier.input == 'theta':
                self._classifier._initialize_classifier_hidden(latent_dim)
            elif self._classifier.input == 'encoder_output':
                self._classifier._initialize_classifier_hidden(self._encoder._architecture.get_output_dim())
            self._classifier._initialize_classifier_out(self._num_labels)
        
        
        
        # we initialize parts of the decoder, classifier, and distribution here so we don't have to repeat
        # dimensions in the config, which can be cumbersome.
        
        param_input_dim = self._encoder._architecture.get_output_dim()

        if self._classifier is not None:
            if self._classifier.input == 'encoder_output':
                param_input_dim += self._num_labels
        
        self._dist._initialize_params(param_input_dim, latent_dim)
        
        self._decoder._initialize_decoder_out(vocab.get_vocab_size(self.vocab_namespace))

        if kl_weight_annealing is not None:
            self.weight_scheduler = lambda x: schedule(x, kl_weight_annealing)
        else:
            self.weight_scheduler = lambda x: 1
        self._reconstruction_loss = torch.nn.CrossEntropyLoss(ignore_index=self.pad_idx,
                                                              reduction="none")
        if pretrained_file is not None:
            archive = load_archive(pretrained_file)
            self._initialize_weights_from_archive(archive, freeze_weights)
        else:
            initializer(self)

    def _initialize_weights_from_archive(self,
                                         archive: Archive,
                                         freeze_weights: bool = False) -> None:
        """
        Initialize weights from a model archive.

        Params
        ______
        archive : `Archive`
            pretrained model archive
        """
        model_parameters = dict(self.named_parameters())
        archived_parameters = dict(archive.model.named_parameters())
        for item, val in archived_parameters.items():
            new_weights = val.data
            model_parameters[item].data.copy_(new_weights)
            if freeze_weights:
                model_parameters[item].requires_grad = False
    
    def _freeze_weights(self) -> None:
        """
        Freeze the weights of the model
        """

        model_parameters = dict(self.named_parameters())
        for item in model_parameters:
            model_parameters[item].requires_grad = False        

    def drop_words(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # randomly tokens with <unk>
        tokens = tokens['tokens']
        prob = torch.rand(tokens.size()).to(tokens.device)
        prob[(tokens.data - self.sos_idx) * (tokens.data - self.pad_idx) * (tokens.data - self.eos_idx) == 0] = 1
        tokens_with_unks = tokens.clone()
        tokens_with_unks[prob < self.word_dropout] = self.unk_idx
        return {"tokens": tokens_with_unks}

    def compute_npmi(self, topics, n=20, cols_to_skip=0):
        """
        compute NPMI
        """
        vocab_index = dict(zip(self.ref_vocab, range(len(self.ref_vocab))))
        ref_counts = self.ref_counts
        n_docs, _ = ref_counts.shape
        npmi_means = []
        for topic in topics[1:]:
            words = topic[1][cols_to_skip:]
            npmi_vals = []
            for word_i, word1 in enumerate(words[:n]):
                if word1 in vocab_index:
                    index1 = vocab_index[word1]
                else:
                    index1 = None
                for word2 in words[word_i+1:n]:
                    if word2 in vocab_index:
                        index2 = vocab_index[word2]
                    else:
                        index2 = None
                    if index1 is None or index2 is None:
                        npmi = 0.0
                    else:
                        col1 = np.array(ref_counts[:, index1] > 0, dtype=int)
                        col2 = np.array(ref_counts[:, index2] > 0, dtype=int)
                        c1 = col1.sum()
                        c2 = col2.sum()
                        c12 = np.sum(col1 * col2)
                        if c12 == 0:
                            npmi = 0.0
                        else:
                            npmi = ((np.log10(n_docs) + np.log10(c12) - np.log10(c1) - np.log10(c2)) 
                                    / (np.log10(n_docs) - np.log10(c12)))
                    npmi_vals.append(npmi)
            npmi_means.append(np.mean(npmi_vals))
        return np.mean(npmi_means)

    def extract_topics(self, k=20):
        """
        Given the learned (topic, vocabulary size) beta, print the
        top k words from each topic.
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
    
    def run(self,
            tokens: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor]=None,
            label: torch.IntTensor=None) -> Dict[str, torch.Tensor]:
        """
        Run one step of VAE with RNN decoder
        """

        if not self.training:
            self.weight_scheduler = lambda x: 1.0
            if self.word_dropout_ < 1.0:
                self.word_dropout = 0.0
            self._dist._apply_batchnorm = self.dist_apply_batchnorm
            self._decoder._apply_batchnorm = self.decoder_apply_batchnorm
        else:
            self.weight_scheduler = lambda x: schedule(x, self.kl_weight_annealing)
            self.word_dropout = self.word_dropout_
            self._dist._apply_batchnorm = False
            self._decoder._apply_batchnorm = False

        output = {}
        batch_size, _ = tokens['tokens'].shape
        
        mask = self._masker(tokens)

        # encode tokens
        cont_repr = self._continuous_embedder(tokens)
        cont_repr = self.dropout(cont_repr)
        encoder_output = self._encoder(embedded_text=cont_repr, mask=mask)


        # concatenate generated labels and continuous document vecs as input representation
        input_repr = [encoder_output['encoder_output']]

        if label is not None:
            if self._classifier is not None:
                if self._classifier.input == 'encoder_output':
                    clf_output = self._classifier(encoder_output['encoder_output'], label)
                    input_repr.append(clf_output['label_repr'])
        
        input_repr = torch.cat(input_repr, 1)

        # use parameterized distribution to compute latent code and KL divergence
        _, kld, theta = self._dist.generate_latent_code(input_repr, n_sample=1)
        
        if label is not None:
            if self._classifier is not None:
                if self._classifier.input == 'theta':
                    clf_output = self._classifier(theta, label)

        decoder_input = self.drop_words(tokens)
        decoder_input = self._continuous_embedder(decoder_input)
        decoder_input = self.dropout(decoder_input)
        # decode using the latent code.
        decoder_output = self._decoder(embedded_text=decoder_input,
                                       mask=mask,
                                       theta=theta,
                                       bg=self.bg)
        
        if targets is not None:
            
            num_tokens = mask.sum().float()
            if self.nll_objective == 'lm':
                reconstruction_loss = self._reconstruction_loss(decoder_output['flattened_decoder_output'],
                                                                targets['tokens'].view(-1))
            elif self.nll_objective == 'reconstruction':
                reconstruction_loss = self._reconstruction_loss(decoder_output['flattened_decoder_output'],
                                                                tokens['tokens'].view(-1))
            # compute marginal likelihood
            nll = reconstruction_loss.sum() / num_tokens
            
            kld_weight = self.weight_scheduler(self.batch_num)

            # add in the KLD to compute the ELBO
            kld = kld.to(nll.device) / num_tokens
            
            elbo = (nll + kld * kld_weight).mean()

            if label is not None and self._classifier is not None:
                elbo += clf_output['loss']

            avg_cos = check_dispersion(theta)

            output = {
                    'elbo': elbo,
                    'nll': nll,
                    'kld': kld,
                    'kld_weight': kld_weight,
                    'avg_cos': float(avg_cos.mean()),
                    }

            if label is not None and self._classifier is not None:
                output['clf_loss'] = clf_output['loss']
                output['logits'] = clf_output['logits']
            
            
            # if self.track_topics:
            #     if self.step == 1:
            #         topics = self.extract_topics()
            #         self.metrics['npmi'](self.compute_npmi(topics))
            #         tqdm.write(tabulate(topics))
            #         self.step = 0
            #     else:
            #         self.step += 1

        return output
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        output = {}
        for metric_name, metric in self.metrics.items():
            if isinstance(metric, float):
                output[metric_name] = metric
            else:
                output[metric_name] = float(metric.get_metric(reset))
        return output

    def merge(self, labeled_output=None, unlabeled_output=None):
        output = defaultdict(list)
        if unlabeled_output is not None:
            output['elbo'].append(unlabeled_output['elbo'])
            output['kld'].append(unlabeled_output['kld'])
            output['nll'].append(unlabeled_output['nll'])
            output['avg_cos'].append(unlabeled_output['avg_cos'])
            output['kld_weight'].append(unlabeled_output['kld_weight'])

        if labeled_output is not None:
            output['clf_loss'].append(labeled_output['clf_loss'])
            output['logits'].append(labeled_output['logits'])
            output['elbo'].append(labeled_output['elbo'])
            output['kld'].append(labeled_output['kld'])
            output['nll'].append(labeled_output['nll'])
            output['avg_cos'].append(labeled_output['avg_cos'])
            output['kld_weight'].append(labeled_output['kld_weight'])

        for key, item in output.items():
            output[key] = sum(item) / len(item)
        return output

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                is_labeled: torch.IntTensor,
                targets: Dict[str, torch.Tensor]=None,
                label: torch.IntTensor=None) -> Dict[str, torch.Tensor]: 
        is_labeled_tokens=torch.Tensor(np.array([int(self.vocab.get_token_from_index(x.item(), namespace="is_labeled")) for x in is_labeled]))
        supervised_instances, unsupervised_instances = split_instances(tokens=tokens, label=label, is_labeled=is_labeled_tokens, targets=targets)

        if supervised_instances.get('tokens') is not None:
            labeled_output = self.run(**supervised_instances)
        else:
            labeled_output = None

        if unsupervised_instances.get('tokens') is not None:
            unlabeled_output = self.run(**unsupervised_instances)
        else:
            unlabeled_output = None

        output = self.merge(labeled_output, unlabeled_output)

        self.metrics["elbo"](output['elbo'])
        self.metrics["kld"](output['kld'])
        self.metrics["kld_weight"] = output['kld_weight']
        self.metrics["nll"](output['nll'])
        self.metrics["perp"] = float(np.exp(self.metrics['nll'].get_metric()))
        self.metrics["cos"] = output['avg_cos']

        if self._classifier is not None and supervised_instances.get('tokens') is not None:
            self.metrics['accuracy'](output['logits'], supervised_instances['label'])
        
        output['loss'] = output['elbo']
        self.batch_num += 1
        return output