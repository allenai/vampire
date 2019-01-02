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
from common.util import schedule, compute_bow, log_standard_categorical, check_dispersion, compute_background_log_frequency, split_instances
from typing import Dict
from allennlp.training.metrics import CategoricalAccuracy, Average
from tabulate import tabulate
from modules import Classifier
from common.file_handling import load_sparse, read_text
from tqdm import tqdm
from collections import defaultdict

@Model.register("nvdm")
class NVDM(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 latent_dim: int,
                 hidden_dim: int,
                 encoder: Encoder,
                 decoder: Decoder,
                 distribution: Distribution,
                 onehot_embedder: TextFieldEmbedder,
                 reference_vocabulary: str=None,
                 reference_count_matrix: str=None,
                 continuous_embedder: TextFieldEmbedder=None,
                 use_stopless_vocab: bool = False,
                 background_data_path: str = None,
                 update_bg : bool = False,
                 kl_weight_annealing: str = None,
                 dropout: float = 0.5,
                 track_topics: bool = False,
                 topic_log_interval: int = 100,
                 tie_weights: bool = False,
                 classifier: Classifier = None,
                 pretrained_file: str = None,
                 freeze_weights: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(NVDM, self).__init__(vocab)

        self.metrics = {
            'kld': Average(),
            'nll': Average(),
            'elbo': Average(),
        }
        if use_stopless_vocab:
            self.vocab_namespace = "stopless"
        else:
            self.vocab_namespace = "full"

        if classifier is not None:
            self.metrics['accuracy'] = CategoricalAccuracy()
        
        self.track_topics = track_topics
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
        
        self.vocab = vocab
        self._onehot_embedder = onehot_embedder
        self._continuous_embedder = continuous_embedder
        self._masker = get_text_field_mask
        self._dist = distribution
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.topic_log_interval = topic_log_interval
        if continuous_embedder is not None:
            self.embedding_dim = continuous_embedder.token_embedder_tokens.get_output_dim()
        else:
            self.embedding_dim = onehot_embedder.token_embedder_tokens.get_output_dim()
        self._encoder = encoder
        self._decoder = decoder
        self.dist_apply_batchnorm  =  self._dist._apply_batchnorm
        self.decoder_apply_batchnorm = self._decoder._apply_batchnorm
        self.tie_weights = tie_weights
        
        if self.tie_weights:
            if self.hidden_dim != self.embedding_dim:
                raise ConfigurationError('When using the tied flag, '
                                         'hidden_dim must be equal to embedding_dim')
            self.decoder.weight = self.encoder.weight

        self.step = 0
        self.batch_num = 0
        self.dropout = torch.nn.Dropout(dropout)
        self.kl_weight_annealing = kl_weight_annealing
        self._classifier = classifier
        if reference_vocabulary is not None and reference_count_matrix is not None:
            self.ref_vocab = read_text(reference_vocabulary)
            self.ref_counts = load_sparse(reference_count_matrix).todense()

        # we initialize parts of the decoder, classifier, and distribution here so we don't have to repeat
        # dimensions in the config, which can be cumbersome.
    
        self._encoder._initialize_encoder_architecture(self.embedding_dim)
        
        param_input_dim = self._encoder._architecture.get_output_dim() 
        if self._classifier is not None:
            if self._classifier.input == 'theta':
                self._classifier._initialize_classifier_hidden(latent_dim)
            elif self._classifier.input == 'encoder_output':
                self._classifier._initialize_classifier_hidden(self._encoder._architecture.get_output_dim())
            self._classifier._initialize_classifier_out(vocab.get_vocab_size("labels"))

        if self._classifier is not None:
            if self._classifier.input == 'encoder_output':
                param_input_dim += vocab.get_vocab_size("labels")

        self._dist._initialize_params(param_input_dim, latent_dim)

        self._decoder._initialize_decoder_out(latent_dim, vocab.get_vocab_size(self.vocab_namespace))
        
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
            if freeze_weights and "classifier" not in item:
                model_parameters[item].requires_grad = False
    
    def _freeze_weights(self) -> None:
        """
        Freeze the weights of the model
        """

        model_parameters = dict(self.named_parameters())
        for item in model_parameters:
            model_parameters[item].requires_grad = False

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
                            npmi = (np.log10(n_docs) + np.log10(c12) - np.log10(c1) - np.log10(c2)) / (np.log10(n_docs) - np.log10(c12))
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
        
        output = {}
        input_repr = []
        batch_size, seq_len = tokens['tokens'].shape
        mask = self._masker(tokens)
        onehot_repr = self._onehot_embedder(tokens)
        onehot_repr = self.dropout(onehot_repr)
        num_tokens = onehot_repr.sum()
        
        if self._continuous_embedder is not None:
            cont_repr = self._continuous_embedder(tokens)
            cont_repr = self.dropout(cont_repr)
            encoder_output = self._encoder(embedded_text=cont_repr, mask=mask)
            input_repr.append(encoder_output['encoder_output'])
        else:
            encoder_output = self._encoder(embedded_text=onehot_repr, mask=mask)
            input_repr.append(encoder_output['encoder_output'])

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

        # decode using the latent code.
        decoder_output = self._decoder(theta=theta,
                                       bg=self.bg)
        
        if targets is not None:
            
            decoder_probs = torch.nn.functional.log_softmax(decoder_output['decoder_output'], dim=1)
            error = torch.mul(onehot_repr, decoder_probs)
            reconstruction_loss = -torch.sum(error)
            # compute marginal likelihood
            nll = reconstruction_loss / num_tokens
            
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
                    'avg_cos': float(avg_cos.mean())
                    }

            if self._classifier is not None and label is not None:
                output['clf_loss'] = clf_output['loss']
                output['logits'] = clf_output['logits']

        if self.track_topics and self.training:
            if self.step == self.topic_log_interval:
                topics = self.extract_topics()
                # self.metrics['npmi'](self.compute_npmi(topics))
                tqdm.write(tabulate(topics))
                self.step = 0
            else:
                self.step += 1

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
            output['logits'] = labeled_output['logits']
            output['elbo'].append(labeled_output['elbo'])
            output['kld'].append(labeled_output['kld'])
            output['nll'].append(labeled_output['nll'])
            output['avg_cos'].append(labeled_output['avg_cos'])
            output['kld_weight'].append(labeled_output['kld_weight'])

        for key, item in output.items():
            if key != 'logits':
                output[key] = sum(item) / len(item)
        return output

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                is_labeled: torch.IntTensor,
                targets: Dict[str, torch.Tensor]=None,
                label: torch.IntTensor=None) -> Dict[str, torch.Tensor]: 
        if not self.training:
            self.weight_scheduler = lambda x: 1.0
            self._dist._apply_batchnorm = self.dist_apply_batchnorm
            self._decoder._apply_batchnorm = self.decoder_apply_batchnorm
        else:
            self.weight_scheduler = lambda x: schedule(x, self.kl_weight_annealing)
            self._encoder._apply_batchnorm = False
            self._decoder._apply_batchnorm = False

        is_labeled_tokens=torch.Tensor(np.array([int(self.vocab.get_token_from_index(x.item(), namespace="is_labeled")) for x in is_labeled]))
        supervised_instances, unsupervised_instances = split_instances(tokens=tokens, label=label, is_labeled=is_labeled_tokens, targets=targets)
        
        if (supervised_instances.get('tokens') is not None and supervised_instances['tokens']['tokens'].shape[0] > 1 
            or supervised_instances.get('tokens') is not None and not self.training):
            labeled_output = self.run(**supervised_instances)
            clf = 1
        else:
            clf = 0
            labeled_output = None

        if unsupervised_instances.get('tokens') is not None and unsupervised_instances['tokens']['tokens'].shape[0] > 1:
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

        if self._classifier is not None and clf == 1:
            self.metrics['accuracy'](output['logits'], supervised_instances['label'])
        
        output['loss'] = output['elbo']
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

