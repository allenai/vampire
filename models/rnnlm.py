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
from common.util import schedule, compute_bow, log_standard_categorical, check_dispersion, compute_background_log_frequency
from typing import Dict
from allennlp.training.metrics import CategoricalAccuracy, Average
from modules import Classifier


@Model.register("RNNLM")
class RNNLM(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 hidden_dim: int,
                 encoder: Encoder,
                 decoder: Decoder,
                 dropout: float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(RNNLM, self).__init__(vocab)

        self.metrics = {
            'nll': Average(),
        }

        self.pad_idx = vocab.get_token_to_index_vocabulary("full")["@@PADDING@@"]
        self.unk_idx = vocab.get_token_to_index_vocabulary("full")["@@UNKNOWN@@"]
        self.sos_idx = vocab.get_token_to_index_vocabulary("full")["@@start@@"]
        self.eos_idx = vocab.get_token_to_index_vocabulary("full")["@@end@@"]
        self.vocab = vocab
        self._embedder = text_field_embedder
        self._masker = get_text_field_mask
        self.hidden_dim = hidden_dim
        self._encoder = encoder
        self._decoder = decoder
        self.dropout = torch.nn.Dropout(dropout)

        embedding_dim = text_field_embedder.token_embedder_tokens.get_output_dim()
        
        # we initialize parts of the decoder, classifier, and distribution here so we don't have to repeat
        # dimensions in the config, which can be cumbersome.                
        self._decoder._initialize_decoder_out(vocab.get_vocab_size("full"))

        self._reconstruction_loss = torch.nn.CrossEntropyLoss(ignore_index=self.pad_idx,
                                                              reduction="none")
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
            item_sub = ".".join(item.split('.')[1:])
            model_parameters[item_sub].data.copy_(new_weights)
            if freeze_weights:
                item_sub = ".".join(item.split('.')[1:])
                model_parameters[item_sub].requires_grad = False
    
    def _freeze_weights(self) -> None:
        """
        Freeze the weights of the model
        """

        model_parameters = dict(self.named_parameters())
        for item in model_parameters:
            model_parameters[item].requires_grad = False

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]=None,
                label: torch.IntTensor=None) -> Dict[str, torch.Tensor]:
        """
        Run one step of VAE with RNN decoder
        """
        output = {}
        batch_size, _ = tokens['tokens'].shape

        encoder_input = self._embedder(tokens)
        
        encoder_input = self.dropout(encoder_input)

        mask = self._masker(tokens)
        
        # encode tokens
        encoder_output = self._encoder(embedded_text=encoder_input, mask=mask)

        encoder_output['encoded_docs'] = self.dropout(encoder_output['encoded_docs'])

        # decode tokens
        decoder_output = self._decoder(embedded_text=encoder_output['encoded_docs'],
                                       mask=mask)
        
        if targets is not None:
            
            num_tokens = mask.sum().float()
            reconstruction_loss = self._reconstruction_loss(decoder_output['flattened_decoder_output'],
                                                            targets['tokens'].view(-1))

            # compute marginal likelihood
            nll = reconstruction_loss.sum() / num_tokens
                        
            loss = nll.mean()

            output = {
                    'loss': loss,
                    'nll': nll,
                    'perplexity': torch.exp(nll),
                    }
            self.metrics["nll"](output['nll'])
            self.metrics["perp"] = float(np.exp(self.metrics['nll'].get_metric()))

        output['encoded_docs'] = encoder_output['encoded_docs']
        output['decoder_output'] = decoder_output
        return output
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        output = {}
        for metric_name, metric in self.metrics.items():
            if isinstance(metric, float):
                output[metric_name] = metric
            else:
                output[metric_name] = float(metric.get_metric(reset))
        return output
