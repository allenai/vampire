from typing import Dict, Optional, List, Any
import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules import TextFieldEmbedder
from allennlp.modules import Seq2SeqEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask, masked_mean, get_final_encoder_states
from allennlp.models.archival import load_archive, Archive
from modules.vae import VAE
import os

@Model.register("seq2seq_classifier")
class Seq2SeqClassifier(Model):
    """
    This ``Model`` implements a Seq2Seq classifier. See allennlp.modules.seq2seq_classifier for available encoders.
    By default, this model runs a maxpool over the output of the seq2seq encoder.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    encoder : ``Seq2SeqEncoder``
        Used to encode the text
    output_feedforward : ``FeedForward``
        Used to prepare the text for prediction.
    output_logit : ``FeedForward``
        This feedforward network computes the output logits.
    dropout : ``float``, optional (default=0.5)
        Dropout percentage to use.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 output_logit: FeedForward,
                 dropout: float = 0.2,
                 pretrained_vae_file: str = None,
                 freeze_pretrained_weights: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self._encoder = encoder
        self._output_logit = output_logit

        self._num_labels = vocab.get_vocab_size(namespace="labels")

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)
        
        if pretrained_vae_file is not None:
            archive = load_archive(pretrained_vae_file)
            self._vae = archive.model._vae
            self._vae.vocab = vocab
            self._vae._unlabel_index = None
            if freeze_pretrained_weights:
                self._vae._freeze_weights()
        else:
            self._vae = None

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None  # pylint:disable=unused-argument
               ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Metadata to persist

        Returns
        -------
        An output dictionary consisting of:

        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_text = self._text_field_embedder(tokens)
        
        mask = get_text_field_mask(tokens).float() 

        
        encoder_output = self._encoder(embedded_text, mask)
        
        vecs = []

        broadcast_mask = mask.unsqueeze(-1).float()
        context_vectors = encoder_output * broadcast_mask

        vecs.append(masked_mean(context_vectors, broadcast_mask, dim=1, keepdim=False))
        
        if self._vae is not None:
            embedded_text_ = self._vae._embedder(tokens)
            mask_ = self._vae._masker(tokens)
            encoder_output = self._vae._encoder(embedded_text=embedded_text_, mask=mask_)
            classifier_output = self._vae._classifier(input=encoder_output['encoder_output'],
                                                      label=label)
            master = torch.cat([encoder_output['encoder_output'], classifier_output['label_repr']], 1)
            # theta = torch.ones([tokens['tokens'].shape[0], self._vae.latent_dim]).to(master.device)
            _, _, theta = self._vae._dist.generate_latent_code(master, n_sample=1)
            decoder_output = self._vae._decoder(embedded_text=embedded_text, theta=theta, mask=mask_)
            broadcast_mask = mask.unsqueeze(-1).float()
            context_vectors = decoder_output['decoder_output'] * broadcast_mask
            vecs.append(masked_mean(context_vectors, broadcast_mask, dim=1, keepdim=False))
            # lat_to_cat = (theta.unsqueeze(0).expand(embedded_text.shape[1], embedded_text.shape[0], -1)
            #                             .permute(1, 0, 2)
            #                             .contiguous())
            # master_to_cat = (master.unsqueeze(0)
            #                     .expand(embedded_text.shape[1], embedded_text.shape[0], -1)
            #                     .permute(1, 0, 2)
            #                     .contiguous())
            # import ipdb; ipdb.set_trace()
            # vecs.append(decoder_output['decoder_output'])
        pooled_embeddings = torch.cat(vecs, dim=1)

            
            # vecs.append(master)
            # vecs.append(theta)

            
        
        # encoder_output = get_final_encoder_states(encoder_output, mask, self._encoder.is_bidirectional())

        # maxpool
        if self.dropout:
            pooled_embeddings = self.dropout(pooled_embeddings)

        label_logits = self._output_logit(pooled_embeddings)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss
        
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}

