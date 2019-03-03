from typing import Any, Dict, List, Optional

import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy

from vae.modules.encoder import Encoder

@Model.register("classifier")
class Classifier(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 input_embedder: TextFieldEmbedder,
                 encoder: Encoder = None,
                 dropout: float = None,
                 vae_embedding_dim: int = 0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab)
        self._input_embedder = input_embedder
        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._encoder = encoder
        self._num_labels = vocab.get_vocab_size(namespace="labels")
        if self._encoder:
            self._clf_input_dim = self._encoder.get_output_dim()
        else:
            self._clf_input_dim = self._input_embedder.get_output_dim()

        # Accomodates the stacked model when it includes pre-trained embeddings.
        self._clf_input_dim += vae_embedding_dim

        self._classification_layer = torch.nn.Linear(self._clf_input_dim,
                                                     self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None,  # pylint:disable=unused-argument
                vae_embedding: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
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
            A tensor of shape ``(batch_size, num_labels)`` representing
            unnormalized log probabilities of the label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_text = self._input_embedder(tokens)
        mask = get_text_field_mask(tokens).float()

        if self._encoder:
            embedded_text = self._encoder(embedded_text=embedded_text,
                                          mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        # Incorporate an additional VAE embedding for the deep generative model.
        if vae_embedding is not None:
            embedded_text = torch.cat([embedded_text, vae_embedding], dim=-1)

        label_logits = self._classification_layer(embedded_text)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(label_logits, label)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'accuracy': self._accuracy.get_metric(reset)}

        if hasattr(self._input_embedder, 'token_embedder_vae_tokens'):
            scalar_params = (self._input_embedder  # pylint:disable=protected-access
                             .token_embedder_vae_tokens
                             ._vae
                             .scalar_mix.scalar_parameters)
            scalar_mix = [float(weight.data.item()) for weight in scalar_params]
            layers = self._input_embedder.token_embedder_vae_tokens._layers # pylint:disable=protected-access
            for layer, mix in zip(layers, scalar_mix):
                metrics[layer + "_weight"] = mix

        return metrics
