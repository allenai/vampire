from vae.common.util import compute_bow, sample, one_hot
from allennlp.modules import FeedForward
from allennlp.nn.util import get_text_field_mask
from allennlp.modules import Seq2SeqEncoder
from allennlp.common import Registrable
import torch
from typing import Dict

class Classifier(Registrable, torch.nn.Module):

    default_implementation='feedforward_generative'

    def forward(self, **kwargs):
        raise notImplementedError

@Classifier.register("feedforward_theta")
class FeedForwardTheta(Classifier):
    
    def __init__(self):
        super(FeedForwardTheta, self).__init__()
        self.input = "theta"
        self._classifier_loss = torch.nn.CrossEntropyLoss()
        
    def _initialize_classifier_hidden(self, input_dim: int):
        self._classifier_hidden = FeedForward(input_dim=input_dim,
                                              num_layers=1,
                                              hidden_dims=input_dim * 2,
                                              activations=torch.nn.ReLU())

    def _initialize_classifier_out(self, num_labels: int):
        self._num_labels = num_labels
        self._classifier_out = torch.nn.Linear(self._classifier_hidden.get_output_dim(),
                                               num_labels)

    def forward(self, theta: torch.FloatTensor, label: torch.IntTensor) -> Dict[str, torch.Tensor]:
        projection = self._classifier_hidden(theta)
        logits = self._classifier_out(projection)
        output = {"logits": logits}
        loss = self._classifier_loss(logits, label)
        output['loss'] = loss
        return output

@Classifier.register("feedforward_generative")
class FeedForwardGenerative(Classifier):
    
    def __init__(self):
        super(FeedForwardGenerative, self).__init__()
        self.input = "encoder_output"
        self._classifier_loss = torch.nn.CrossEntropyLoss()
        
    def _initialize_classifier_hidden(self, input_dim: int):
        self._classifier_hidden = FeedForward(input_dim=input_dim,
                                              num_layers=1,
                                              hidden_dims=input_dim * 2,
                                              activations=torch.nn.ReLU())

    def _initialize_classifier_out(self, num_labels: int):
        self._num_labels = num_labels
        self._classifier_out = torch.nn.Linear(self._classifier_hidden.get_output_dim(),
                                               num_labels)

    def forward(self, embedded_text: torch.FloatTensor, label: torch.IntTensor) -> Dict[str, torch.Tensor]:
        projection = self._classifier_hidden(embedded_text)
        logits = self._classifier_out(projection)
        output = {"logits": logits}
        gen_label = sample(logits, strategy="greedy")
        label_onehot = one_hot(gen_label, self._num_labels)
        output['label_repr'] = label_onehot
        loss = self._classifier_loss(logits, label)
        output['loss'] = loss
        return output
