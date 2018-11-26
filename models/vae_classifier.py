from typing import Dict
import numpy as np
import torch
from allennlp.models.model import Model
from modules.vae import VAE
from allennlp.data import Vocabulary
from allennlp.nn import InitializerApplicator
from overrides import overrides
from allennlp.training.metrics import CategoricalAccuracy, Average
from allennlp.modules import FeedForward


@Model.register("vae_classifier")
class VAE_CLF(Model):
    """
    Perform text classification with a VAE

    Params
    ______

    vocab: ``Vocabulary``
        vocabulary
    vae : ``VAE``
        variational autoencoder (RNN or BOW-based)
    classifier: ``FeedForward``
        feedforward network classifying input
    """
    def __init__(self, 
                 vocab: Vocabulary,
                 vae: VAE,
                 classifier: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator()):
        super(VAE_CLF, self).__init__(vocab)
        self.metrics = {
            'l_kld': Average(),
            'u_kld': Average(),
            'l_recon': Average(),
            'u_recon': Average(),
            'l_nll': Average(),
            'u_nll': Average(),
            'accuracy': CategoricalAccuracy(),
            'elbo': Average(),
        }
        self._num_labels = vocab.get_vocab_size("labels")
        self.unlabeled_index = self.vocab.get_token_to_index_vocabulary("labels")["-1"] if self.vocab.get_token_to_index_vocabulary("labels").get("-1") is not None else -1
        # if vocab.get_token_to_index_vocabulary("labels").get("-1") is not None:
        #     self._num_labels = self._num_labels - 1
        self._vae = vae
        self._classifier = classifier
        self._classifier_loss = torch.nn.CrossEntropyLoss()
        self._output_logits = torch.nn.Linear(self._classifier.get_output_dim(), self._num_labels)
        initializer(self)

    @overrides
    def forward(self, tokens, label, **metadata):  # pylint: disable=W0221
        """
        Given tokens and labels, generate document representation with
        a latent code and classify.
        """

        # run VAE to decode with a latent code
        vae_output = self._vae(tokens, label, **metadata)

        if self._vae.__class__.__name__ in ('RNN_VAE', 'SCHOLAR_RNN'):
            decoded_output = vae_output.get('decoded_output')
            if decoded_output is not None:
                document_vectors = torch.max(decoded_output, 1)[0]
                is_labeled = (label != self.unlabeled_index).nonzero().squeeze()
                output = self._classifier(document_vectors)
                logits = self._output_logits(output)
                label = label[is_labeled]
                if len(logits.shape) == 1:
                    try:
                        logits = logits.unsqueeze(0)
                    except:
                        import ipdb; ipdb.set_trace()
                if len(label.shape) == 0:
                    label = label.unsqueeze(0)
                classifier_loss = self._classifier_loss(logits, label)
                self.metrics['accuracy'](logits, label)
            else:
                classifier_loss = 0
        else:
            document_vectors = vae_output['decoded_output'].squeeze(0)
            

        # set metrics
        l_recon = vae_output.get('l_recon', np.zeros(1))
        u_recon = vae_output.get('u_recon', np.zeros(1))
        elbo = vae_output['elbo']
        u_kld = vae_output.get('u_kld', np.zeros(1))
        l_kld = vae_output.get('l_kld', np.zeros(1))
        l_nll = vae_output.get('l_nll', np.zeros(1))
        u_nll = vae_output.get('u_nll', np.zeros(1))
        self.metrics["l_recon"](l_recon.mean())
        self.metrics["u_recon"](u_recon.mean())
        self.metrics["elbo"](elbo.mean())
        self.metrics["l_kld"](l_kld.mean())
        self.metrics["u_kld"](u_kld.mean())
        self.metrics["l_nll"](l_nll.mean())
        self.metrics["u_nll"](u_nll.mean())
        # create clf_output
        clf_output = vae_output
        clf_output['loss'] = vae_output['elbo'] + classifier_loss

        return clf_output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: float(metric.get_metric(reset)) for metric_name, metric in self.metrics.items()}
