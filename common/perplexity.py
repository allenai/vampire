from typing import Optional

from overrides import overrides
import torch

from allennlp.training.metrics.metric import Metric


@Metric.register("perplexity")
class Perplexity(Metric):
    """
    Computes per-word perplexity for a validation / test corpus.
    Raw probabilities predicted for each ground truth token in the corpus is collected. The average
    log probability sum l = - 1/M * sum(log_2 P(corpus)) where P(corpus) is equal to the joint
    probablity of the tokens in the corpus. The metric reported is then PP = 2^l.
    Without resetting, perplexity over the entire corpus is computed. When allowing resetting,
    computes per-batch perplexity treating the current batch of words as a corpus itself.
    """
    def __init__(self) -> None:
        self._log_probs_sum = 0.0
        self._num_tokens = 0

    @overrides
    def __call__(self,  # type: ignore
                 logits: torch.Tensor,
                 targets: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        logits : ``torch.Tensor``, required.
            A tensor of unnormalized log probabilities of shape (batch_size, sequence_length, num_classes).
        targets: ``torch.Tensor``, required.
            A tensor of indices of shape (batch, sequence_length) representing the ground truth.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor of shape (batch_size, sequence_length).
        """
        # placeholder for BOW till I get this working...

        num_classes = logits.size(-1)
        
        if mask is None:
            mask = torch.ones(logits.size()[:-1])
        else:
            logits = logits[mask.view(-1).byte(), :]
            targets = torch.masked_select(targets.view(-1), mask.view(-1).byte())

        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = torch.log2(probs)

        # Select only the log likelihoods of the ground truth.
        targets = targets.view(-1, 1)
        log_probs = log_probs.view(-1, num_classes)

        # At training time, targets may be on the GPU.
        log_probs = log_probs.to(device=targets.device)

        log_probs = log_probs.gather(-1, targets).squeeze()
        log_probs_sum = log_probs.sum(-1)

        # Grab the value from the 0d tensor.
        log_probs_sum = log_probs_sum.item()

        self._log_probs_sum += log_probs_sum
        self._num_tokens += mask.sum().item()

    @overrides
    def get_metric(self, reset: bool = True):
        """
        Returns
        -------
        The reported perplexity 2^l where l is the negative average log probability
        of the input.
        """
        negative_average_log_prob_sum = - self._log_probs_sum / self._num_tokens
        perplexity = 2 ** negative_average_log_prob_sum
        if reset:
            self.reset()

        return perplexity

    @overrides
    def reset(self):
        self._log_probs_sum = 0.0
        self._num_tokens = 0