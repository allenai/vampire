import torch
import numpy as np

from typing import Dict, List, Iterable

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary


@Vocabulary.register("custom_vocabulary")
class VocabularyWithPretrainedVAE(Vocabulary):
    """
    Augment the allennlp Vocabulary with filtered vocabulary
    Idea: override from_params to "set" the vocab from a file before
    constructing in a normal fashion.
    """
    @classmethod
    def from_params(cls, params: Params, instances: Iterable['adi.Instance'] = None):
        supervised_vocab_file = params.pop('supervised_vocab_file')
        vae_vocab_file = params.pop('vae_vocab_file')
        label_file = params.pop('label_file')
        non_padded_namespaces = params.pop('non_padded_namespaces')
        vocab = cls(non_padded_namespaces=non_padded_namespaces)

        #if `filtered_vocab_file` is a URL, redirect to the cache
        supervised_vocab_file = cached_path(supervised_vocab_file)
        vocab.set_from_file(filename=supervised_vocab_file, namespace="tokens", is_padded=not "tokens" in non_padded_namespaces, oov_token="@@UNKNOWN@@")
        # if `full_vocab_file` is a URL, redirect to the cache
        vae_vocab_file = cached_path(vae_vocab_file)
        vocab.set_from_file(filename=vae_vocab_file, namespace="vae", is_padded=not "vae" in non_padded_namespaces, oov_token="@@UNKNOWN@@")
        label_file = cached_path(label_file)
        vocab.set_from_file(filename=label_file, namespace="labels", is_padded=False)
        return vocab