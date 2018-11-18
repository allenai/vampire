import torch
import numpy as np

from typing import Dict, List, Iterable

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary


@Vocabulary.register("vocabulary_with_vae")
class VocabularyWithVAE(Vocabulary):
    """
    Augment the allennlp Vocabulary with a pre-trained LM

    Idea: override from_params to "set" the vocab from a file before
    constructing in a normal fashion.
    """
    @classmethod
    def from_params(cls, params: Params, instances: Iterable['adi.Instance'] = None):
        # set the VAE piece
        stopless_vocab_file = params.pop('stopless_vocab_file')
        full_vocab_file = params.pop('full_vocab_file')
        oov_token = params.pop('oov_token')
        namespace = params.pop('namespace', 'stopless')
        vocab = super(VocabularyWithVAE, cls).from_params(params, instances)
        # if `stopless_vocab_file` is a URL, redirect to the cache
        stopless_vocab_file = cached_path(stopless_vocab_file)
        vocab.set_from_file(stopless_vocab_file, namespace=namespace)
        namespace = params.pop('namespace', 'full')
        # if `stopless_vocab_file` is a URL, redirect to the cache
        full_vocab_file = cached_path(full_vocab_file)
        vocab.set_from_file(full_vocab_file, namespace=namespace)
        vocab.set_from_file(full_vocab_file, namespace="tokens")
        return vocab
