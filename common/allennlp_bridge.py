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
    Augment the allennlp Vocabulary with filtered vocabulary
    Idea: override from_params to "set" the vocab from a file before
    constructing in a normal fashion.
    """
    @classmethod
    def from_params(cls, params: Params, instances: Iterable['adi.Instance'] = None):
        stopword_file = params.pop('stopword_file')
        full_vocab_file = params.pop('full_vocab_file')
        oov_token = params.pop('oov_token')
        namespace = params.pop('namespace', 'filtered')
        vocab = super(VocabularyWithVAE, cls).from_params(params, instances)
        #if `filtered_vocab_file` is a URL, redirect to the cache
        filtered_vocab_file = cached_path(filtered_vocab_file)
        vocab.set_from_file(filename=filtered_vocab_file, namespace=namespace, is_padded=False)
        namespace = params.pop('namespace', 'full')
        # if `full_vocab_file` is a URL, redirect to the cache
        full_vocab_file = cached_path(full_vocab_file)
        vocab.set_from_file(filename=full_vocab_file, namespace=namespace, oov_token=oov_token)
        return vocab
