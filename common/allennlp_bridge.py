import torch
import numpy as np

from typing import Dict, List, Iterable

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary


@Vocabulary.register("vocabulary_with_pretrained_vae")
class VocabularyWithPretrainedVAE(Vocabulary):
    """
    Augment the allennlp Vocabulary with filtered vocabulary
    Idea: override from_params to "set" the vocab from a file before
    constructing in a normal fashion.
    """
    @classmethod
    def from_params(cls, params: Params, instances: Iterable['adi.Instance'] = None):
        sample_vocab_file = params.pop('supervised_vocab_file')
        vae_vocab_file = params.pop('vae_vocab_file', None)
        pad = params.pop('pad')
        vocab = super(VocabularyWithPretrainedVAE, cls).from_params(params, instances)

        #if `filtered_vocab_file` is a URL, redirect to the cache
        sample_vocab_file = cached_path(sample_vocab_file)
        vocab.set_from_file(filename=sample_vocab_file, namespace="full", is_padded=pad, oov_token="@@UNKNOWN@@")
        # if `full_vocab_file` is a URL, redirect to the cache
        if vae_vocab_file is not None:
            vae_vocab_file = cached_path(vae_vocab_file)
            vocab.set_from_file(filename=vae_vocab_file, namespace="vae", is_padded=False, oov_token="@@UNKNOWN@@")
        vocab.add_token_to_namespace(token="0", namespace="labels")
        vocab.add_token_to_namespace(token="1", namespace="labels")
        vocab.add_token_to_namespace(token="0", namespace="is_labeled")
        vocab.add_token_to_namespace(token="1", namespace="is_labeled")
        return vocab
