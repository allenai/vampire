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
        vae_vocab_file = params.pop('vae_vocab_file')
        non_padded_namespaces = params.pop('non_padded_namespaces')
        vocab = cls()
        #if `filtered_vocab_file` is a URL, redirect to the cache
        vocab = vocab.from_instances(instances=instances,
                                    non_padded_namespaces=non_padded_namespaces,
                                    tokens_to_add={"tokens": ["@@UNKNOWN@@"]})
        # if `full_vocab_file` is a URL, redirect to the cache
        vae_vocab_file = cached_path(vae_vocab_file)
        vocab.set_from_file(filename=vae_vocab_file,
                            namespace="vae",
                            is_padded=not "vae" in non_padded_namespaces,
                            oov_token="@@UNKNOWN@@")
        return vocab
