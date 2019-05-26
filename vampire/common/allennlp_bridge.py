import codecs
import json
import logging
import os
from typing import Iterable

from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params
from allennlp.common.util import namespace_match
from allennlp.data import instance as adi  # pylint: disable=unused-import
from allennlp.data.vocabulary import Vocabulary, pop_max_vocab_size
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_NON_PADDED_NAMESPACES = ("*tags", "*labels")
DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"
NAMESPACE_PADDING_FILE = 'non_padded_namespaces.txt'


@Vocabulary.register("extended_vocabulary")
class ExtendedVocabulary(Vocabulary):
    """
    Augment the allennlp Vocabulary with ability to dump background
    frequencies.
    """

    @classmethod
    def from_files(cls, directory: str) -> 'Vocabulary':
        """
        Loads a ``Vocabulary`` that was serialized using ``save_to_files``.
        Parameters
        ----------
        directory : ``str``
            The directory containing the serialized vocabulary.
        """
        logger.info("Loading token dictionary from %s.", directory)
        with codecs.open(os.path.join(directory, NAMESPACE_PADDING_FILE), 'r', 'utf-8') as namespace_file:
            non_padded_namespaces = [namespace_str.strip() for namespace_str in namespace_file]

        vocab = cls(non_padded_namespaces=non_padded_namespaces)

        # Check every file in the directory.
        files = os.listdir(directory)
        files = [file for file in files if "vae.bgfreq.json" not in file]
        for namespace_filename in files:
            if namespace_filename == NAMESPACE_PADDING_FILE:
                continue
            if namespace_filename.startswith("."):
                continue
            namespace = namespace_filename.replace('.txt', '')
            if any(namespace_match(pattern, namespace) for pattern in non_padded_namespaces):
                is_padded = False
            else:
                is_padded = True
            filename = os.path.join(directory, namespace_filename)
            vocab.set_from_file(filename, is_padded, namespace=namespace)

        return vocab

    @overrides
    def save_to_files(self, directory: str) -> None:
        """
        Persist this Vocabulary to files so it can be reloaded later.
        Each namespace corresponds to one file.
        Parameters
        ----------
        directory : ``str``
            The directory where we save the serialized vocabulary.
        """
        self.serialization_dir = directory  # pylint: disable=W0201
        os.makedirs(directory, exist_ok=True)
        if os.listdir(directory):
            logging.warning("vocabulary serialization directory %s is not empty", directory)

        with codecs.open(os.path.join(directory, NAMESPACE_PADDING_FILE), 'w', 'utf-8') as namespace_file:
            for namespace_str in self._non_padded_namespaces:
                print(namespace_str, file=namespace_file)

        for namespace, mapping in self._retained_counter.items():
            if namespace == 'vae':
                freqs = {k: c / self.get_vocab_size(namespace) for k, c in mapping.items()}
                output_filename = os.path.join(directory, namespace + '.bgfreq.json')
                with codecs.open(output_filename, 'w', encoding='utf-8') as output_file:
                    json.dump(freqs, output_file, indent=2, sort_keys=True)

        for namespace, mapping in self._index_to_token.items():
            # Each namespace gets written to its own file, in index order.
            with codecs.open(os.path.join(directory, namespace + '.txt'), 'w', 'utf-8') as token_file:
                num_tokens = len(mapping)
                start_index = 1 if mapping[0] == self._padding_token else 0
                for i in range(start_index, num_tokens):
                    print(mapping[i].replace('\n', '@@NEWLINE@@'), file=token_file)

    @classmethod
    def from_params(cls, params: Params, instances: Iterable['adi.Instance'] = None):
        max_vocab_size = pop_max_vocab_size(params)
        vocabulary_directory = params.pop("directory_path", None)
        if vocabulary_directory:
            vocab = cls.from_files(vocabulary_directory)
        else:
            vocab = cls()
            vocab = vocab.from_instances(instances=instances,
                                         max_vocab_size=max_vocab_size)
        return vocab


@Vocabulary.register("vocabulary_with_vae")
class VocabularyWithPretrainedVAE(Vocabulary):
    """
    Augment the allennlp Vocabulary with filtered vocabulary
    Idea: override from_params to "set" the vocab from a file before
    constructing in a normal fashion.
    """

    @classmethod
    def from_params(cls, params: Params, instances: Iterable['adi.Instance'] = None):
        vae_vocab_file = params.pop('vae_vocab_file')
        vocab = cls()
        vocab = vocab.from_instances(instances=instances,
                                     tokens_to_add={"classifier": ["@@UNKNOWN@@"]})
        vae_vocab_file = cached_path(vae_vocab_file)
        vocab.set_from_file(filename=vae_vocab_file,
                            namespace="vae",
                            oov_token="@@UNKNOWN@@")
        return vocab
