import logging
from typing import Dict, Union

import numpy as np
from scipy import sparse
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, Field, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.instance import Instance
from overrides import overrides
from vampire.common.util import load_sparse
from tqdm import tqdm


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

class SparseRowIndexer:
    def __init__(self, csr_matrix):
        data = []
        indices = []
        indptr = []

        # Iterating over the rows this way is significantly more efficient
        # than csr_matrix[row_index,:] and csr_matrix.getrow(row_index)
        for row_start, row_end in tqdm(zip(csr_matrix.indptr[:-1], csr_matrix.indptr[1:])):
             data.append(csr_matrix.data[row_start:row_end])
             indices.append(csr_matrix.indices[row_start:row_end])
             indptr.append(row_end-row_start) # nnz of the row

        self.data = np.array(data)
        self.indices = np.array(indices)
        self.indptr = np.array(indptr)
        self.n_columns = csr_matrix.shape[1]

    def __getitem__(self, row_selector):
        data = np.concatenate(self.data[row_selector])
        indices = np.concatenate(self.indices[row_selector])
        indptr = np.append(0, np.cumsum(self.indptr[row_selector]))

        shape = [indptr.shape[0]-1, self.n_columns]

        return sparse.csr_matrix((data, indices, indptr), shape=shape)
        
@DatasetReader.register("vampire_reader")
class VampireReader(DatasetReader):
    """
    Reads bag of word vectors from a sparse matrices representing training and validation data.

    Expects a sparse matrix of size N documents x vocab size, which can be created via
    the scripts/preprocess_data.py file.

    The output of ``read`` is a list of ``Instances`` with the field:
        vec: ``ArrayField``

    Parameters
    ----------
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be read lazily.
    sample : ``int``, optional, (default = ``None``)
        If specified, we will randomly sample the provided
        number of lines from the dataset. Useful for debugging.
    min_sequence_length : ``int`` (default = ``3``)
        Only consider examples from data that are greater than
        the supplied minimum sequence length.
    """
    def __init__(self,
                 lazy: bool = False,
                 sample: int = None,
                 array_conversion_batch_size: int = 10000,
                 min_sequence_length: int = 0) -> None:
        super().__init__(lazy=lazy)
        self._sample = sample
        self._array_conversion_batch_size = array_conversion_batch_size
        self._min_sequence_length = min_sequence_length

    @overrides
    def _read(self, file_path):
        logger.info("loading sparse matrix")
        # load sparse matrix
        mat = load_sparse(file_path)
        # optionally sample the matrix
        mat = mat.tocsr()
        if self._sample:
            indices = np.random.choice(range(mat.shape[0]), self._sample)
        else:
            indices = range(mat.shape[0])

        seq_lengths = mat[:, 1:].sum(1)
        logger.info("indexing rows")
        row_indexer = SparseRowIndexer(mat)
        target_indices = [index for index in indices if seq_lengths[index] > self._min_sequence_length]
        logger.info("converting to array")
        target_indices_batches = batch(target_indices, n=self._array_conversion_batch_size)
        for target_indices in tqdm(target_indices_batches, total=len(target_indices) // self._array_conversion_batch_size):
            rows = row_indexer[target_indices].toarray()
            for row in rows:
                instance = self.text_to_instance(vec=row)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(self, vec: Union[np.ndarray, str] = None) -> Instance:  # type: ignore
        """
        Parameters
        ----------
        text : ``str``, required.
            The text to classify
        label ``str``, optional, (default = None).
            The label for this text.

        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence or phrase.
            label : ``LabelField``
                The label label of the sentence or phrase.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        if isinstance(vec, np.ndarray):
            fields['tokens'] = ArrayField(vec)
        else:
            fields['tokens'] = TextField(vec, {'tokens': SingleIdTokenIndexer(namespace='vampire')})
        return Instance(fields)
