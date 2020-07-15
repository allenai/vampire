from vampire.data import VampireReader
from allennlp.common.file_utils import cached_path 
from vampire.predictors import VampirePredictor
from vampire.models import VAMPIRE
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.data import DataLoader


class VampireManager:

    def __init__(self, **kwargs):
        self.model = VAMPIRE.from_params(Params(kwargs['model']))
        self.vocabulary = Vocabulary.from_params(Params(kwargs['vocabulary']))
        self.reader = VampireReader.from_params(Params(kwargs['dataset_reader']))
        self.predictor = VampirePredictor(self.model, dataset_reader=self.reader)
        
    def read_data(self, train_path: str, dev_path: str, **kwargs):
        train_dataset = self.reader.read(cached_path(train_path))
        validation_dataset = self.reader.read(cached_path(dev_path))
        return train_dataset, validation_dataset

    def fit(self, train_path: str, dev_path: str, device: int, **kwargs):
        self.trainer = Trainer.from_params(model=self.model,
                               params=Params(kwargs['trainer']))
        train_dataset, validation_dataset = self.read_data(train_path, dev_path)
        vocab = self.vocabulary.from_instances(train_dataset + validation_dataset)
        self.data_loader = DataLoader.from_params(Params(kwargs['data_loader']))
        self.data_loader.index_with(vocab)
        self.trainer.train()
    
    def predict(self, input_):
        self.predictor.predict_json(input_)
    