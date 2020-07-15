from vampire.data import VampireReader
from allennlp.common.file_utils import cached_path 
from vampire.predictors import VampirePredictor
from vampire.models import VAMPIRE
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.trainer import Trainer
from allennlp.data import DataLoader


class VampireManager:

    def __init__(self, params):
        self.reader = VampireReader.from_params(Params(params['dataset_reader']))
        self.vocabulary = Vocabulary.from_params(Params(params['vocabulary']))
        self.model = VAMPIRE.from_params(Params(params['model']))
        self.trainer = Trainer.from_params(model=self.model,
                               params=Params(params['trainer']))
        self.predictor = VampirePredictor(self.model, dataset_reader=self.reader)
        self.data_loader = DataLoader.from_params(Params(params['data_loader']))
        
    def read_data(self, train_path, dev_path):
        train_dataset = self.reader.read(cached_path(train_path))
        validation_dataset = self.reader.read(cached_path(dev_path))
        return train_dataset, validation_dataset

    def fit(self, train_path: str, dev_path: str, device: int, **kwargs):
        train_dataset, validation_dataset = self.read_data(train_path, dev_path)
        vocab = self.vocabulary.from_instances(train_dataset + validation_dataset)
        self.data_loader.index_with(vocab)
        self.trainer.train()
    
    def predict(self, input_):
        self.predictor.predict_json(input_)
    