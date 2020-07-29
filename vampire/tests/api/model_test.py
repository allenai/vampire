from vampire.api import VampireModel
import pytest
from vampire.common.testing import VAETestCase
import numpy as np
from vampire.common.util import load_sparse


class TestModelAPI(VAETestCase):

    def test_from_pretrained(self):
        archive_path = self.FIXTURES_ROOT / "vae" / "model.tar.gz"
        model = VampireModel.from_pretrained(archive_path, cuda_device=-1, for_prediction=False)
        assert model.model.training
    
    def test_from_pretrained_for_prediction(self):
        archive_path = self.FIXTURES_ROOT / "vae" / "model.tar.gz"
        model = VampireModel.from_pretrained(archive_path, cuda_device=-1, for_prediction=True)
        assert not model.model.training

    def test_from_params(self):
        data_dir = self.FIXTURES_ROOT / "ag"
        model = VampireModel.from_params(data_dir)
        assert model.model.training

    def test_read_data_works(self):
        data_dir = self.FIXTURES_ROOT / "ag"
        train_path = data_dir / "train.npz"
        dev_path = data_dir / "dev.npz"
        model = VampireModel.from_params(data_dir)
        train_dataset, dev_dataset = model.read_data(train_path, dev_path, lazy=False)
        assert train_dataset

    def test_fit_works(self):
        data_dir = self.FIXTURES_ROOT / "ag"
        serialization_dir = self.TEST_DIR / "vampire"
        model = VampireModel.from_params(data_dir)
        model.fit(data_dir, serialization_dir)
        assert (self.TEST_DIR / "vampire" / "model.tar.gz").exists()
    
    def test_extract_features(self):
        archive_path = self.FIXTURES_ROOT / "vae" / "model.tar.gz"
        model = VampireModel.from_pretrained(archive_path, cuda_device=-1, for_prediction=True)
        output = model.extract_features({"text": "Hello World"}, False, True)
        np.testing.assert_almost_equal(np.array(output), np.array([[2.095332145690918, 2.0823867321014404, -1.944653034210205, 
                                                                    4.842313289642334, 1.8353369235992432, 1.5818876028060913, 
                                                                    2.465179443359375, -0.417280912399292, 2.734004259109497, 
                                                                    -4.794219970703125]]))
    
    def test_extract_features_array(self):
        dev_file = self.FIXTURES_ROOT / "ag" / "dev.npz"
        archive_path = self.FIXTURES_ROOT / "vae" / "model.tar.gz"
        model = VampireModel.from_pretrained(archive_path, cuda_device=-1, for_prediction=True)
        z = load_sparse(dev_file)
        outputs = []
        for row in z:
            output = model.extract_features({"text": "Hello World"}, False, True)
            outputs.append(output)
        np.testing.assert_almost_equal(np.array(outputs), np.array([[[2.095332145690918, 2.0823867321014404, -1.944653034210205, 4.842313289642334, 
                                                                    1.8353369235992432, 1.5818876028060913, 2.465179443359375, -0.417280912399292, 
                                                                    2.734004259109497, -4.794219970703125]], [[2.095332145690918, 2.0823867321014404,
                                                                    -1.944653034210205, 4.842313289642334, 1.8353369235992432, 1.5818876028060913,
                                                                    2.465179443359375, -0.417280912399292, 2.734004259109497, -4.794219970703125]], 
                                                                    [[2.095332145690918, 2.0823867321014404, -1.944653034210205, 4.842313289642334,
                                                                    1.8353369235992432, 1.5818876028060913, 2.465179443359375, -0.417280912399292,
                                                                    2.734004259109497, -4.794219970703125]]]))
