
import numpy as np
import argparse

def step():
    hidden_dim = np.random.uniform(64, 1024)
    vocab_size = np.random.uniform(5000, 60000)
    latent_dim = np.random.uniform(10, 1000)
    apply_batchnorm = np.random.choice([True, False])
    update_bg_freq = np.random.choice([True, False])
    kl_weight_annealing = np.random.choice(['linear', 'constant', 'sigmoid'])
    encoder_layers = np.random.choice([1,2,3])
    encoder_activations = np.random.choice(['softplus', 'tanh', 'relu'])
    param_projection_layers = np.random.choice([1, 2, 3])
    decoder_layers = np.random.choice([1, 2, 3])
    z_dropout = np.random.uniform(0, 1)

    return {
        "vocabulary.max_vocab_size.vae": vocab_size,
        "model.encoder.input_dim": vocab_size + 2,
        "model.encoder.hidden_dims": [hidden_dim],
        "model.mean_projection.input_dim": hidden_dim,
        "model.mean_projection.hidden_dims": [latent_dim],
        "model.log_variance_projection.input_dim": hidden_dim,
        "model.log_variance_projection.hidden_dims": [latent_dim],
        "model.decoder.input_dim": latent_dim,
        "model.decoder.hidden_dims": [vocab_size + 2],
        "model.apply_batchnorm": apply_batchnorm,
        "model.z_dropout": z_dropout,
        "model.encoder.activations": encoder_activations,
        "model.decoder.activations": decoder_activations,
        "model.update_background_freq": update_bg_freq,
        "model.kl_weight_annealing": kl_weight_annealing,
        "model.encoder.num_layers": encoder_layers,
        "model.decoder.num_layers": decoder_layers,
        "model.log_variance_projection.num_layers": param_projection_layers,
        "model.mean_projection.num_layers": param_projection_layers,
    }

def generate_json(num_samples):
    res = []
    for _ in range(num_samples):
        res.append(step())
    return res