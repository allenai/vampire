{
    "vae_indexer": {
        "vae_tokens": {
            "type": "single_id",
            "namespace": "vae",
            "lowercase_tokens": true
        }
    },  
    "vae_embedder": {
        "vae_tokens": {
            "type": "vae_token_embedder",
            "representation": "encoder_output",
            "expand_dim": false,
            "model_archive": "/home/ubuntu/vae/model_logs/nvdm/model.tar.gz",
            "combine": false, 
            "dropout": 0.2
        }
    }
}

