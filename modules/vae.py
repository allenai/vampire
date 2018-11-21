import torch
from allennlp.common import Registrable
from allennlp.models.archival import Archive
from typing import Dict, Optional, List, Any


class VAE(Registrable, torch.nn.Module):
    '''
    General module for variational autoencoders.
    '''
    def get_latent_dim(self) -> int:
        '''
        get the dimension of the latent space
        '''
        return self.latent_dim

    def get_hidden_dim(self) -> int:
        '''
        get the hidden dimension of the model
        '''
        return self.hidden_dim

    def _initialize_weights_from_archive(self, archive: Archive) -> None:
        '''
        initialize weights of the VAE from a model archive
        '''
        raise NotImplementedError

    def _encode(self, tokens, n_sample):
        '''
        encode the tokens into a high dimensional embedding
        '''
        raise NotImplementedError

    def _discriminate(self, tokens: Dict, label: torch.IntTensor):
        '''
        Generate labels from the input, and use supervision to compute a loss.
        Used for semi-supervision.
        '''
        raise NotImplementedError

    def _decode(self, latent_code):
        '''
        decode the latent code into the vocabulary space
        '''
        raise NotImplementedError

    def _reconstruction_loss(self, **kwargs):
        '''
        Compute the reconstruction loss, comparing the output of the decoder with
        the input embedding
        '''
        raise NotImplementedError

    

    def forward(self, tokens, label) -> Dict: 
        raise NotImplementedError
