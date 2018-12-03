import torch
from allennlp.common import Registrable
from allennlp.models.archival import Archive
from typing import Dict, Optional, List, Any
from modules.encoder import Seq2SeqEncoder, BowEncoder
from allennlp.data import Vocabulary


    
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

    def _initialize_weights_from_archive(self,
                                         archive: Archive,
                                         freeze_weights: bool = False) -> None:
        """
        Initialize weights (theta?) from a model archive.

        Params
        ______
        archive : `Archive`
            pretrained model archive
        """
        model_parameters = dict(self.named_parameters())
        archived_parameters = dict(archive.model.named_parameters())
        for item, val in archived_parameters.items():
            new_weights = val.data
            item_sub = ".".join(item.split('.')[1:])
            model_parameters[item_sub].data.copy_(new_weights)
            if freeze_weights:
                item_sub = ".".join(item.split('.')[1:])
                model_parameters[item_sub].requires_grad = False
    
    def _freeze_weights(self) -> None:
        """
        Freeze the weights of the model
        """

        model_parameters = dict(self.named_parameters())
        for item in model_parameters:
            model_parameters[item].requires_grad = False        

    def drop_words(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # randomly tokens with <unk>
        tokens = tokens['tokens']
        prob = torch.rand(tokens.size()).to(tokens.device)
        prob[(tokens.data - self.sos_idx) * (tokens.data - self.pad_idx) == 0] = 1
        tokens_with_unks = tokens.clone()
        tokens_with_unks[prob < self.word_dropout] = self.unk_idx
        return {"tokens": tokens_with_unks}

    def forward(self, tokens, label, metadata=None) -> Dict: 
        raise NotImplementedError
