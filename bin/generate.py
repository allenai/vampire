import argparse
from allennlp.models.archival import load_archive, Archive
import torch
from torch.autograd import Variable
import os
from models import *
from modules import *
from tqdm import trange
from common.util import sample 


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

parser.add_argument('--words', type=int, default=10,
                    help='number of words to generate')
parser.add_argument('--temperature', type=float, default=10.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--pretrained-file', type=str,
                    help='pretrained vae')
parser.add_argument('--use-theta',
                    action='store_true')
args = parser.parse_args()

latent_size = 16
pretrained_file = args.pretrained_file
if os.path.isfile(pretrained_file):
    archive = load_archive(pretrained_file)
else:
    raise ValueError("{} doesn't exist".format(pretrained_file))

archive.model.eval()

vae = archive.model
vae.weight_scheduler = lambda x: schedule(x, "constant")
vae.word_dropout=0.0
vae.kl_weight_annealing="constant"
decoder = vae._decoder
model_type = archive.config['model'].pop('type')

token2idx = archive.model.vocab.get_token_to_index_vocabulary("full")
idx2token = archive.model.vocab.get_index_to_token_vocabulary("full")


theta = torch.randn([latent_size])
input = torch.randint(len(idx2token), (1, 1), dtype=torch.long)
input.fill_(token2idx["@@start@@"])
generated_words = "@@start@@ "
success = 0
i = 0
with torch.no_grad():  # no tracking history
    while success < args.words:
        output = vae(tokens={'tokens': input})
        word_weights = torch.nn.functional.softmax(output['decoder_output']['decoder_output'].squeeze().div(args.temperature), dim=-1)
        word_idx = sample(word_weights)
        if word_idx == 0:
            continue
        else:
            success += 1
        input.fill_(word_idx)
        word = idx2token[word_idx.item()]
        generated_words += word + ('\n' if i % 20 == 19 else ' ')
        i += 1
    print(generated_words)