from typing import Dict
import numpy as np
import torch
from allennlp.models.model import Model
from modules.vae import VAE
from allennlp.data import Vocabulary
from allennlp.nn import InitializerApplicator
from overrides import overrides
from allennlp.training.metrics import CategoricalAccuracy, Average
from allennlp.modules import FeedForward
from common.perplexity import Perplexity
from allennlp.nn.util import get_text_field_mask
from modules.vae import VAE
from argparse import ArgumentParser


parser = ArgumentParser()

parser.add_argument("--pretrained-file", '-p', dest='pretrained_file')
parser.add_argument("--sentence-start", '-s', dest='sentence_start', required=False)
parser.add_argument("--max-len", '-m', dest='max_len', default=100)

args = parser.parse_args()

pretrained_file = args.pretrained_file
if os.path.isfile(pretrained_file):
    archive = load_archive(pretrained_file)

theta = archive.model._vae.theta
vae = archive.model._vae

token2idx = archive.model.vocab.get_token_to_index_vocabulary("full")
if args.sentence_start is not None:
    inp_sentence = args.sentence_start.split(" ")
    inp_sentence = [token2idx[x] for x in inp_sentence]
    sentence = [token2idx["<BOS>"]] + inp_sentence
else:
    sentence = [token2idx["<BOS>"]]

for _ in range(args.max_len):
    output = vae(sentence)
    next_token_logits = output['flattened_decoder_output']
    next_token_probs = torch.nn.softmax(next_token_logits)
    next_token = vocab[next_token_probs.max(1)[1]]
    sentence.append(next_token)

print(sentence)