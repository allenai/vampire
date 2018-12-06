from allennlp.models.archival import load_archive, Archive
from argparse import ArgumentParser
import os
from models import *
from modules import *
import numpy as np
import torch
from allennlp.nn.util import get_text_field_mask
from common.util import interpolate, sample


parser = ArgumentParser()

parser.add_argument("--pretrained-file", '-p', dest='pretrained_file', required=True)
parser.add_argument("--start", '-s', dest='start', required=False)
parser.add_argument("--label", '-l', dest='label', required=False)
parser.add_argument("--max-len", '-m', dest='max_len', default=10)

args = parser.parse_args()

pretrained_file = args.pretrained_file
if os.path.isfile(pretrained_file):
    archive = load_archive(pretrained_file)
else:
    raise ValueError("{} doesn't exist".format(pretrained_file))
vae = archive.model._vae
model_type = archive.config['model'].pop('type')

token2idx = archive.model.vocab.get_token_to_index_vocabulary("full")
idx2token = archive.model.vocab.get_index_to_token_vocabulary("full")

if args.start is not None:
    inp_sentence = args.start.split(" ")
    inp_sentence = [token2idx[x] for x in inp_sentence]
    sentence = [token2idx["@@start@@"]] + inp_sentence
else:
    sentence = [token2idx["@@start@@"]]

if model_type == 'semisupervised_vae':
    if args.label is not None:
        label_init = np.int(args.label)
    else:
        label_init = np.random.choice(range(0, vae._classifier._num_labels), 1)

def save_sample(save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to

latent_size = 16
batch_size = 10
sos_idx = token2idx["@@start@@"]
pad_idx = token2idx["@@PADDING@@"]
eos_idx = token2idx["@@end@@"]

sequence_idx = torch.arange(0, batch_size).long() # all idx of batch
sequence_running = torch.arange(0, batch_size).long() # all idx of batch which are still generating
sequence_mask = torch.ones(batch_size).byte()

running_seqs = torch.arange(0, batch_size).long() # idx of still generating sequences with respect to current loop

generations = torch.zeros(batch_size, args.max_len).fill_(pad_idx).long()

input_sequence_1 = torch.from_numpy(np.array([token2idx[x] for x in "@@start@@ president united states @@end@@".split()])).unsqueeze(0)
input_sequence_2 = torch.from_numpy(np.array([token2idx[x] for x in "@@start@@ president united states @@end@@".split()])).unsqueeze(0)
z1 = torch.randn([latent_size]).numpy()
z2 = torch.randn([latent_size]).numpy()
z = interpolate(start=z1, end=z2, steps=batch_size-2)
theta = torch.from_numpy(z).float()
t=0 
while(t<args.max_len and len(running_seqs)>0):

    if t == 0:
        input_sequence = torch.Tensor(batch_size).fill_(sos_idx).long()
    
    input_sequence = input_sequence.unsqueeze(1)
    embedded_text = vae._embedder({"tokens": input_sequence})
    mask = get_text_field_mask({"tokens": input_sequence})
    output = vae._decoder(embedded_text=embedded_text, theta=theta[running_seqs, :].unsqueeze(0), mask=mask)
    next_token_logits = output['decoder_output']
    input_sequence = sample(next_token_logits.squeeze(1))
    # save next input
    generations = save_sample(generations, input_sequence, sequence_running, t)

    # update gloabl running sequence
    sequence_mask[sequence_running] = (input_sequence != eos_idx).data
    sequence_running = sequence_idx.masked_select(sequence_mask)

    # update local running sequences
    running_mask = (input_sequence != eos_idx).data
    running_seqs = running_seqs.masked_select(running_mask)

    # prune input and hidden state according to local update
    if len(running_seqs) > 0:
        input_sequence = input_sequence[running_seqs]
        running_seqs = torch.arange(0, len(running_seqs)).long()
    t += 1

generated_tokens = []
for row in generations:
    generated_tokens.append([idx2token[x.item()] for x in row.data if idx2token[x.item()] != "@@PADDING@@"])

for row in generated_tokens:
    print(" ".join(row))