import argparse
from allennlp.models.archival import load_archive, Archive
import torch
from torch.autograd import Variable
import os
from models import *
from modules import *
from tqdm import trange
from common.util import sample 
from sklearn import manifold
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

parser.add_argument('--pretrained-file', type=str,
                    help='pretrained vae')

args = parser.parse_args()

latent_size = 16
pretrained_file = args.pretrained_file
if os.path.isfile(pretrained_file):
    archive = load_archive(pretrained_file)
else:
    raise ValueError("{} doesn't exist".format(pretrained_file))

archive.model.eval()

vae = archive.model._vae
vae.weight_scheduler = lambda x: schedule(x, "constant")
if vae.word_dropout < 1.0:
    vae.word_dropout=0.0
vae.kl_weight_annealing="constant"
decoder = vae._decoder
model_type = archive.config['model'].pop('type')

token2idx = archive.model.vocab.get_token_to_index_vocabulary("full")
idx2token = archive.model.vocab.get_index_to_token_vocabulary("full")

import pandas as pd

df = pd.read_json("/home/ubuntu/data/ptb/dev.jsonl", lines=True)
max_len = 50
s1 = list(df.sample(frac=1).tokens)[:10]

seqs = []
for item in s1:
    s1_item = "@@start@@ {} @@end@@".format(item)
    idxs = []
    idxs = np.array([token2idx["@@PADDING@@"]] * 50)
    sub_idxs = []
    for word in s1_item.split()[:50]:
        sub_idxs.append(token2idx.get(word, token2idx["@@UNKNOWN@@"]))
    idxs[:len(s1_item.split())] = sub_idxs
    seqs.append(torch.from_numpy(idxs).unsqueeze(0))
input_sequence = torch.cat(seqs, 0)
z = vae(tokens={"tokens": input_sequence})['theta'].detach().numpy()

def computeTSNEProjectionOfLatentSpace(z, display=True):
    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(z)

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots()
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
        for i, txt in enumerate(s1):
            ax.annotate(txt, (X_tsne[i, 0], X_tsne[i, 1]), size=5)

        plt.savefig("test.png")
    else:
        return X_tsne

computeTSNEProjectionOfLatentSpace(z)
