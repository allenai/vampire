import argparse
import numpy as np
from allennlp.models import load_archive
from modules import *
from models import *


def print_top_words(beta, feature_names, topic_names=None, n_top_words=8, sparsity_threshold=1e-5, values=False):
    """
    Display the highest and lowest weighted words in each topic, along with mean ave weight and sparisty
    """
    sparsity_vals = []
    maw_vals = []
    for i in range(len(beta)):
        # sort the beta weights
        order = list(np.argsort(beta[i]))
        order.reverse()
        output = ''
        # get the top words
        for j in range(n_top_words):
            if np.abs(beta[i][order[j]]) > sparsity_threshold:
                output += feature_names[order[j]] + ' '
                if values:
                    output += '(' + str(beta[i][order[j]]) + ') '

        order.reverse()
        output += ' / '
        # get the bottom words
        for j in range(n_top_words):
            if np.abs(beta[i][order[j]]) > sparsity_threshold:
                output += feature_names[order[j]] + ' '
                if values:
                    output += '(' + str(beta[i][order[j]]) + ') '

        # compute sparsity
        sparsity = float(np.sum(np.abs(beta[i]) < sparsity_threshold) / float(len(beta[i])))
        maw = np.mean(np.abs(beta[i]))
        sparsity_vals.append(sparsity)
        maw_vals.append(maw)
        output += ': MAW=%0.4f' % maw + '; sparsity=%0.4f' % sparsity

        # print the topic summary
        if topic_names is not None:
            output = topic_names[i] + ': ' + output
        else:
            output = str(i) + ': ' + output
        print(output)

    # return mean average weight and sparsity
    return np.mean(maw_vals), np.mean(sparsity_vals)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-path", dest="archive_path")
    args = parser.parse_args()
    archive_path = args.archive_path
    print("loading model at {}".format(archive_path))
    archive = load_archive(archive_path)
    print("Loaded.")
    decoder_weights = archive.model._vae._decoder._decoder_out.weight.detach().numpy()
    vocab = archive.model.vocab.get_index_to_token_vocabulary("full")
    background = archive.model._vae.bg.detach().numpy()
    order = list(np.argsort(background))
    print("Smallest background")
    for i in range(6):
        print(vocab[order[i]])
    print("Largest background")
    order.reverse()
    for i in range(6):
        print(vocab[order[i]])
    print_top_words(decoder_weights.T, vocab)
