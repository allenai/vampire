
# Troubleshooting

This document gathers practical insights from training VAMPIRE on variety of corpora, which hopefully will aid users of VAMPIRE when encountering issues. 

## Make sure topics are good!

The best indicator of successful pretraining and useful VAMPIRE embeddings is how good the VAMPIRE internal topics are qualitatively, and how high its training NPMI is. If your VAMPIRE embeddings are not useful downstream, make sure you see qualitatively reasonable topic assignments in the latent space, by inspecting the topic files in the serialization directory. 

If the topics are bad, either something weird is going on during pretraining, or something weird is going on with your data. If the data is reasonable, and you expect to see distinctive topics surface from it, try playing around with the VAMPIRE hyperparameters in `environments/environments.py` and with vocabulary sizes during tokenization. Anecdotally, learning rate, hidden dimension size, and vocabulary size are probably the most important hyperparameters that can affect pretraining quality.

## Recovering from NaN loss errors

Pretraining instability usually manifests as NaN loss errors. This can sometimes happen because the KL divergence randomly diverges, or because the learning rate is too high. There are a number of things you can try to stabilize training:

* Use TFIDF vectorizer instead of Count Vectorizer as input
* Increase batch size to at least 256.
* Reduce LR to 1e-4 or 1e-5. If you are training over a very large corpus, shouldn’t affect representation quality much.
* Use some learning rate scheduler, slanted triangular scheduler has worked well for me. Make sure you tinker with the total number of epochs you train over.
* Clamp the KLD to some max value (e.g. 1000) so it doesn’t diverge.
* Use a different KLD annealing scheduler (ie sigmoid).

If these don't work, please open an issue with a description of your data, there might be a bug somewhere.