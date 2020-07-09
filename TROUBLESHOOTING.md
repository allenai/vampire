## Troubleshooting

A few insights have been received after playing around with the model since publication, including some methods to circumvent training instability, especially when training on larger corpora.

Training instability usually manifests as NaN loss errors. To circumvent this, some easy things to try:

* Use TFIDF as input instead of raw word frequencies. You can do this by setting `--tfidf` flag in `scripts/prepreprocess_data.py`

* Increase batch size to at least 256

* Reduce LR to 1e-4 or 1e-5. If you are training over a very large corpus, shouldn’t affect representation quality much.

* Use some learning rate scheduler, slanted triangular scheduler has worked well for me. Make sure you tinker with the total number of epochs you train over.

* Clamp the KLD to some max value (e.g. 1000) so it doesn’t diverge

* Use a different KLD annealing scheduler (ie sigmoid)

If you still have issues after trying these modifications, please submit an issue!
