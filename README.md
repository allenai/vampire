# VAMPIRE <img src="figures/bat.png" width="60"> 

VAriational Methods for Pretraining In Resource-limited Environments

Read paper [here](https://arxiv.org/abs/1906.02242).

## Citation

```
@inproceedings{vampire,
 author = {Suchin Gururangan and Tam Dang and Dallas Card and Noah A. Smith},
 title = {Variational Pretraining for Semi-supervised Text Classification},
 year = {2019},
 booktitle = {Proceedings of ACL},
}
```

## Quick links

* [Installation](#installation)
* [Basic Tutorial](TUTORIAL.md)
* [Scaling Up](SCALING.md)
* [Troubleshooting](TROUBLESHOOTING.md)

## Installation

Install necessary dependencies via `requirements.txt`:

```bash
pip install -r requirements.txt
```

Install the spacy english model with:

```bash
python -m spacy download en
```

Verify your installation by running: 

```bash
SEED=42 pytest -v --color=yes vampire
```

All tests should pass.


### Install from Docker

Alternatively, you can install the repository with Docker.

First, build the container: 

```bash
docker build -f Dockerfile --tag vampire/vampire:latest .
```

Then, run the container:

```bash
docker run -it vampire/vampire:latest
```

This will open a shell in a docker container that has all the dependencies installed.
