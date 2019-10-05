# README

This repository contains the Pytorch code we used to produce the results described in our paper [Localised Generative Flows](https://arxiv.org/abs/1909.13833).

## Setup

1. Make sure you have `pipenv` installed. Run e.g. `pip install pipenv` if not.
2. From the same directory as `Pipfile`, run `pipenv install`

## Running

To train our model on a simple 2D dataset, run:

    pipenv run ./main.py --dataset 2uniforms

By default, this will create a directory `runs/`, which will contain Tensorboard logs giving various information about the training run, including 2D density plots in this case. To inspect this, ensure you have `tensorboard` installed (e.g. `pip install tensorboard`), and run in a new terminal:

    tensorboard --logdir runs/ --port=8008

Keep this running, and navigate to http://localhost:8008, where the results should be visible.

Other datasets can also be launched using the same command as above. Run

    pipenv run ./main.py --help

to see the full options.

Each dataset has a default configuration set up for it that is described in the paper. However, to try out alternative configurations, simply modify the relevant options in `config.py`.

For comparison purposes, we also provide comparable baseline models (i.e. not LGFs) for each configuration. To run these, simply add the `--baseline` option to `main.py`.

## Bibtex

    @misc{cornish2019localised,
        title={Localised Generative Flows},
        author={Rob Cornish and Anthony L. Caterini and George Deligiannidis and Arnaud Doucet},
        year={2019},
        eprint={1909.13833},
        archivePrefix={arXiv},
        primaryClass={stat.ML}
    }
