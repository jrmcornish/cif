# README

ICML code release for _Relaxing Bijectivity Constraints with Continuously Indexed Normalising Flows_ paper.

## Setup

First, install dependencies. If you use `conda`, the following will create an environment called `cif`:

    conda env create -f environment.yml

Activate this with

    conda activate cif

before running any code or tests.

If you don't use `conda`, then please see `environment.yml` for a list of required packages, which will need to be installed manually e.g. via `pip`.

### Obtaining datasets

Our code runs on several types of datasets, including synthetic 2-D data, tabular data, and image data. For a full list run

    ./main.py --help

The 2-D datasets are automatically generated, and the image datasets are downloaded automatically. However the tabular datasets will need to be manually downloaded from [this location](https://zenodo.org/record/1161203). The following should do the trick:

    mkdir -p data/ && wget -O - https://zenodo.org/record/1161203/files/data.tar.gz | tar --strip-components=1 -C data/ -xvzf - data/{gas,hepmass,miniboone,power,BSDS300}

This will download the data to `data/`. If `data/` is in the same directory as `main.py`, then everything will work out-of-the-box by default. If not, the location to the data directory will need to be specified as an argument to `main.py` (see `--help`).

## Usage

To train our model on a simple 2D dataset, run:

    ./main.py --model resflow --dataset 2uniforms

By default, this will create a directory `runs/`, which will contain Tensorboard logs giving various information about the training run, including 2-D density plots in this case. To inspect this, ensure you have `tensorboard` installed (e.g. `pip install tensorboard`), and run in a new terminal:

    tensorboard --logdir runs/ --port=8008

Keep this running, and navigate to http://localhost:8008, where the results should be visible.

Each dataset has a default configuration set up for it that is described in the paper and specified in the appropriate file `two_d.py`, `tabular.py`, or `images.py` in `config/`. To try out alternative configurations, either modify these files directly, or use the `--config` argument to `main.py` to override the value specified by the config file. E.g.

    ./main.py --model resflow --dataset mnist --config 'scales=[3, 3, 3]'

will override the default config value for `scales` set in `config/images.py` (which contains the `resflow` config for `mnist`).

For comparison purposes, for each model we also provide a standard baseline flow with roughly the same number of parameters. To run these, simply add the `--baseline` option when running `main.py`.

To inspect the model (either CIF or baseline) used for a given dataset, add the `--print-schema` argument to show a high-level schema of the model that is used, and the `--print-model` argument to see the actual PyTorch object created. To see the number of parameters used by the model, add `--print-num-params`.
