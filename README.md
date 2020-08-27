# README

This is the code we used to produce the experiments in our paper [Relaxing Bijectivity Constraints with Continuously Indexed Normalising Flows](https://arxiv.org/abs/1909.13833) (ICML 2020). It is a fork of our original codebase, which was previously maintained at https://github.com/jrmcornish/lgf.

This code may be useful to anyone interested in CIFs, as well as normalising flows more generally. In particular, this code allows specifying a large variety of different common architectures -- including various primitive flow steps and multi-scale configurations -- via an intuitive intermediate representation, which is then converted into an actual flow object. Please see the "Specifying models" section below for more details.

## Setup

First, install submodules:

    $ git submodule init
    $ git submodule update

Next, install dependencies. If you use `conda`, the following will create an environment called `cif`:

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

    CUDA_VISIBLE_DEVICES=0 ./main.py --model resflow --dataset 2uniforms

It may be necessary to change `CUDA_VISIBLE_DEVICES=0` depending on the availability of device 0.

> **Note:** It is recommended to specify `CUDA_VISIBLE_DEVICES` explicitly for all runs.
> This is because, for generality, we always wrap our modules in `nn.DataParallel`, whether or not we are using multiple GPUs.
> (This in turn helps with keeping `state_dict`s uniform and easy to save/load.)
> As such, `./main.py` will default to running on all available CUDA devices, even for small datasets for which this is inefficient.

By default, this will create a directory `runs/`, which will contain Tensorboard logs giving various information about the training run, including 2-D density plots in this case. To inspect this, ensure you have `tensorboard` installed (e.g. `pip install tensorboard`), and run in a new terminal:

    tensorboard --logdir runs/ --port=8008

Keep this running, and navigate to http://localhost:8008, where the results should be visible.
For 2D datasets, the "Images" tab shows the learned density, and for image datasets, the "Images" tab shows samples from the model over training.
The "Text" tab shows the config used to produce each run, which is also stored (alongside various other information like version control info) in the directory inside `runs/` that was created.

Each dataset has a default configuration set up for it that is described in the paper and specified in the appropriate file `two_d.py`, `tabular.py`, or `images.py` in `config/`. To try out alternative configurations, either modify these files directly, or use the `--config` argument to `main.py` to override the value specified by the config file. E.g.

    ./main.py --model resflow --dataset mnist --config 'scales=[3, 3, 3]'

will override the default config value for `scales` set in `config/images.py` (which contains the `resflow` config for `mnist`).

For comparison purposes, for each model we also provide a standard baseline flow with roughly the same number of parameters. To run these, simply add the `--baseline` option when running `main.py`.

To inspect the model (either CIF or baseline) used for a given dataset, add the `--print-schema` argument to show a high-level schema of the model that is used, and the `--print-model` argument to see the actual PyTorch object created. To see the number of parameters used by the model, add `--print-num-params`.

## Specifying models

### Configs

Config files are contained within config/ in the root directory.
They are specified as Python code - see e.g. config/gaussian.py for a working example using ML-LL-SS for a linear Gaussian target.
There's a small DSL powering this (it's self-contained within config/) - basically, each config is named (e.g. `resflow`) and is attached to a group (e.g. `2d`) which can handle a set of datasets (e.g. `2uniforms`).
Each group requires a `@base` config, which is shared globally across the group.
Each config then can override specific settings in the base config as desired.
(It is then possible to add manual overrides via the `--config` option as described above.)

### Schemas

Configs get translated into a schema (see `schemas.py`) before being eventually converted into a Pytorch module in factory.py.
Schemas are essentially just JSON objects that serve as an intermediate representation for downstream processing.
You can see the schema corresponding to a given config by using the `--print-schema` option to main.py.
(Incidentally you can also view the config by the `--print-config` flag).

### Coupler configs

Throughout this codebase, "Couplers" refer to modules that output a "shift" and "log-scale" quantity.
This pattern appears in several places for CIFs, as well as normalising flows/VAEs more generally.

The translation process from configs to schemas includes several features to allow specifying couplers using a convenient shorthand.
For example, all three of the following configurations define the neural networks to be used inside the $q_{U_\ell|Z_\ell}$ densities used at each layer of the model:

    # e.g.
    "q_nets": [10] * 2,

    # or e.g.
    "q_nets": "fixed-constant",

    # or e.g.
    "q_mu_nets": "learned-constant",
    "q_sigma_nets": "fixed-constant"

These inference networks, which are all mean-field Gaussian in our implementation for now, require both a mean and standard deviation output.
Here:

- The first usage produces a $q_{U_\ell|Z_\ell}$ whose mean and log-standard deviation are produced as the joint output of a network with 2 hidden layers of 10 hidden units each. (It will be either an MLP or a ResFlow depending on whether the schema involves a flattening layer.)
- The second usage replaces this network with a fixed constant of zero (so effectively $q_{U_\ell|Z_\ell}$ is a standard Gaussian independent of $Z_\ell$).
- The third usage separately produces the mean as a learned constant (which is independent of $Z_\ell$) and the log-standard deviation as a fixed constant of zero.

These settings can be useful for recovering other familiar models. For example, to obtain a traditional VAE, simply set `p_nets = "fixed-constant"`. (See Section 4.3 in the paper for more details.)

For full details of this, please see `get_coupler_config()` inside `schemas.py`.
Additionally, note that once a config has been translated into a schema, the schema provides all the information required to construct the Pytorch module that is subsequently trained (which happens inside `factory.py`).
As such, the schema provides a single source of truth for the model, and so `--print-schema` can be used for understanding the effect of various config changes like these.

## Bibtex

    @misc{cornish2019relaxing,
        title={Relaxing Bijectivity Constraints with Continuously Indexed Normalising Flows},
        author={Rob Cornish and Anthony L. Caterini and George Deligiannidis and Arnaud Doucet},
        year={2019},
        eprint={1909.13833},
        archivePrefix={arXiv},
        primaryClass={stat.ML}
    }
