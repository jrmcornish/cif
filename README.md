# README

Code release for [Localised Generative Flows](https://arxiv.org/abs/1909.13833) (LGFs).

## Usage

### Setup

1. Make sure you have `pipenv` installed. Run e.g. `pip install pipenv` if not.
2. From the same directory as `Pipfile`, run `pipenv install`

### Training

To train our model on a simple 2D dataset, run:

    pipenv run ./main.py --dataset 2uniforms

By default, this will create a directory `runs/`, which will contain Tensorboard logs giving various information about the training run, including 2-D density plots in this case. To inspect this, ensure you have `tensorboard` installed (e.g. `pip install tensorboard`), and run in a new terminal:

    tensorboard --logdir runs/ --port=8008

Keep this running, and navigate to http://localhost:8008, where the results should be visible.

Other datasets can also be launched using the same command as above. Run

    pipenv run ./main.py --help

to see the full options.

Each dataset has a default configuration set up for it that is described in the paper. However, to try out alternative configurations, simply modify the relevant options in `config.py`.

For comparison purposes, we also provide comparable baseline models (i.e. not LGFs) for each configuration. To run these, simply add the `--baseline` option when running `main.py`.

## Example results

When trained on the `2uniforms` dataset, the baseline MAF will produce something like the following:

![MAF](imgs/2d/2uniforms_maf_300_epochs.png)

Notice that the MAF is unable to transform the support of the prior distribution (which is Gaussian) into the support of the target, which has two disconnected components and hence a different topology.

In contrast, LGF-MAF can properly separate the two components:

![LGF-MAF](imgs/2d/2uniforms_lgf-maf_300_epochs.png)

We also obtain improved performance when using LGFs on various UCI and image datasets. Although harder to visualise, we conjecture that an analogous story holds here as for the simple 2-D case.

### Generated image samples

#### RealNVP

FashionMNIST

![RealNVP FashionMNIST](imgs/images/realnvp_fashion-mnist_samples.png)

CIFAR10

![RealNVP FashionMNIST](imgs/images/realnvp_cifar10_samples.png)

#### LGF-RealNVP

FashionMNIST

![RealNVP FashionMNIST](imgs/images/lgf-realnvp_fashion-mnist_samples.png)

CIFAR10

![RealNVP FashionMNIST](imgs/images/lgf-realnvp_cifar10_samples.png)

## Bibtex

    @misc{cornish2019localised,
        title={Localised Generative Flows},
        author={Rob Cornish and Anthony L. Caterini and George Deligiannidis and Arnaud Doucet},
        year={2019},
        eprint={1909.13833},
        archivePrefix={arXiv},
        primaryClass={stat.ML}
    }
