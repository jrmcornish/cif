# Changelog

Version numbers here correspond to git tags in the repository history.

## v1.1.0

Some additional features that we found useful for implementing the experiments for our [recent paper](http://proceedings.mlr.press/v130/shi21d.html) on debiasing gradient estimators for deep latent variable models.
Most significantly, this release adds the ability to train using different objectives, including [IWAE](https://arxiv.org/abs/1509.00519) and [reweighted wake-sleep](https://arxiv.org/abs/1406.2751), and different gradient estimators, including the [doubly reparameterized](https://arxiv.org/abs/1810.04152) and [sticking the landing](https://arxiv.org/abs/1703.09194) gradient estimators.
This release also contains a variety of small fixes and usability improvements.

## v1.0.0

Initial release of CIF version of paper (an earlier version released for the LGF version of the paper is available at https://github.com/jrmcornish/lgf)