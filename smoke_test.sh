#!/bin/bash

set -o nounset
set -o errexit

## 2D
for model in resflow affine maf maf-grid cond-affine-shallow-grid cond-affine-deep-grid dlgm-deep dlgm-shallow realnvp sos planar nsf-ar bnaf ffjord; do
  for dataset in 2uniforms 2lines 8gaussians checkerboard 2spirals rings 2marginals 1uniform annulus split-gaussian; do
    ./main.py --model $model --dataset $dataset --config max_epochs=1 || echo $model $dataset >> 2d-cif-fails
    ./main.py --model $model --dataset $dataset --config max_epochs=1 --baseline || echo $model $dataset >> 2d-baseline-fails
  done
done

## Tabular
for model in resflow cond-affine linear-cond-affine-like-resflow nonlinear-cond-affine-like-resflow resflow-no-g maf realnvp sos nsf-ar; do
  for dataset in gas hepmass power miniboone bsds300; do
    ./main.py --model $model --dataset $dataset --config max_epochs=1 || echo $model $dataset >> tabular-cif-fails
    ./main.py --model $model --dataset $dataset --config max_epochs=1 --baseline || echo $model $dataset >> tabular-baseline-fails
  done
done

## Image
for model in bernoulli-vae realnvp glow resflow-small resflow-big resflow-chen; do
  for dataset in mnist fashion-mnist cifar10 svhn; do
    ./main.py --model $model --dataset $dataset --config max_epochs=1 || echo $model $dataset >> image-cif-fails
    ./main.py --model $model --dataset $dataset --config max_epochs=1 --baseline || echo $model $dataset >> image-baseline-fails
  done
done

## Gaussian
for model in vae; do
  for dataset in linear-gaussian; do
    ./main.py --model $model --dataset $dataset --config max_epochs=1 || echo $model $dataset >> gaussian-cif-fails
    ./main.py --model $model --dataset $dataset --config max_epochs=1 --baseline || echo $model $dataset >> gaussian-baseline-fails
  done
done
