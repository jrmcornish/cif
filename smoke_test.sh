#!/bin/bash

set -o nounset
set -o errexit

FAILURES_DIR=smoke-test-failures
mkdir "$FAILURES_DIR"

## 2D

# CIF

for model in resflow maf maf-grid cond-affine-shallow-grid cond-affine-deep-grid dlgm-deep dlgm-shallow realnvp sos planar nsf-ar bnaf; do
  for dataset in 2uniforms 2lines 8gaussians checkerboard 2spirals rings 2marginals 1uniform annulus split-gaussian; do
    ./main.py --model $model --dataset $dataset --config max_epochs=1 || echo $model $dataset >> "$FAILURES_DIR"/2d-cif
  done
done

# Baseline

for model in resflow affine maf maf-grid realnvp sos planar nsf-ar bnaf; do
  for dataset in 2uniforms 2lines 8gaussians checkerboard 2spirals rings 2marginals 1uniform annulus split-gaussian; do
    ./main.py --model $model --dataset $dataset --config max_epochs=1 --baseline || echo $model $dataset >> "$FAILURES_DIR"/2d-baseline
  done
done

## Tabular

# CIF

for model in resflow cond-affine linear-cond-affine-like-resflow nonlinear-cond-affine-like-resflow maf realnvp nsf-ar; do
  for dataset in gas hepmass power miniboone bsds300; do
    ./main.py --model $model --dataset $dataset --config max_epochs=1 || echo $model $dataset >> "$FAILURES_DIR"/tabular-cif
  done
done

# Baseline

for model in resflow maf realnvp sos nsf-ar; do
  for dataset in gas hepmass power miniboone bsds300; do
    ./main.py --model $model --dataset $dataset --config max_epochs=1 --baseline || echo $model $dataset >> "$FAILURES_DIR"/tabular-baseline
  done
done

## Image

# CIF

for model in bernoulli-vae realnvp glow resflow-small resflow-large resflow-chen; do
  for dataset in mnist fashion-mnist cifar10 svhn; do
    ./main.py --model $model --dataset $dataset --config max_epochs=1 || echo $model $dataset >> "$FAILURES_DIR"/image-cif
  done
done

# Baseline

for model in realnvp glow resflow-small; do
  for dataset in mnist fashion-mnist cifar10 svhn; do
    ./main.py --model $model --dataset $dataset --config max_epochs=1 --baseline || echo $model $dataset >> "$FAILURES_DIR"/image-baseline
  done
done

## Gaussian

# CIF

for model in vae; do
  for dataset in linear-gaussian; do
    ./main.py --model $model --dataset $dataset --config max_epochs=1 || echo $model $dataset >> "$FAILURES_DIR"/gaussian-cif
  done
done
