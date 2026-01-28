# Neuron Counterfactual Validation

This repository contains a proof-of-concept (PoC) implementation
for validating internal neurons of a deep neural network using
counterfactual explanations.

The PoC uses:
- GTSRB (German Traffic Sign Recognition Benchmark)
- ResNet-50 (ImageNet pre-trained)
- Neuron-level counterfactual interventions

Goal:
Validate whether neurons that are semantically meaningful
are also causally responsible for model decisions.
