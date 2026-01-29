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

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train

Train the model and save it to the default path:

Note: the first run may download ResNet-50 ImageNet weights via Keras and cache them locally.

```bash
python src/train.py --data-dir data/gtsrb/Train --epochs 5 --output-model-path models/resnet50_gtsrb.h5
```

Quick smoke test (faster):

```bash
python src/train.py --data-dir data/gtsrb/Train --epochs 1 --output-model-path models/resnet50_gtsrb.h5
```

You can also run it as a module:

```bash
python -m src.train --data-dir data/gtsrb/Train --epochs 5 --output-model-path models/resnet50_gtsrb.h5
```

## Evaluate / Counterfactual

Run evaluation on a single image:

```bash
python src/evaluate.py --model-path models/resnet50_gtsrb.h5 --image-path data/gtsrb/Train/25/00025_00013_00011.png
```

You can also run it as a module:

```bash
python -m src.evaluate --model-path models/resnet50_gtsrb.h5 --image-path data/gtsrb/Train/25/00025_00013_00011.png
```
