import numpy as np


def apply_neuron_scaling(activations, neuron_idx, delta):
    modified = activations.copy()
    modified[..., neuron_idx] *= (1 - delta)
    return modified
