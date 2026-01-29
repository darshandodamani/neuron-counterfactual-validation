import numpy as np

def scale_neurons(activations, neuron_indices, delta):
    modified = activations.copy()
    for idx in neuron_indices:
        modified[..., idx] *= (1.0 - delta)
    return modified


def find_group_counterfactual(
    backbone,
    head,
    image,
    true_class,
    neuron_indices,
    steps=10,
):
    activations = backbone.predict(image)

    low, high = 0.0, 1.0

    for _ in range(steps):
        mid = (low + high) / 2.0
        modified = scale_neurons(activations, neuron_indices, mid)
        preds = head.predict(modified)
        pred_class = preds.argmax()

        if pred_class != true_class:
            high = mid
        else:
            low = mid

    return high

