import tensorflow as tf
import numpy as np

LAYER_NAME = "conv5_block3_out"


def get_backbone_and_head(model):
    backbone = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(LAYER_NAME).output
    )

    x = model.get_layer(LAYER_NAME).output
    for layer in model.layers[model.layers.index(model.get_layer(LAYER_NAME)) + 1:]:
        x = layer(x)

    head = tf.keras.Model(
        inputs=model.get_layer(LAYER_NAME).output,
        outputs=x
    )

    return backbone, head


def spatial_average(activations):
    # (1, H, W, C) → (1, C)
    return activations.mean(axis=(1, 2))
