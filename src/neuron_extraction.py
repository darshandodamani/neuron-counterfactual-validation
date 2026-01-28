import tensorflow as tf
import numpy as np


def get_activation_model(model, layer_name):
    return tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )


def extract_activations(model, image):
    activations = model.predict(image)
    return activations  # shape: (1, H, W, C)
