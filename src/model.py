import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

NUM_CLASSES = 43


def build_model():
    # Load ResNet-50 WITHOUT top
    base_model = ResNet50(
        # Use the built-in ImageNet weights to ensure compatibility with the
        # ResNet50 implementation shipped with the installed TF/Keras.
        # (The local H5 weight files in models/ may be incompatible across
        # versions due to differing layer ordering/naming.)
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze backbone for POC
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
