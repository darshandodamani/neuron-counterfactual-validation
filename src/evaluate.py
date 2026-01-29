import argparse
import os
import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model


# Allow running this file directly: `python src/evaluate.py`
# (When run this way, Python doesn't automatically put the repo root on sys.path.)
if __package__ in (None, ""):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from src.counterfactual import find_group_counterfactual

from src.neuron_extraction import (
    get_backbone_and_head,
    spatial_average
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run neuron counterfactual evaluation on a single image."
    )
    parser.add_argument(
        "--model-path",
        default="models/resnet50_gtsrb.h5",
        help="Path to a trained Keras model (.h5).",
    )
    parser.add_argument(
        "--image-path",
        default="data/gtsrb/Train/14/00014_00000.png",
        help="Path to an input image.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Resize input image to (image_size, image_size).",
    )
    return parser.parse_args()


def _find_first_image_under(directory: str) -> str | None:
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            lower = filename.lower()
            if lower.endswith(":zone.identifier"):
                continue
            if lower.endswith((".png", ".jpg", ".jpeg")):
                return os.path.join(root, filename)
    return None


def main() -> int:
    args = _parse_args()

    if not os.path.exists(args.model_path):
        print(
            "ERROR: Model file not found: "
            f"{args.model_path}\n\n"
            "Provide it via --model-path, or train the model first."
        )
        return 2

    if not os.path.exists(args.image_path):
        fallback = _find_first_image_under(
            os.path.join("data", "gtsrb", "Train"))
        if fallback is None:
            print(
                "ERROR: Image file not found: "
                f"{args.image_path}\n\n"
                "Provide it via --image-path."
            )
            return 2
        print(
            "WARNING: Default image path not found. "
            f"Using first dataset image instead: {fallback}"
        )
        args.image_path = fallback

    # Load trained model
    model = load_model(args.model_path)

    img = cv2.imread(args.image_path)
    if img is None:
        print(
            "ERROR: Failed to read image (cv2.imread returned None): "
            f"{args.image_path}"
        )
        return 2

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (args.image_size, args.image_size))
    img = img / 255.0
    image = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(image)
    true_class = preds.argmax()
    print("Original prediction:", true_class)

    # Get backbone & head
    backbone, head = get_backbone_and_head(model)

    # Extract activations
    acts = backbone.predict(image)
    neuron_vec = spatial_average(acts)

    # # Rank neurons
    # top_neuron = neuron_vec[0].argmax()
    # print("Top neuron index:", top_neuron)

    K = 5
    top_neurons = np.argsort(neuron_vec[0])[::-1][:K]
    print("Top neurons:", top_neurons)

    # Counterfactual (group intervention)
    delta_star = find_group_counterfactual(
        backbone,
        head,
        image,
        true_class,
        top_neurons,
    )

    # If delta_star is ~1.0, it often means even zeroing this group wasn't enough
    # to change the decision (evidence of distributed representation).
    if delta_star >= 0.999:
        print(
            f"No decision flip for top-{K} neurons within delta in [0, 1]. "
            f"(Even scaling them to ~0 was insufficient.)"
        )
    else:
        print(
            f"Decision flips when top-{K} neurons are reduced by ~{delta_star:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
