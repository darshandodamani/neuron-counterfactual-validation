import argparse
import os
import sys

# Allow running this file directly: `python src/train.py`
# (When run this way, Python doesn't automatically put the repo root on sys.path.)
if __package__ in (None, ""):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from src.data_loader import load_gtsrb
from src.model import build_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GTSRB model.")
    parser.add_argument(
        "--data-dir",
        default="data/gtsrb/Train",
        help="Path to the GTSRB training directory.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--output-model-path",
        default="models/resnet50_gtsrb.h5",
        help="Where to save the trained model (.h5).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    train_gen, val_gen = load_gtsrb(args.data_dir)

    model = build_model()
    model.summary()

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
    )

    os.makedirs(os.path.dirname(args.output_model_path) or ".", exist_ok=True)
    model.save(args.output_model_path)
    print(f"Model saved to {args.output_model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
