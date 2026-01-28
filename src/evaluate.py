try:
    # Works when running as a module: `python -m src.evaluate`
    from src.data_loader import load_gtsrb
except ModuleNotFoundError:
    # Works when running as a script: `python src/evaluate.py`
    from data_loader import load_gtsrb
import matplotlib.pyplot as plt

train_gen, val_gen = load_gtsrb("data/gtsrb/Train")

print("Number of classes:", train_gen.num_classes)
print("Training samples:", train_gen.samples)
print("Validation samples:", val_gen.samples)
print("Class indices:", train_gen.class_indices)
images, labels = next(train_gen)

plt.imshow(images[0])
plt.title(f"Label index: {labels[0].argmax()}")
plt.axis("off")
plt.show()
