"""
Lab 4: Implementing a Deep Neural Network (DNN) for Digit Classification
Course: Deep Learning (22AIE304)

Fills in the activity sheet's blanks:
  x_train = x_train / 255.0
  x_test  = x_test / 255.0
  Two hidden Dense(ReLU) layers
  optimizer='adam', loss='sparse_categorical_crossentropy'

Also generates the confusion matrix and pulls 3 misclassified images
required for Part 2 / Part 3 of the activity sheet.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

os.makedirs("outputs", exist_ok=True)

tf.random.set_seed(42)

# 1. Load and Preprocess Data
# NOTE: tf.keras.datasets.mnist.load_data() downloads from Google's servers
# and works fine on a normal machine/Colab. Uncomment the line below to use it:
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#
# A local mnist.npz (same format) is used here instead so the script runs
# without needing internet access to Google's storage:
mnist_path = "mnist.npz"  # place mnist.npz alongside this script
with np.load(mnist_path) as data:
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

# TODO: Normalize the pixel values (0-255) to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2. Build the DNN Architecture
model = models.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    # TODO: Add two dense hidden layers with ReLU activation
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    # Output layer for 10 classes
    layers.Dense(10, activation="softmax"),
])

# 3. Compile the Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=32,
                     validation_split=0.1, verbose=2)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

# ---- Part 2: Confusion Matrix ----
y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Find the most-confused off-diagonal digit pairs
cm_off_diag = cm.copy()
np.fill_diagonal(cm_off_diag, 0)
top_confusions = []
flat_indices = np.argsort(cm_off_diag.flatten())[::-1][:5]
for idx in flat_indices:
    true_label, pred_label = np.unravel_index(idx, cm_off_diag.shape)
    count = cm_off_diag[true_label, pred_label]
    if count > 0:
        top_confusions.append((true_label, pred_label, count))

print("\nTop confused digit pairs (true -> predicted : count):")
for true_l, pred_l, count in top_confusions:
    print(f"  {true_l} -> {pred_l} : {count} times")

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap="Blues", colorbar=True)
plt.title(f"Lab 4: Confusion Matrix (Test Accuracy = {test_acc:.4f})")
plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved plot: outputs/confusion_matrix.png")

# ---- Part 3: Error Analysis - find 3 misclassified images ----
misclassified_idx = np.where(y_pred != y_test)[0]
chosen = misclassified_idx[:3]

fig, axes = plt.subplots(1, 3, figsize=(10, 4))
for ax, idx in zip(axes, chosen):
    ax.imshow(x_test[idx], cmap="gray")
    ax.set_title(f"ID: {idx}\nTrue: {y_test[idx]}, Pred: {y_pred[idx]}")
    ax.axis("off")
plt.suptitle("Lab 4 Part 3: Misclassified Digit Examples")
plt.savefig("outputs/misclassified_examples.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot: outputs/misclassified_examples.png")

print("\n--- Error Analysis Table (copy into Part 3) ---")
for idx in chosen:
    true_l = y_test[idx]
    pred_l = y_pred[idx]
    confidence = y_pred_probs[idx][pred_l] * 100
    print(f"Image ID {idx}: True={true_l}, Predicted={pred_l}, "
          f"Model confidence in wrong label={confidence:.1f}%")
