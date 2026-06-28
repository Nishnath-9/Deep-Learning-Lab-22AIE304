
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

os.makedirs("outputs", exist_ok=True)

tf.random.set_seed(42)

mnist_path = "mnist.npz"  
with np.load(mnist_path) as data:
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

x_train = x_train / 255.0
x_test = x_test / 255.0

model = models.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),

    layers.Dense(10, activation="softmax"),
])

#
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


history = model.fit(x_train, y_train, epochs=5, batch_size=32,
                     validation_split=0.1, verbose=2)


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

y_pred_probs = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

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
