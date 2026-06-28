import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

tf.random.set_seed(42)
np.random.seed(42)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([0, 1, 1, 0], dtype=np.float32)

EPOCHS = 100
configs = [
    ("Sigmoid", 0.01),
    ("Sigmoid", 0.1),
    ("ReLU", 0.01),
    ("ReLU", 0.1),
]

results = []
histories = {}

for activation, lr in configs:
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    act_name = "sigmoid" if activation == "Sigmoid" else "relu"
    model = Sequential([
        tf.keras.layers.Input(shape=(2,)),
        Dense(8, activation=act_name),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=SGD(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(X, y, epochs=EPOCHS, verbose=0)

    final_loss = history.history["loss"][-1]
    final_acc = history.history["accuracy"][-1]
    results.append((activation, lr, final_loss, final_acc))
    histories[f"{activation}_{lr}"] = history.history["loss"]

    print(f"Activation={activation:8s} | LR={lr:5.2f} | "
          f"Final Loss={final_loss:.4f} | Final Acc={final_acc:.2f}")

print("\n--- Hyperparameter Experimentation Log (copy into table) ---")
print(f"{'Exp#':<5}{'Activation':<12}{'LR':<8}{'Final Loss':<14}{'Final Acc.':<10}")
for idx, (act, lr, loss, acc) in enumerate(results, start=1):
    print(f"{idx:<5}{act:<12}{lr:<8}{loss:<14.4f}{acc:<10.2f}")

plt.figure(figsize=(8, 5))
for label, loss_curve in histories.items():
    plt.plot(loss_curve, label=label)
plt.xlabel("Epoch")
plt.ylabel("Loss (binary crossentropy)")
plt.title("Lab 2: MLP Loss Curves - Sigmoid vs ReLU at Different Learning Rates")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("outputs/mlp_loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved plot: outputs/mlp_loss_curves.png")
