import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

tf.random.set_seed(42)
np.random.seed(42)

X, y = make_moons(n_samples=500, noise=0.25, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

trials = [
    {"lr": 0.01,  "hidden_units": 16, "dropout": 0.0},
    {"lr": 0.1,   "hidden_units": 16, "dropout": 0.0},
    {"lr": 0.01,  "hidden_units": 64, "dropout": 0.0},
    {"lr": 0.01,  "hidden_units": 64, "dropout": 0.3},
    {"lr": 0.001, "hidden_units": 128, "dropout": 0.5},
]

results = []

for trial_num, params in enumerate(trials, start=1):
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    model = Sequential([
        Input(shape=(2,)),
        Dense(params["hidden_units"], activation="relu"),
        Dropout(params["dropout"]),
        Dense(params["hidden_units"] // 2, activation="relu"),
        Dropout(params["dropout"]),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(learning_rate=params["lr"]),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60, batch_size=16, verbose=0,
    )

    train_loss = history.history["loss"][-1]
    train_acc = history.history["accuracy"][-1]
    val_loss = history.history["val_loss"][-1]
    val_acc = history.history["val_accuracy"][-1]

    gap = train_acc - val_acc
    overfitting = "Yes" if gap > 0.07 else "No"

    results.append({
        "trial": trial_num,
        "params": params,
        "train_loss": train_loss,
        "val_acc": val_acc,
        "overfitting": overfitting,
        "gap": gap,
    })

    print(f"Trial {trial_num}: lr={params['lr']}, units={params['hidden_units']}, "
          f"dropout={params['dropout']} -> Train Loss={train_loss:.4f}, "
          f"Val Acc={val_acc:.4f}, Train-Val Gap={gap:.4f}, Overfitting={overfitting}")

print("\n--- Output Analysis & Optimization Log (copy into table) ---")
print(f"{'Trial':<7}{'Hyperparameters':<38}{'Train Loss':<13}{'Val Acc':<10}{'Overfit?':<8}")
for r in results:
    p = r["params"]
    hp_str = f"lr={p['lr']}, units={p['hidden_units']}, dropout={p['dropout']}"
    print(f"{r['trial']:<7}{hp_str:<38}{r['train_loss']:<13.4f}{r['val_acc']:<10.4f}{r['overfitting']:<8}")

best = max(results, key=lambda r: r["val_acc"])
print(f"\nBest configuration: Trial {best['trial']} -> {best['params']} "
      f"(Val Acc={best['val_acc']:.4f}, Overfitting={best['overfitting']})")

plt.figure(figsize=(9, 5))
trial_labels = [f"Trial {r['trial']}" for r in results]
val_accs = [r["val_acc"] for r in results]
colors = ["green" if r["overfitting"] == "No" else "red" for r in results]
plt.bar(trial_labels, val_accs, color=colors)
plt.ylabel("Validation Accuracy")
plt.title("Lab 3: Validation Accuracy per Hyperparameter Trial\n(Green = No Overfitting, Red = Overfitting)")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3, axis="y")
plt.savefig("outputs/hyperparam_search_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved plot: outputs/hyperparam_search_results.png")
