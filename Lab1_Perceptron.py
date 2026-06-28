"""
Lab 1: Perceptron Learning Implementation
Course: Deep Learning (22AIE304)

Fills in the activity sheet's three blanks:
  1. linear_output = np.dot(X[i], weights) + bias
  2. weights += update * X[i]
  3. bias    += update
"""

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)


def train_perceptron(X, y, lr=0.1, epochs=10):
    weights = np.zeros(X.shape[1])
    bias = 0.0

    for _ in range(epochs):
        for i in range(len(X)):
            # 1. Calculate Linear Combination
            linear_output = np.dot(X[i], weights) + bias

            # 2. Apply Activation Function (Heaviside step)
            y_pred = 1 if linear_output >= 0 else 0

            # 3. Compute Update (Error * Learning Rate)
            update = lr * (y[i] - y_pred)

            # 4. Update Weights and Bias
            weights += update * X[i]
            bias += update

    return weights, bias


def predict(X, weights, bias):
    return np.array([1 if np.dot(x, weights) + bias >= 0 else 0 for x in X])


def plot_decision_boundary(X, y, weights, bias, gate_name, filename):
    plt.figure(figsize=(6, 6))
    for label, marker, color in [(0, "o", "red"), (1, "s", "blue")]:
        pts = X[y == label]
        plt.scatter(pts[:, 0], pts[:, 1], marker=marker, color=color,
                    s=150, label=f"Class {label}", edgecolors="black", zorder=3)

    # Decision boundary line: w1*x1 + w2*x2 + b = 0  ->  x2 = -(w1*x1 + b)/w2
    x_vals = np.linspace(-0.5, 1.5, 100)
    if weights[1] != 0:
        y_vals = -(weights[0] * x_vals + bias) / weights[1]
        plt.plot(x_vals, y_vals, "k--", linewidth=2, label="Decision Boundary")
    else:
        plt.axvline(x=-bias / weights[0], color="k", linestyle="--", linewidth=2)

    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Perceptron Decision Boundary ({gate_name} Gate)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {filename}")


if __name__ == "__main__":
    datasets = {
        "AND": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1])),
        "OR":  (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 1])),
    }

    for gate_name, (X, y) in datasets.items():
        weights, bias = train_perceptron(X, y, lr=0.1, epochs=10)
        preds = predict(X, weights, bias)
        accuracy = np.mean(preds == y) * 100

        print(f"\n--- {gate_name} Gate ---")
        print(f"Final Weights: w1 = {weights[0]:.4f}, w2 = {weights[1]:.4f}")
        print(f"Final Bias: b = {bias:.4f}")
        print(f"Decision Boundary Equation: {weights[0]:.4f}*x1 + {weights[1]:.4f}*x2 + ({bias:.4f}) = 0")
        print(f"Predictions: {preds}  | Actual: {y}")
        print(f"Accuracy: {accuracy:.1f}%  | Converged: {'Yes' if accuracy == 100 else 'No'}")

        plot_decision_boundary(X, y, weights, bias, gate_name,
                                f"outputs/perceptron_{gate_name.lower()}_boundary.png")

    # ---- XOR demonstration for the Critical Thinking question ----
    print("\n--- XOR Gate (non-linearly separable) ---")
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    w_xor, b_xor = train_perceptron(X_xor, y_xor, lr=0.1, epochs=20)
    preds_xor = predict(X_xor, w_xor, b_xor)
    acc_xor = np.mean(preds_xor == y_xor) * 100
    print(f"Final Weights: w1 = {w_xor[0]:.4f}, w2 = {w_xor[1]:.4f}")
    print(f"Final Bias: b = {b_xor:.4f}")
    print(f"Predictions: {preds_xor} | Actual: {y_xor}")
    print(f"Accuracy: {acc_xor:.1f}% (weights oscillate, never converges to 100%)")
