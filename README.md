# Deep Learning Lab — 22AIE304

Implementations for the Deep Learning Lab Activity Sheet (Course Code: 22AIE304).
Each script fills in the code blanks from the activity sheet and produces the
output/screenshots needed for the written analysis sections.

## Contents

| File | Lab | What it does |
|---|---|---|
| `Lab1_Perceptron.py` | Lab 1 | Perceptron from scratch (NumPy). Trains on AND, OR, and XOR gates. Prints final weights/bias/accuracy and saves decision boundary plots. |
| `Lab2_MLP.py` | Lab 2 | Keras MLP solving XOR. Runs the 4 required experiments (Sigmoid/ReLU × lr 0.01/0.1) and saves a loss-curve comparison plot. |
| `Lab3_Hyperparameter_Optimization.py` | Lab 3 | Real grid search (5 trials) over learning rate / hidden units / dropout on a non-linearly separable dataset (`sklearn.make_moons`). Logs train loss, validation accuracy, and an overfitting flag per trial. |
| `Lab4_DNN_MNIST.py` | Lab 4 | DNN (2 hidden ReLU layers + softmax output) for MNIST digit classification. Generates a confusion matrix and pulls 3 misclassified images for error analysis. |
| `download_mnist.py` | Lab 4 helper | One-time helper to fetch `mnist.npz` if `tf.keras.datasets.mnist.load_data()` is blocked on your network (e.g. restricted sandbox/CI). On a normal machine or Colab, you don't need this — the built-in loader just works. |
| `outputs/` | All | Generated plots: decision boundaries, loss curves, hyperparameter results, confusion matrix, misclassified digit examples. |

## Setup

```bash
pip install numpy tensorflow matplotlib scikit-learn
```

## Running

```bash
python Lab1_Perceptron.py
python Lab2_MLP.py
python Lab3_Hyperparameter_Optimization.py

# Lab 4: MNIST loads automatically via Keras on most machines.
# If you hit a network error downloading it, run this once first:
python download_mnist.py
python Lab4_DNN_MNIST.py
```

## Results Summary

**Lab 1 — Perceptron**
- AND gate: 100% accuracy, converged
- OR gate: 100% accuracy, converged
- XOR gate: ~50% accuracy, does **not** converge (non-linearly separable — consistent with the Perceptron Convergence Theorem)

**Lab 2 — MLP on XOR**

| Exp # | Activation | Learning Rate | Final Loss | Final Acc. |
|---|---|---|---|---|
| 1 | Sigmoid | 0.01 | ~0.70 | 0.50 |
| 2 | Sigmoid | 0.1  | ~0.70 | 0.50 |
| 3 | ReLU    | 0.01 | ~0.67 | 0.50 |
| 4 | ReLU    | 0.1  | ~0.57 | 0.75 |

(Exact values vary slightly by run/seed — see console output for your actual numbers to copy into the sheet.)

**Lab 3 — Hyperparameter Search (5 trials, `make_moons` dataset)**
Best configuration found: `lr=0.001, hidden_units=128, dropout=0.5` — highest validation accuracy with no overfitting.

**Lab 4 — DNN on MNIST**
- Test accuracy: **~97.5%**
- Most confused digit pairs: 9→4, 7→2, 4→6, 5→3 (classic MNIST ambiguous shapes)

## Notes
- Run each script yourself and use the exact printed numbers/plots for your activity sheet — values can shift slightly by machine, library version, or random seed.
- Plots are saved to `outputs/` automatically when each script runs.
