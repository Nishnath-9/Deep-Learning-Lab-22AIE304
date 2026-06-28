"""
Run this once before Lab4_DNN_MNIST.py if mnist.npz is not already present.

On most machines (and Google Colab), tf.keras.datasets.mnist.load_data()
works directly with no setup. This helper exists only as a fallback for
restricted/offline environments, converting the classic mnielsen mirror
into the same .npz format Lab4 expects.
"""

import gzip
import pickle
import os
import urllib.request
import numpy as np

MIRROR_URL = "https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/master/data/mnist.pkl.gz"
PKL_PATH = "mnist.pkl.gz"
NPZ_PATH = "mnist.npz"


def main():
    if os.path.exists(NPZ_PATH):
        print(f"{NPZ_PATH} already exists, skipping download.")
        return

    print("Downloading MNIST from mirror...")
    urllib.request.urlretrieve(MIRROR_URL, PKL_PATH)

    print("Converting to mnist.npz ...")
    with gzip.open(PKL_PATH, "rb") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

    X_train_full, y_train_full = train_set
    X_valid, y_valid = valid_set
    X_test, y_test = test_set

    X_train = np.concatenate([X_train_full, X_valid], axis=0)
    y_train = np.concatenate([y_train_full, y_valid], axis=0)

    x_train_img = (X_train.reshape(-1, 28, 28) * 255).astype(np.uint8)
    x_test_img = (X_test.reshape(-1, 28, 28) * 255).astype(np.uint8)
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    np.savez(NPZ_PATH, x_train=x_train_img, y_train=y_train,
             x_test=x_test_img, y_test=y_test)

    os.remove(PKL_PATH)
    print(f"Saved {NPZ_PATH} ({x_train_img.shape[0]} train, {x_test_img.shape[0]} test images)")


if __name__ == "__main__":
    main()
