import numpy as np
import torch
from scipy.io import loadmat
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.on_loggp_pytorch import LoGGP_PyTorch

def load_sarcos_data():
    train_data = loadmat('sarcos_inv.mat')
    test_data = loadmat('sarcos_inv_test.mat')
    X_train = train_data['sarcos_inv'][:, :21]
    Y_train = train_data['sarcos_inv'][:, 21:]
    X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    Y_train = (Y_train - Y_train.mean(axis=0)) / Y_train.std(axis=0)
    X_test = test_data['sarcos_inv_test'][:, :21]
    Y_test = test_data['sarcos_inv_test'][:, 21:]
    return X_train, Y_train, X_test, Y_test


def train_loggp_on_sarcos():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    X_train, Y_train, X_test, Y_test = load_sarcos_data()
    x_dim = X_train.shape[1]
    y_dim = Y_train.shape[1]

    model = LoGGP_PyTorch(x_dim, y_dim, max_data_per_expert=300, max_experts=32, lr=0.005, steps=10, device=device)

    batch_size = 200
    print("Training...")
    batch_mses = []

    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i + batch_size]
        Y_batch = Y_train[i:i + batch_size]

        for x, y in tqdm(zip(X_batch, Y_batch), total=len(X_batch), desc=f"Batch {i//batch_size + 1:02d}"):
            model.update_point(x, y, optimize=True)

        # for model_id in model.model_params.keys():
        #     if model.children[0, model_id] == -1:
        #         model.optimize_model(model_id)

        # Compute training MSE on batch
        batch_preds = [model.predict(x, return_std=False) for x in X_batch]
        batch_preds = np.array(batch_preds)
        batch_mse = np.mean((batch_preds - Y_batch) ** 2)
        batch_mses.append(batch_mse)

        print(f"Batch {i // batch_size + 1}, Train MSE: {batch_mse:.4f}")

    # Plot training MSE
    plt.figure(figsize=(10, 4))
    plt.plot(batch_mses, marker='o')
    plt.title("Training MSE per Batch")
    plt.xlabel("Batch Index")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_mse_plot.png")
    plt.show()

    # Final prediction on test set
    # Final prediction on test set
    print("Predicting...")
    predictions = [model.predict(x, return_std=False) for x in tqdm(X_test, desc="Testing")]
    predictions = np.array(predictions)
    test_mse = np.mean((predictions - Y_test) ** 2)
    print(f"Test MSE: {test_mse:.4f}")


if __name__ == "__main__":
    train_loggp_on_sarcos()
