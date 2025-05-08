import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.online_gp_pytorch import OnlineGPTorch


def load_sarcos_data():
    train_data = loadmat('sarcos_inv.mat')
    test_data = loadmat('sarcos_inv_test.mat')
    X_train = train_data['sarcos_inv'][:, :21]
    Y_train = train_data['sarcos_inv'][:, 21:]
    X_test = test_data['sarcos_inv_test'][:, :21]
    Y_test = test_data['sarcos_inv_test'][:, 21:]

    # Normalize
    X_mean, X_std = X_train.mean(0), X_train.std(0)
    Y_mean, Y_std = Y_train.mean(0), Y_train.std(0)
    X_train = (X_train - X_mean) / X_std
    Y_train = (Y_train - Y_mean) / Y_std
    X_test = (X_test - X_mean) / X_std
    Y_test = (Y_test - Y_mean) / Y_std

    return X_train, Y_train, X_test, Y_test


def train_online_gp_on_sarcos(optimize_every=100):
    X_train, Y_train, X_test, Y_test = load_sarcos_data()
    y_train = Y_train[:, 0]  # Only use the first output dimension
    y_test = Y_test[:, 0]

    model = OnlineGPTorch(max_points=50, lr=0.001, optimize_steps=5)

    predictions = []
    trues = []
    sample_mses = []
    max_samples = 1000

    print("Training Online GP on SARCOS...")
    for i in tqdm(range(min(len(X_train), max_samples))):
        x = X_train[i]
        y = y_train[i]
        pred_mean, _ = model.predict(torch.tensor(x, dtype=torch.float64).unsqueeze(0))
        predictions.append(pred_mean)
        trues.append(y)
        sample_mse = (pred_mean - y) ** 2
        sample_mses.append(sample_mse)

        model.add_point(x, y)  # 不自动优化，只加点
        if (i + 1) % optimize_every == 0 and len(model.X) >= 2:
            for _ in range(model.optimize_steps):
                model.optimizer.zero_grad()
                loss = -model.marginal_log_likelihood()
                loss.backward()
                model.optimizer.step()
            print(f"[{i+1}] Optimization step completed.")

        if (i + 1) % 100 == 0:
            avg_mse = np.mean(sample_mses[-100:])
            print(f"[Samples {i - 99}–{i + 1}] Avg MSE: {avg_mse:.4f}")

    # Plot MSE curve
    plt.figure(figsize=(10, 4))
    plt.plot(sample_mses, label="Per-sample MSE")
    plt.xlabel("Sample Index")
    plt.ylabel("MSE")
    plt.title("Online GP Sample-wise MSE (Train)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("onlinegp_train_mse.png")
    plt.show()

    # Final testing
    print("Testing...")
    preds_test = []
    for x in tqdm(X_test):
        pred, _ = model.predict(torch.tensor(x, dtype=torch.float64).unsqueeze(0))
        preds_test.append(pred)

    preds_test = np.array(preds_test)
    test_mse = np.mean((preds_test - y_test) ** 2)
    print(f"Final Test MSE: {test_mse:.4f}")

    # Plot predictions vs ground truth
    plt.figure(figsize=(10, 5))
    plt.plot(y_test[:200], label="True", marker='o', alpha=0.7)
    plt.plot(preds_test[:200], label="Predicted", marker='x', alpha=0.7)
    plt.title("Prediction vs. Ground Truth (First 200 test points)")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Normalized Output")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("onlinegp_test_prediction_vs_true.png")
    plt.show()


if __name__ == "__main__":
    train_online_gp_on_sarcos(optimize_every=200)

