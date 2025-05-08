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


def train_loggp_on_sarcos(enable_pretraining=True,
                          enable_online_optimization=True,
                          pretrain_limit=1000,
                          max_samples=6000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    X_train, Y_train, X_test, Y_test = load_sarcos_data()
    x_dim, y_dim = X_train.shape[1], 1
    Y_train, Y_test = Y_train[:, :1], Y_test[:, :1]  # 只使用第一个输出维度

    model = LoGGP_PyTorch(
        x_dim, y_dim,
        max_data_per_expert=100,
        max_experts=16,
        lr=0.0001,
        steps=100,
        device=device,
        min_points_to_optimize=20,
        optimize_every=5,
        enable_variance_cache=False
    )

    mse_list = []
    preds = []
    trues = []
    sample_count = 0
    report_every = 500
    window_preds = []
    window_trues = []

    print("Training (sample-wise)...")

    for x, y in tqdm(zip(X_train, Y_train), total=min(len(X_train), max_samples)):
        if sample_count >= max_samples:
            break

        x = x.reshape(-1)
        y = y.reshape(-1)

        # 1. Predict
        pred, _ = model.predict(x, return_std=True)
        preds.append(pred)
        trues.append(y)
        window_preds.append(pred)
        window_trues.append(y)

        # 2. Decide if optimize
        if enable_pretraining and sample_count < pretrain_limit:
            optimize = True
        elif enable_online_optimization:
            optimize = True
        else:
            optimize = False

        # 3. Update model
        model.update_point(x, y, optimize=optimize)

        # 4. Record MSE
        mse = np.mean((pred - y) ** 2)
        mse_list.append(mse)
        sample_count += 1

        if sample_count % report_every == 0:
            window_preds_np = np.array(window_preds)
            window_trues_np = np.array(window_trues)
            window_mse = np.mean((window_preds_np - window_trues_np) ** 2)
            print(f"[Samples {sample_count - report_every + 1}–{sample_count}] Average MSE: {window_mse:.6f}")
            window_preds.clear()
            window_trues.clear()

    # 🎯 Plot window-averaged MSE
    window_size = 20
    mse_list = np.array(mse_list)
    prepoint_mse = [np.mean(mse_list[i:i+window_size]) for i in range(0, len(mse_list), window_size)]
    prepoint_indices = list(range(window_size, len(mse_list) + 1, window_size))

    plt.figure(figsize=(10, 4))
    plt.plot(prepoint_indices, prepoint_mse, marker='o', label='Avg MSE per window')
    plt.xlabel("Sample Index")
    plt.ylabel("Average MSE")
    plt.title("MSE over Samples (Window Averaged)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("prepoint_mse_plot.png")
    plt.show()

    # ✅ Testing
    print("Testing...")
    predictions = [model.predict(x, return_std=False) for x in tqdm(X_test)]
    predictions = np.array(predictions)
    test_mse = np.mean((predictions - Y_test) ** 2)
    print(f"Final Test MSE: {test_mse:.6f}")

    # 📈 Test prediction vs true
    plt.figure(figsize=(10, 5))
    plt.plot(Y_test[:200], label="True", marker='o', alpha=0.7)
    plt.plot(predictions[:200], label="Predicted", marker='x', alpha=0.7)
    plt.title("Prediction vs. Ground Truth (First 200 test points)")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Normalized Output")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test_prediction_vs_true.png")
    plt.show()


if __name__ == "__main__":
    train_loggp_on_sarcos(
        enable_pretraining=True,
        enable_online_optimization=True,
        pretrain_limit=500,
        max_samples=5000)

