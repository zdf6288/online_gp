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

    # 标准化
    X_mean, X_std = X_train.mean(0), X_train.std(0)
    Y_mean, Y_std = Y_train.mean(0), Y_train.std(0)
    X_train = (X_train - X_mean) / X_std
    Y_train = (Y_train - Y_mean) / Y_std
    X_test = (X_test - X_mean) / X_std
    Y_test = (Y_test - Y_mean) / Y_std
    return X_train, Y_train, X_test, Y_test


def train_loggp_on_sarcos():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    X_train, Y_train, X_test, Y_test = load_sarcos_data()
    x_dim, y_dim = X_train.shape[1], 1
    Y_train, Y_test = Y_train[:, :1], Y_test[:, :1]

    max_data_per_expert = 300
    max_experts = 32
    lr = 0.005
    steps = 10
    batch_size = 200
    reoptimize_every = 5
    pretrain_limit = 1000
    iteration = 30

    model = LoGGP_PyTorch(x_dim, y_dim,
                          max_data_per_expert=max_data_per_expert,
                          max_experts=max_experts,
                          lr=lr,
                          steps=steps,
                          device=device)

    batch_mses = []
    no_room_batches = []

    print("Training (up to 30 batches)...")
    for batch_i in range(0, X_train.shape[0], batch_size):
        batch_idx = batch_i // batch_size
        if batch_idx >= iteration:
            break

        X_batch = X_train[batch_i:batch_i + batch_size]
        Y_batch = Y_train[batch_i:batch_i + batch_size]

        optimize = batch_i + batch_size <= pretrain_limit
        count_before = model.count

        for step_id, (x, y) in enumerate(zip(X_batch, Y_batch)):
            model.update_point(x, y, step_id=batch_i + step_id, optimize=optimize)

        if model.auxUbic[-1] != -1 and model.count == count_before:
            no_room_batches.append(batch_idx)

        if (batch_idx + 1) % reoptimize_every == 0:
            print(f"🔧 Re-optimizing hyperparameters at Batch {batch_idx + 1}")
            for model_id in model.model_params.keys():
                if model.children[0, model_id] == -1:
                    model.optimize_model(model_id)

        batch_preds = [model.predict(x, return_std=False) for x in X_batch]
        batch_preds = np.array(batch_preds)
        batch_mse = np.mean((batch_preds - Y_batch) ** 2)
        batch_mses.append(batch_mse)

        print(f"Batch {batch_idx + 1}, Train MSE: {batch_mse:.4f}")

    # 📈 Plot
    plt.figure(figsize=(10, 5))
    plt.plot(batch_mses, marker='o', label='Train MSE')
    for b in no_room_batches:
        plt.axvline(x=b, color='red', linestyle='--', alpha=0.6,
                    label='No room for divide' if b == no_room_batches[0] else "")
    plt.title("MSE vs. Batch Index (First 30 Batches) 1 Dimensional output")
    plt.xlabel("Batch Index")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mse_vs_batch_30_batches.png")
    plt.show()

    # 🧪 Test
    print("Testing...")
    predictions = [model.predict(x, return_std=False) for x in tqdm(X_test)]
    predictions = np.array(predictions)
    test_mse = np.mean((predictions - Y_test) ** 2)
    print(f"Test MSE after 30 batches: {test_mse:.4f}")
        
        # 可视化预测 vs 真实值（适用于 1D 输出）
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
    train_loggp_on_sarcos()
