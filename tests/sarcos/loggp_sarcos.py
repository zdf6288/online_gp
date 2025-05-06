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
    Y_train, Y_test = Y_train[:, :1], Y_test[:, :1]  # 只预测第一个输出维度

    model = LoGGP_PyTorch(
        x_dim, y_dim,
        max_data_per_expert=50,
        max_experts=64,
        lr=0.001,
        steps=5,
        device=device,
        min_points_to_optimize=10,
        optimize_every=10
    )

    smse_list = []
    preds = []
    trues = []
    sample_count = 0
    max_samples = 5000
    pretrain_limit = 1000
    report_every = 100  # 每 N 个样本输出一次平均 SMSE
    window_preds, window_trues = [], []

    #开始训练
    print("Training (sample-wise)...")
    
    for x, y in tqdm(zip(X_train, Y_train), total=min(len(X_train), max_samples)):
        if sample_count >= max_samples:
            break

        x = x.reshape(-1)
        y = y.reshape(-1)

        # 1. 预测
        pred, _ = model.predict(x, return_std=True)
        preds.append(pred)
        trues.append(y)
        
        window_preds.append(pred)
        window_trues.append(y)

        # 2. 更新模型
        model.update_point(x, y, optimize=True)

        # 3. 记录误差
        smse = np.mean((pred - y) ** 2) / (np.var(y) + 1e-8)
        smse_list.append(smse)
        sample_count += 1

        # 4. 每 report_every 样本输出 SMSE
        if sample_count % report_every == 0:
            window_preds_np = np.array(window_preds)
            window_trues_np = np.array(window_trues)
            mse = np.mean((window_preds_np - window_trues_np) ** 2)
            variance = np.var(window_trues_np)
            window_smse = mse / (variance + 1e-8)
            print(f"[Samples {sample_count - report_every + 1}–{sample_count}] Average SMSE: {window_smse:.6f}")
            window_preds.clear()
            window_trues.clear()

    # 绘制训练误差
    plt.figure(figsize=(10, 4))
    plt.plot(smse_list, label='Sample-wise SMSE', alpha=0.7)
    plt.title("Per-sample SMSE During Online Training")
    plt.xlabel("Sample Index")
    plt.ylabel("SMSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("train_smse_per_sample.png")
    plt.show()

    # 测试阶段
    print("Testing...")
    predictions = [model.predict(x, return_std=False) for x in tqdm(X_test)]
    predictions = np.array(predictions)
    mse = np.mean((predictions - Y_test) ** 2)
    var = np.var(Y_test)
    smse = mse / (var + 1e-8)
    print(f"Final Test SMSE: {smse:.6f}")

    # 可视化预测结果
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

