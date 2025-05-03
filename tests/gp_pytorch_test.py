import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.online_gp_pytorch import OnlineGPTorch
from tools.error_bound import ErrorBoundCalculator

# === 数据生成 ===
X_online = (np.random.rand(100) * 5).reshape(-1, 1)
y_online = np.sin(X_online).ravel() + 0.2 * np.random.randn(100)
X_test = np.linspace(0, 5, 200).reshape(-1, 1)
y_true = np.sin(X_test).ravel()

# 初始化模型、误差计算器和优化器
gp = OnlineGPTorch(max_points=50)
optimizer = torch.optim.Adam(gp.parameters(), lr=0.05)
err_calc = ErrorBoundCalculator(input_dim=1)

# 记录动画数据
history_means = []
history_stds = []
history_bounds = []
used_points = []

# 在线训练并记录中间状态
for i, (x_i, y_i) in enumerate(zip(X_online, y_online)):
    gp.update(x_i, y_i)
    used_points.append((x_i.item(), y_i))

    if len(gp.X) >= 2:
        optimizer.zero_grad()
        loss = -gp.marginal_log_likelihood()
        loss.backward()
        optimizer.step()

    if i % 2 == 0:
        means, stds, bounds = [], [], []
        for x_star in X_test:
            mean, std = gp.predict(torch.tensor(x_star.reshape(1, -1), dtype=torch.float64))
            means.append(mean)
            stds.append(std)
            bounds.append(err_calc.compute_bound(std))
        history_means.append(means)
        history_stds.append(stds)
        history_bounds.append(bounds)

# === 创建动画 ===
fig, ax = plt.subplots(figsize=(10, 5))
line_pred, = ax.plot([], [], 'b', label='Predicted mean')
line_true, = ax.plot(X_test, y_true, 'g--', label='True function')
scatter_pts = ax.scatter([], [], c='k', s=20, label='Training points')

# 提前创建图例的图层（仅用于 legend 显示）
ax.fill_between([], [], [], alpha=0.2, color='blue', label='Confidence ±2σ')
ax.fill_between([], [], [], alpha=0.2, color='orange', label='Error Bound')

fill_conf = None
fill_bound = None

def init():
    ax.set_xlim(0, 5)
    ax.set_ylim(-2, 2)
    ax.legend()
    return line_pred, scatter_pts

def update(frame):
    global fill_conf, fill_bound
    if fill_conf:
        fill_conf.remove()
    if fill_bound:
        fill_bound.remove()

    y_pred = history_means[frame]
    y_std = history_stds[frame]
    y_bound = history_bounds[frame]

    line_pred.set_data(X_test.ravel(), y_pred)
    fill_conf = ax.fill_between(X_test.ravel(),
                                np.array(y_pred) - 2 * np.array(y_std),
                                np.array(y_pred) + 2 * np.array(y_std),
                                alpha=0.2, color='blue')

    fill_bound = ax.fill_between(X_test.ravel(),
                                 np.array(y_pred) - np.array(y_bound),
                                 np.array(y_pred) + np.array(y_bound),
                                 alpha=0.2, color='orange')

    pts = np.array(used_points[:(frame * 2) + 1])
    scatter_pts.set_offsets(pts)
    return line_pred, scatter_pts

ani = FuncAnimation(fig, update, frames=len(history_means),
                    init_func=init, blit=False, repeat=False, interval=300)

plt.title("Online GPR with Error Bound Visualization")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()