import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.on_loggp_pytorch import LoGGP_PyTorch
from tools.error_bound import ErrorBoundCalculator

# 初始化模型
x_dim, y_dim = 1, 1
model = LoGGP_PyTorch(x_dim=x_dim, y_dim=y_dim, max_data_per_expert=200, max_experts=1)
err_calc = ErrorBoundCalculator(input_dim=1)

# 构造训练数据
X_train = np.random.uniform(0, 5, 100).reshape(-1, 1)
Y_train = np.sin(X_train) + 0.1 * np.random.randn(100, 1)

# 构造测试数据
X_test = np.linspace(0, 5, 100).reshape(-1, 1)
Y_true = np.sin(X_test)

# 图形初始化
fig, ax = plt.subplots(figsize=(10, 5))
line_pred, = ax.plot([], [], 'b-', label='Prediction')
line_true, = ax.plot(X_test, Y_true, 'g--', label='True sin(x)')
scat_train = ax.scatter([], [], c='k', s=10, label='Train Data')
ax.fill_between([], [], [], alpha=0.2, color='blue', label='Confidence')
ax.fill_between([], [], [], alpha=0.2, color='orange', label='Error Bound')

ax.set_xlim(0, 5)
ax.set_ylim(-1.5, 1.5)
ax.legend()
ax.grid(True)

fill_conf = None
fill_bound = None

# 初始化函数
def init():
    line_pred.set_data([], [])
    scat_train.set_offsets(np.empty((0, 2)))
    return line_pred, scat_train

import numpy as np

def update(frame):
    global fill_conf, fill_bound

    x, y = X_train[frame], Y_train[frame]
    model.update_point(x, y)

    y_pred, y_std = zip(*[model.predict(x_star, return_std=True) for x_star in X_test])
    y_pred = np.array(y_pred).ravel()
    y_std = np.array(y_std).ravel()

    if np.any(np.isnan(y_pred)) or np.any(np.isnan(y_std)) or np.any(y_std < 0):
        print(f"Warning: NaN or negative std at frame {frame}. Skipping.")
        return line_pred, scat_train

    y_bound = np.array([err_calc.compute_bound(std) for std in y_std]).ravel()

    line_pred.set_data(X_test.flatten(), y_pred)
    scat_train.set_offsets(np.c_[X_train[:frame+1].flatten(), Y_train[:frame+1].flatten()])

    if fill_conf:
        fill_conf.remove()
    if fill_bound:
        fill_bound.remove()

    fill_conf = ax.fill_between(
        X_test.flatten(),
        y_pred - 2 * y_std,
        y_pred + 2 * y_std,
        alpha=0.2, color='blue'
    )

    fill_bound = ax.fill_between(
        X_test.flatten(),
        y_pred - y_bound,
        y_pred + y_bound,
        alpha=0.2, color='orange'
    )

    return line_pred, scat_train


ani = FuncAnimation(fig, update, frames=len(X_train), init_func=init,
                    blit=False, repeat=False, interval=100)

plt.title("LoGGP with Error Bound Visualization")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()
