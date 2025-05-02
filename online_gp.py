import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import cho_solve, cho_factor
from scipy.spatial.distance import cdist

# === Online GP 模型（LoG-GP 无先验） ===
class OnlineGP:
    def __init__(self, noise=1e-4, max_points=30, lengthscale=1.5, variance=1.0):
        self.noise = noise
        self.max_points = max_points
        self.lengthscale = lengthscale
        self.variance = variance
        self.X = []
        self.y = []

    def rbf_kernel(self, X1, X2):
        dists = cdist(X1 / self.lengthscale, X2 / self.lengthscale, metric='sqeuclidean')
        return self.variance * np.exp(-0.5 * dists)

    def predict(self, x_star):
        if len(self.X) == 0:
            return 0.0, 1.0
        X = np.array(self.X)
        y = np.array(self.y)
        K = self.rbf_kernel(X, X) + self.noise * np.eye(len(X))
        k_star = self.rbf_kernel(X, np.atleast_2d(x_star))
        k_star_star = self.rbf_kernel(np.atleast_2d(x_star), np.atleast_2d(x_star))[0, 0]
        K_chol = cho_factor(K)
        alpha = cho_solve(K_chol, y)
        mean = k_star.T @ alpha
        v = cho_solve(K_chol, k_star)
        var = k_star_star - k_star.T @ v
        return mean.item(), np.clip(var, 1e-6, 1e6)

    def is_informative(self, x_new, y_new, threshold=0.01):
        _, var = self.predict(x_new)
        return var > threshold

    # def update(self, x_new, y_new):
    #     if self.is_informative(x_new, y_new):
    #         print(f"Adding point: {x_new}, {y_new}")
    #         self.X.append(np.array(x_new))
    #         self.y.append(y_new)
    #         if len(self.X) > self.max_points:
    #             self.X.pop(0)
    #             self.y.pop(0)
    #     else:
    #         print(f"Point {x_new} is not informative enough.")
    
    def update(self, x_new, y_new):
        self.X.append(np.array(x_new))
        self.y.append(y_new)
        if len(self.X) > self.max_points:
            self.X.pop(0)
            self.y.pop(0)
        

# === 数据生成 ===
np.random.seed(1)
X_online = (np.random.rand(100) * 5).reshape(-1, 1)
y_online = np.sin(X_online).ravel() + 0.2 * np.random.randn(100)

X_test = np.linspace(0, 5, 200).reshape(-1, 1)
y_true = np.sin(X_test).ravel()

# === 初始化模型 ===
gp = OnlineGP(max_points=30, lengthscale=1.5)

# === 模拟在线过程并记录预测历史 ===
history_preds = []
history_stds = []
added_points = []

for x_i, y_i in zip(X_online, y_online):
    gp.update(x_i, y_i)
    mean, std = zip(*[gp.predict(x) for x in X_test])
    history_preds.append(np.array(mean))
    history_stds.append(np.array(std))
    if gp.X:
        added_points.append((x_i.item(), y_i))

# === 动画部分 ===
fig, ax = plt.subplots(figsize=(10, 5))
line_pred, = ax.plot([], [], 'b', label='Online GP prediction')
line_true, = ax.plot(X_test, y_true, 'g--', label='True function')
scatter_pts = ax.scatter([], [], c='k', label='Online data')

def init():
    ax.set_xlim(0, 5)
    ax.set_ylim(-2, 2)
    return line_pred, scatter_pts

def update(frame):
    ax.collections.clear()
    y_pred = history_preds[frame].flatten()
    y_std = history_stds[frame].flatten()
    line_pred.set_data(X_test.ravel(), y_pred)
    ax.fill_between(X_test.ravel(),
                    y_pred - 2 * y_std,
                    y_pred + 2 * y_std,
                    alpha=0.2, color='blue')
    pts = np.array(added_points[:frame + 1])
    ax.scatter(pts[:, 0], pts[:, 1], c='k', label='Online data', zorder=5)
    return line_pred,

ani = FuncAnimation(fig, update, frames=len(history_preds), init_func=init,
                    blit=False, repeat=False, interval=300)

plt.title("Pure Online GP Learning Over Time")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()