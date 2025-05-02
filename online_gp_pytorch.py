import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from kernels.rbf_kernel import RBFKernel

# === Online GPR 模型 ===
class OnlineGPTorch(torch.nn.Module):
    def __init__(self, max_points=30):
        super().__init__()
        self.X = []
        self.y = []
        self.max_points = max_points
        self.log_lengthscale = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.log_variance = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.log_noise = torch.nn.Parameter(torch.tensor(-2.0, dtype=torch.float64))
        self.kernel = RBFKernel()
        

    def predict(self, x_star):
        if len(self.X) == 0:
            return torch.tensor(0.0), torch.tensor(1.0)

        X_train = torch.stack(self.X)
        y_train = torch.stack(self.y)

        lengthscale = torch.exp(self.log_lengthscale)
        variance = torch.exp(self.log_variance)
        noise = torch.exp(self.log_noise)

        self.kernel.lengthscale = lengthscale
        self.kernel.variance = variance
        K = self.kernel(X_train, X_train)
        K = self.kernel(X_train, X_train)
        K += noise**2 * torch.eye(len(X_train), dtype=torch.float64)
        K += 1e-6 * torch.eye(K.size(0), dtype=torch.float64)

        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y_train.unsqueeze(1), L)

        k_star = self.kernel(X_train, x_star)
        k_star_star = self.kernel(x_star, x_star)[0, 0]

        mean = k_star.t().matmul(alpha).squeeze()
        v = torch.cholesky_solve(k_star, L)
        var = k_star_star - k_star.t().matmul(v).squeeze()
        return mean.item(), max(var.item(), 1e-6)

    def update(self, x_new, y_new):
        self.X.append(torch.tensor(x_new, dtype=torch.float64))
        self.y.append(torch.tensor(y_new, dtype=torch.float64))
        if len(self.X) > self.max_points:
            self.X.pop(0)
            self.y.pop(0)

    def marginal_log_likelihood(self):
        if len(self.X) == 0:
            return torch.tensor(0.0)

        X_train = torch.stack(self.X)
        y_train = torch.stack(self.y).unsqueeze(1)

        lengthscale = torch.exp(self.log_lengthscale)
        variance = torch.exp(self.log_variance)
        noise = torch.exp(self.log_noise)

        self.kernel.lengthscale = lengthscale
        self.kernel.variance = variance
        K = self.kernel(X_train, X_train)
        K += noise**2 * torch.eye(len(X_train), dtype=torch.float64)
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y_train, L)

        log_det = torch.sum(torch.log(torch.diag(L)))
        n = len(X_train)
        ll = -0.5 * y_train.t().matmul(alpha).squeeze() - log_det - 0.5 * n * torch.log(torch.tensor(2 * np.pi, dtype=torch.float64))
        return ll
    
# 数据生成
# np.random.seed(1)
X_online = (np.random.rand(100) * 5).reshape(-1, 1)
y_online = np.sin(X_online).ravel() + 0.2 * np.random.randn(100)
X_test = np.linspace(0, 5, 200).reshape(-1, 1)
y_true = np.sin(X_test).ravel()

# 记录每一帧的预测历史
history_means = []
history_stds = []
used_points = []

# 初始化模型、优化器
gp = OnlineGPTorch(max_points=50)
optimizer = torch.optim.Adam(gp.parameters(), lr=0.05)
X_test_tensor = torch.tensor(X_test, dtype=torch.float64)

# 执行在线学习并记录中间状态
for i, (x_i, y_i) in enumerate(zip(X_online, y_online)):
    gp.update(x_i, y_i)
    used_points.append((x_i.item(), y_i))

    if len(gp.X) >= 2:
        optimizer.zero_grad()
        loss = -gp.marginal_log_likelihood()
        loss.backward()
        optimizer.step()

    # 每若干步存一次帧数据（减轻内存压力）
    if i % 2 == 0:
        means, stds = [], []
        for x_star in X_test:
            mean, std = gp.predict(torch.tensor(x_star.reshape(1, -1), dtype=torch.float64))
            means.append(mean)
            stds.append(std)
        history_means.append(means)
        history_stds.append(stds)

# === 创建动画 ===
fig, ax = plt.subplots(figsize=(10, 5))
line_pred, = ax.plot([], [], 'b', label='Predicted mean')
line_true, = ax.plot(X_test, y_true, 'g--', label='True function')
scatter_pts = ax.scatter([], [], c='k', s=20, label='Training points')
fill = None

def init():
    ax.set_xlim(0, 5)
    ax.set_ylim(-2, 2)
    return line_pred, scatter_pts

def update(frame):
    global fill
    if fill:
        fill.remove()

    y_pred = history_means[frame]
    y_std = history_stds[frame]
    line_pred.set_data(X_test.ravel(), y_pred)
    fill = ax.fill_between(X_test.ravel(),
                           np.array(y_pred) - 2 * np.array(y_std),
                           np.array(y_pred) + 2 * np.array(y_std),
                           alpha=0.2, color='blue')

    pts = np.array(used_points[:(frame * 2) + 1])  # 每2步存一次
    scatter_pts.set_offsets(pts)
    return line_pred, scatter_pts

ani = FuncAnimation(fig, update, frames=len(history_means),
                    init_func=init, blit=False, repeat=False, interval=300)

plt.title("Online GPR Animation (Pointwise Prediction)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()