import torch
import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kernels.rbf_kernel import RBFKernel

# === Online GPR 模型 ===
class OnlineGPTorch(torch.nn.Module):
    def __init__(self, max_points=30, lr=0.005, optimize_steps=10):
        super().__init__()
        self.X = []
        self.y = []
        self.max_points = max_points
        self.lr = lr
        self.optimize_steps = optimize_steps

        self.log_lengthscale = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.log_variance = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.log_noise = torch.nn.Parameter(torch.tensor(-2.0, dtype=torch.float64))
        self.kernel = RBFKernel()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

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
        if len(self.X) >= 2:
            for _ in range(self.optimize_steps):
                self.optimizer.zero_grad()
                loss = -self.marginal_log_likelihood()
                loss.backward()
                # print(f"Loss: {loss.item()}")
                self.optimizer.step()

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
