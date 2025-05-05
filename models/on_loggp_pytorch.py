import numpy as np
import torch
import torch.nn as nn
import random

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kernels.rbf_kernel import RBFKernel

# class RBFKernel:
#     def __init__(self, lengthscale=1.0, variance=1.0):
#         self.lengthscale = torch.tensor(lengthscale, dtype=torch.float64)
#         self.variance = torch.tensor(variance, dtype=torch.float64)

#     def __call__(self, X1, X2):
#         device = X1.device
#         lengthscale = self.lengthscale.to(device)
#         variance = self.variance.to(device)

#         X1 = X1 / lengthscale
#         X2 = X2 / lengthscale
#         sqdist = torch.cdist(X1, X2).pow(2)
#         return variance * torch.exp(-0.5 * sqdist)


def to_tensor_2d(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor):
        raise TypeError("Input must be a numpy.ndarray or torch.Tensor")
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D input, got shape {x.shape}")
    return x.double().to(device)


class LoGGP_PyTorch(nn.Module):
    def __init__(self, x_dim, y_dim, max_data_per_expert=50, max_experts=4, lr=0.0005, steps=10, device=None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.max_data = max_data_per_expert
        self.max_experts = max_experts
        self.wo = 300

        self.X = np.zeros((x_dim, max_data_per_expert * max_experts), dtype=float)
        self.Y = np.zeros((y_dim, max_data_per_expert * max_experts), dtype=float)
        self.K = np.zeros((max_data_per_expert * y_dim, max_data_per_expert * max_experts), dtype=float)
        self.alpha = np.zeros((max_data_per_expert * y_dim, max_experts), dtype=float)

        self.count = 0 # Number of models
        self.localCount = np.zeros(2 * max_experts - 1, dtype=int)
        self.medians = np.zeros(2 * max_experts - 1) # Median of the split
        self.overlapD = np.zeros(2 * max_experts - 1, dtype=int) # Dimension of the split
        self.overlapW = np.zeros(2 * max_experts - 1) # Width of the split
        self.auxUbic = np.zeros(2 * max_experts - 1, dtype=int) - 1 # Auxiliary index for the model
        self.auxUbic[0] = 0 # Root node
        self.children = np.zeros((2, 2 * max_experts - 1), dtype=int) - 1 # Children of the model

        self.model_params = {}
        self.model_optimizers = {}

        self.lr = lr
        self.optimize_steps = steps

    def init_model_params(self, model_id):
        log_sigma_f = nn.Parameter(torch.randn(self.y_dim, device=self.device) * 0.1)
        log_sigma_n = nn.Parameter(torch.randn(self.y_dim, device=self.device) * 0.1 - 2.0)
        log_lengthscale = nn.Parameter(torch.randn((self.x_dim, self.y_dim), device=self.device) * 0.1)
        self.model_params[model_id] = {
            'log_sigma_f': log_sigma_f,
            'log_sigma_n': log_sigma_n,
            'log_lengthscale': log_lengthscale
        }
        optimizer = torch.optim.Adam([log_sigma_f, log_sigma_n, log_lengthscale], lr=self.lr)
        self.model_optimizers[model_id] = optimizer

    def kernel(self, Xi, Xj, out, model_id):
        params = self.model_params[model_id]
        variance = torch.exp(params['log_sigma_f'][out]) ** 2
        lengthscale = torch.exp(params['log_lengthscale'][:, out]).detach().cpu().numpy()
        rbf = RBFKernel(lengthscale=lengthscale, variance=variance.item())
        Xi = to_tensor_2d(Xi, self.device)
        Xj = to_tensor_2d(Xj, self.device)
        return rbf(Xi, Xj).squeeze()

    # Activation function for the split
    def activation(self, x, model):
        if self.children[0, model] == -1:
            return 0
        mP = self.medians[model]
        xD = x[self.overlapD[model]]
        o = self.overlapW[model]
        if xD < mP - 0.5 * o:
            return 1
        elif mP - 0.5 * o <= xD <= mP + 0.5 * o:
            return 0.5 + (xD - mP) / o
        else:
            return 0

    # def update_point(self, x, y, optimize=True):
    #     model = 0
    #     while self.children[0, model] != -1:
    #         pL = self.activation(x, model)
    #         # Randomly choose the child node based on the activation probability
    #         if pL >= random.random() and pL != 0:
    #             model = self.children[0, model]
    #         else:
    #             model = self.children[1, model]
    #     self.add_point(x, y, model, optimize)
    
    def update_point(self, x, y, step_id=None, optimize=True):
        model = 0
        while self.children[0, model] != -1:
            pL = self.activation(x, model)
            if pL >= random.random() and pL != 0:
                model = self.children[0, model]
            else:
                model = self.children[1, model]
        self.add_point(x, y, model, step_id=step_id, optimize=optimize)
        
    def entropy_based_replace(self, x, y, model):
        """
        替换当前专家中熵最小（即信息最少）的样本点为新样本。
        用 GP 的后验方差作为熵近似指标。
        """
        pos = self.auxUbic[model]
        n = self.max_data
        min_entropy = float('inf')
        replace_idx = -1

        for i in range(n):
            x_i = self.X[:, pos * self.max_data + i].reshape(-1, 1)
            try:
                # 计算 k(x_i, x_i)
                k_xx = self.kernel(x_i.T, x_i.T, 0, model).item()

                # 计算 k(X, x_i)
                X_all = self.X[:, pos * self.max_data: pos * self.max_data + n]
                kval = self.kernel(X_all.T, x_i.T, 0, model).cpu().numpy()

                # 方差估计（避免负数）
                var = k_xx - np.dot(kval, kval.T).item()
                var = max(var, 1e-6)
            except Exception:
                var = 1e6  # 错误时视为最大信息

            if var < min_entropy:
                min_entropy = var
                replace_idx = i

        # 执行替换
        self.X[:, pos * self.max_data + replace_idx] = x
        self.Y[:, pos * self.max_data + replace_idx] = y
        self.update_kernel(x, y, model)

    def add_point(self, x, y, model, step_id=None, optimize=True, min_points_to_optimize=10, optimize_every=20):
        if model not in self.model_params:
            self.init_model_params(model)
        pos = self.auxUbic[model]
        idx = self.localCount[model]
        if idx >= self.max_data and self.auxUbic[-1] != -1:
            # print("replacing point")
            self.entropy_based_replace(x, y, model)
            return

        if idx >= self.max_data:
            # 点数已满：替换冗余点
            X_sub = self.X[:, pos * self.max_data: pos * self.max_data + self.max_data]
            kmat = self.kernel(X_sub.T, X_sub.T, 0, model).detach().cpu().numpy()
            sim = kmat.sum(axis=1) - np.diag(kmat)
            remove_idx = np.argmax(sim)
            self.X[:, pos * self.max_data + remove_idx] = x
            self.Y[:, pos * self.max_data + remove_idx] = y
            self.update_kernel(x, y, model)
        else:
            self.X[:, pos * self.max_data + idx] = x
            self.Y[:, pos * self.max_data + idx] = y
            self.localCount[model] += 1
            self.update_kernel(x, y, model)
            if self.localCount[model] == self.max_data:
                self.divide(model)

        # ✅ 控制优化策略
        if (
            optimize and
            self.localCount[model] >= min_points_to_optimize and
            step_id is not None and
            step_id % optimize_every == 0
        ):
            self.optimize_model(model)

    def update_kernel(self, x, y, model):
        pos = self.auxUbic[model]
        l = self.localCount[model]
        if l > self.max_data:
            return
        for p in range(self.y_dim):
            sigma_n = torch.exp(self.model_params[model]['log_sigma_n'][p])
            kx = self.kernel(x, x, p, model).item()
            if l == 1:
                self.K[p * self.max_data, pos * self.max_data] = kx + sigma_n.item()**2 + 1e-6
                self.alpha[p * self.max_data, pos] = self.Y[p, pos * self.max_data] / kx
            else:
                auxX = self.X[:, pos * self.max_data: pos * self.max_data + l]
                auxY = self.Y[p, pos * self.max_data: pos * self.max_data + l]
                b = self.kernel(auxX.T, x, p, model).detach().cpu().numpy()
                b[-1] += sigma_n.item()**2 + 1e-6
                auxOut = p * self.max_data
                self.K[auxOut + l - 1, pos * self.max_data: pos * self.max_data + l - 1] = b[0:-1]
                self.K[auxOut: auxOut + l, pos * self.max_data + l - 1] = b
                K_sub = self.K[auxOut: auxOut + l, pos * self.max_data: pos * self.max_data + l]
                K_sub += np.eye(l) * 1e-6
                self.alpha[auxOut: auxOut + l, pos] = np.linalg.solve(K_sub, auxY)

    def divide(self, model):
            if self.auxUbic[-1] != -1:
                # print("no room for more divisions")
                return
            # compute widths in all dimensions
            width = self.X[:, self.auxUbic[model] * self.max_data: self.auxUbic[model] *
                                                            self.max_data + self.max_data].max(axis=1) - self.X[:,
                                                                                                self.auxUbic[model] *
                                                                                                self.max_data: self.auxUbic[
                                                                                                            model] * self.max_data + self.max_data].min(
                axis=1)

            # obtain cutting dimension
            cutD = np.argmax(width)
            width = width.max()
            # compute hyperplane
            mP = (self.X[cutD, self.auxUbic[model] * self.max_data: self.auxUbic[model] *
                                                            self.max_data + self.max_data].max() + self.X[cutD,
                                                                                            self.auxUbic[model] *
                                                                                            self.max_data: self.auxUbic[
                                                                                                        model] * self.max_data + self.max_data].min()) / 2

            # get overlapping region
            o = width / self.wo
            if o == 0:
                o = 0.1

            self.medians[model] = mP  # set model hyperplane
            self.overlapD[model] = cutD  # cut dimension
            self.overlapW[model] = o  # width of overlap

            xL = np.zeros([self.x_dim, self.max_data], dtype=float)
            xR = np.zeros([self.x_dim, self.max_data], dtype=float)
            yL = np.zeros([self.y_dim, self.max_data], dtype=float)
            yR = np.zeros([self.y_dim, self.max_data], dtype=float)

            lcount = 0
            rcount = 0

            iL = np.zeros(self.max_data, dtype=int)
            iR = np.zeros(self.max_data, dtype=int)

            for i in range(self.max_data):
                xD = self.X[cutD, self.auxUbic[model] * self.max_data + i]
                if xD < mP - 0.5 * o:
                    xL[:, lcount] = self.X[:, self.auxUbic[model] * self.max_data + i]
                    yL[:, lcount] = self.Y[:, self.auxUbic[model] * self.max_data + i]
                    iL[lcount] = i
                    lcount += 1
                elif xD >= mP - 0.5 * o and xD <= mP + 0.5 * o:  # if in overlapping
                    pL = 0.5 + (xD - mP) / o  # prob. of being in left
                    if pL >= random.random() and pL != 0:  # left selected
                        xL[:, lcount] = self.X[:, self.auxUbic[model] * self.max_data + i]
                        yL[:, lcount] = self.Y[:, self.auxUbic[model] * self.max_data + i]
                        iL[lcount] = i
                        lcount += 1
                    else:
                        xR[:, rcount] = self.X[:, self.auxUbic[model] * self.max_data + i]
                        yR[:, rcount] = self.Y[:, self.auxUbic[model] * self.max_data + i]
                        iR[rcount] = i
                        rcount += 1
                elif xD > mP + 0.5 * o:  # if in right
                    xR[:, rcount] = self.X[:, self.auxUbic[model] * self.max_data + i]
                    yR[:, rcount] = self.Y[:, self.auxUbic[model] * self.max_data + i]
                    iR[rcount] = i
                    rcount += 1
            self.localCount[model] = 0
            # update counter
            if self.count == 0:
                self.count += 1
            else:
                self.count += 2
            # assign children
            self.children[0, model] = self.count
            self.children[1, model] = self.count + 1
            # set local count of children
            self.localCount[self.count] = lcount
            self.localCount[self.count + 1] = rcount
            self.auxUbic[self.count] = self.auxUbic[model]
            self.auxUbic[self.count + 1] = self.auxUbic.max() + 1
            # values for K permutation
            order = np.concatenate((iL[0:lcount], iR[0:rcount]))
            # update parameters of child models
            for p in range(self.y_dim):
                newK = self.K[p * self.max_data: (p + 1) * self.max_data, self.auxUbic[model] * self.max_data:
                                                                self.auxUbic[model] * self.max_data + self.max_data]
                # permute K
                newK[range(self.max_data), :] = newK[order, :]
                newK[:, range(self.max_data)] = newK[:, order]
                # set child K
                self.K[p * self.max_data: p * self.max_data + lcount, self.auxUbic[self.count] * self.max_data:
                                                            self.auxUbic[self.count] * self.max_data + lcount] = newK[0: lcount,
                                                                                                            0: lcount]
                self.K[p * self.max_data: p * self.max_data + rcount, self.auxUbic[self.count + 1] * self.max_data:
                                                            self.auxUbic[self.count + 1] * self.max_data + rcount] = \
                    newK[lcount: self.max_data, lcount: self.max_data]
                # set child alpha
                self.alpha[p * self.max_data: p * self.max_data + lcount, self.auxUbic[self.count]] = \
                    np.linalg.solve(newK[0: lcount, 0: lcount], yL[p, 0:lcount].transpose())
                self.alpha[p * self.max_data: p * self.max_data + rcount, self.auxUbic[self.count + 1]] = \
                    np.linalg.solve(newK[lcount: self.max_data, lcount: self.max_data], yR[p, 0:rcount].transpose())
            # parent will not have more data:
            self.auxUbic[model] = -1
            # relocate X Y to children
            self.X[:, self.auxUbic[self.count] * self.max_data:
                    self.auxUbic[self.count] * self.max_data + self.max_data] = xL
            self.X[:, self.auxUbic[self.count + 1] * self.max_data:
                    self.auxUbic[self.count + 1] * self.max_data + self.max_data] = xR
            self.Y[:, self.auxUbic[self.count] * self.max_data:
                    self.auxUbic[self.count] * self.max_data + self.max_data] = yL
            self.Y[:, self.auxUbic[self.count + 1] * self.max_data:
                    self.auxUbic[self.count + 1] * self.max_data + self.max_data] = yR
            
            self.init_model_params(self.count)       # 初始化左子模型
            self.init_model_params(self.count + 1)   # 初始化右子模型

    def predict(self, x, return_std=True):
        x = np.array(x)
        models = np.zeros(1000, dtype=int)
        probs = np.ones(1000)
        mCount = 1
        while self.children[0, models[:mCount]].sum() != -1 * mCount:
            for j in range(mCount):
                if self.children[0, models[j]] != -1:
                    pL = self.activation(x, models[j])
                    if pL == 1:
                        models[j] = self.children[0, models[j]]
                    elif pL == 0:
                        models[j] = self.children[1, models[j]]
                    elif 1 > pL > 0:
                        models[mCount] = self.children[1, models[j]]
                        probs[mCount] = probs[j] * (1 - pL)
                        models[j] = self.children[0, models[j]]
                        probs[j] *= pL
                        mCount += 1

        out = np.zeros(self.y_dim)
        out_var = np.zeros(self.y_dim)
        for p in range(self.y_dim):
            for i in range(mCount):
                model = models[i]
                pos = self.auxUbic[model]
                n_data = self.localCount[model]
                if n_data == 0:
                    continue
                X_sub = self.X[:, pos * self.max_data: pos * self.max_data + n_data]
                alpha_sub = self.alpha[p * self.max_data: p * self.max_data + n_data, pos]
                min_len = min(X_sub.shape[1], alpha_sub.shape[0])
                kval = np.atleast_1d(self.kernel(X_sub[:, :min_len].T, x, p, model).detach().cpu().numpy())
                pred = np.dot(kval, alpha_sub[:min_len])
                out[p] += pred * probs[i]
                k_xx = self.kernel(x, x, p, model).item()
                var = k_xx - np.dot(kval[:min_len], kval[:min_len])
                out_var[p] += max(var, 1e-6) * probs[i]
        return (out, np.sqrt(out_var)) if return_std else out

    def optimize_model(self, model_id):
        pos = self.auxUbic[model_id]
        n = self.localCount[model_id]
        if n < 5:
            return
        x_data = self.X[:, pos * self.max_data: pos * self.max_data + n]
        y_data = self.Y[:, pos * self.max_data: pos * self.max_data + n]
        for p in range(self.y_dim):
            optimizer = self.model_optimizers[model_id]
            for _ in range(self.optimize_steps):
                optimizer.zero_grad()
                loss = self.nll_loss(x_data, y_data, model_id, p)
                loss.backward()
                optimizer.step()

    def nll_loss(self, x_data, y_data, model_id, output_dim):
        params = self.model_params[model_id]
        sigma_f = torch.exp(params['log_sigma_f'][output_dim])
        sigma_n = torch.exp(params['log_sigma_n'][output_dim])
        lengthscale = torch.exp(params['log_lengthscale'][:, output_dim])
        X = torch.tensor(x_data.T, dtype=torch.float64, device=self.device)
        Y = torch.tensor(y_data[output_dim], dtype=torch.float64, device=self.device).unsqueeze(-1)
        dists = ((X[:, None, :] - X[None, :, :]) / lengthscale).pow(2).sum(-1)
        K = sigma_f**2 * torch.exp(-0.5 * dists)
        jitter = 1e-6
        K += (sigma_n**2 + jitter) * torch.eye(K.size(0), dtype=torch.float64, device=self.device)
        try:
            L = torch.linalg.cholesky(K)
        except RuntimeError:
            K += 1e-4 * torch.eye(K.size(0), dtype=torch.float64, device=self.device)
            try:
                L = torch.linalg.cholesky(K)
            except RuntimeError:
                return torch.tensor(1e6, device=self.device)
        alpha = torch.cholesky_solve(Y, L)
        log_det = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
        nll = 0.5 * Y.T @ alpha + 0.5 * log_det + 0.5 * X.shape[0] * torch.log(torch.tensor(2 * np.pi))
        return nll.squeeze()