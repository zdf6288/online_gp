import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kernels.rbf_kernel import RBFKernel
from tools.to_tensor_2d import to_tensor_2d


class LoGGP_PyTorch(nn.Module):
    def __init__(self, x_dim, y_dim, max_data_per_expert=50, max_experts=4, lr=0.0005, steps=10, min_points_to_optimize=10, optimize_every=10, device=None, enable_variance_cache=False):
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
        self.min_points_to_optimize = min_points_to_optimize
        self.optimize_every = optimize_every
        self.update_counters = {}  # 每个模型独立的更新计数器
        
        self.enable_variance_cache = enable_variance_cache
        self.k_diag = np.full((y_dim * max_data_per_expert, max_experts), np.nan)  # 初始化为 NaN，表示未缓存
    

        self.data_index_queue = [deque() for _ in range(2 * max_experts - 1)]

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
        self.update_counters[model_id] = 0
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
        
    # Update the model with new point    
    def update_point(self, x, y, step_id=None, optimize=True):
        # Step 1: predict first
        pred, std = self.predict(x, return_std=True)

        # Step 2: find target expert node
        model = 0
        while self.children[0, model] != -1:
            pL = self.activation(x, model)
            if pL >= random.random() and pL != 0:
                model = self.children[0, model]
            else:
                model = self.children[1, model]

        # Step 3: update the model with new point
        self.add_point(x, y, model, step_id=step_id, optimize=optimize)
        return pred, std

        
    def add_point(self, x, y, model, step_id=None, optimize=True):
        if model not in self.model_params:
            self.init_model_params(model)
        pos = self.auxUbic[model]
        idx = self.localCount[model]

        if idx < self.max_data:
            # 1. Expert not full → directly add
            write_idx = idx
            self.X[:, pos * self.max_data + idx] = x
            self.Y[:, pos * self.max_data + idx] = y
            self.localCount[model] += 1
            self.data_index_queue[model].append(write_idx)
            self.update_kernel(x, y, model)
        
        if idx == self.max_data:
            if self.auxUbic[-1] != -1:
                self.entropy_based_replace(x, y, model)
            
                # return
                # replace_idx = self.data_index_queue[model].popleft()
                # self.X[:, pos * self.max_data + replace_idx] = x
                # self.Y[:, pos * self.max_data + replace_idx] = y
                # self.update_kernel(x, y, model)
                # self.data_index_queue[model].append(replace_idx)              
            else:
                #Full expert → divide
                self.divide(model)

        # 2. Update the model parameters
        if optimize and self.localCount[model] >= self.min_points_to_optimize:
            self.update_counters[model] += 1
            if self.update_counters[model] % self.optimize_every == 0:
                self.optimize_model(model)

        
    def entropy_based_replace(self, x, y, model):
        """
        Replace the least informative point in the expert using entropy (variance) as proxy.
        Now with k(xi, xi) cached to avoid recomputation.
        """
        pos = self.auxUbic[model]
        n = self.max_data
        min_entropy = float('inf')
        replace_idx = -1

        X_all = self.X[:, pos * self.max_data: pos * self.max_data + n]
        
        for i in range(n):
            x_i = self.X[:, pos * self.max_data + i].reshape(-1, 1)

            if self.enable_variance_cache:
                var = self.k_diag[0 * self.max_data + i, pos]
                if not np.isnan(var):
                    # 缓存命中，直接使用
                    pass
                else:
                    # 重新计算并写入缓存
                    k_xx = self.kernel(x_i.T, x_i.T, 0, model).item()
                    X_all = self.X[:, pos * self.max_data: pos * self.max_data + n]
                    kval = self.kernel(X_all.T, x_i.T, 0, model).cpu().numpy()
                    var = k_xx - np.dot(kval, kval.T).item()
                    var = max(var, 1e-6)
                    self.k_diag[0 * self.max_data + i, pos] = var
            else:
                # 不使用缓存，直接计算
                k_xx = self.kernel(x_i.T, x_i.T, 0, model).item()
                X_all = self.X[:, pos * self.max_data: pos * self.max_data + n]
                kval = self.kernel(X_all.T, x_i.T, 0, model).cpu().numpy()
                var = k_xx - np.dot(kval, kval.T).item()
                var = max(var, 1e-6)

            if var < min_entropy:
                min_entropy = var
                replace_idx = i

        # Replace data and update kernel
        self.X[:, pos * self.max_data + replace_idx] = x
        self.Y[:, pos * self.max_data + replace_idx] = y
        self.update_kernel(x, y, model)


    def update_kernel(self, x, y, model):
        pos = self.auxUbic[model]
        idx = self.localCount[model]
        if idx > self.max_data:
            return
        for p in range(self.y_dim):
            sigma_n = torch.exp(self.model_params[model]['log_sigma_n'][p])
            kx = self.kernel(x, x, p, model).item()
            self.k_diag[p * self.max_data + idx - 1, pos] = kx  # ✅ 缓存 k(xi, xi)
            if idx == 1:
                self.K[p * self.max_data, pos * self.max_data] = kx + sigma_n.item()**2
                # self.K[p * self.max_data, pos * self.max_data] = kx + sigma_n.item()**2 + 1e-6
                self.alpha[p * self.max_data, pos] = self.Y[p, pos * self.max_data] / kx
            else:
                auxX = self.X[:, pos * self.max_data: pos * self.max_data + idx]
                auxY = self.Y[p, pos * self.max_data: pos * self.max_data + idx]
                b = self.kernel(auxX.T, x, p, model).detach().cpu().numpy()
                b[-1] += sigma_n.item()**2
                # b[-1] += sigma_n.item()**2 + 1e-6
                auxOut = p * self.max_data
                self.K[auxOut + idx - 1,
                       pos * self.max_data: pos * self.max_data + idx - 1] = b[0:-1]
                self.K[auxOut: auxOut + idx, 
                       pos * self.max_data + idx - 1] = b
                K_sub = self.K[auxOut: auxOut + idx, 
                               pos * self.max_data: pos * self.max_data + idx]
                # K_sub += np.eye(idx) * 1e-6
                self.alpha[auxOut: auxOut + idx, pos] = np.linalg.solve(K_sub, auxY)
            
            # 缓存更新
            if self.enable_variance_cache:
                for j in range(idx):
                    xj = self.X[:, pos * self.max_data + j].reshape(-1, 1)
                    try:
                        k_xx = self.kernel(xj.T, xj.T, p, model).item()
                        X_all = self.X[:, pos * self.max_data: pos * self.max_data + idx]
                        kval = self.kernel(X_all.T, xj.T, p, model).cpu().numpy()
                        var = k_xx - np.dot(kval.T, kval).item()
                        var = max(var, 1e-6)
                    except:
                        var = 1e6
                    self.k_diag[p * self.max_data + j, pos] = var

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