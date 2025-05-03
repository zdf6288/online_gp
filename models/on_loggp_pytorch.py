import numpy as np
import torch
import torch.nn as nn
import random

class LoGGP_PyTorch(nn.Module):
    def __init__(self, x_dim, y_dim, max_data_per_expert=50, max_experts=4):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.max_data = max_data_per_expert
        self.max_experts = max_experts
        self.wo = 10

        self.X = np.zeros((x_dim, max_data_per_expert * max_experts), dtype=float)
        self.Y = np.zeros((y_dim, max_data_per_expert * max_experts), dtype=float)
        self.K = np.zeros((max_data_per_expert * y_dim, max_data_per_expert * max_experts), dtype=float)
        self.alpha = np.zeros((max_data_per_expert * y_dim, max_experts), dtype=float)

        self.count = 0
        self.localCount = np.zeros(2 * max_experts - 1, dtype=int)
        self.medians = np.zeros(2 * max_experts - 1)
        self.overlapD = np.zeros(2 * max_experts - 1, dtype=int)
        self.overlapW = np.zeros(2 * max_experts - 1)
        self.auxUbic = np.zeros(2 * max_experts - 1, dtype=int) - 1
        self.auxUbic[0] = 0
        self.children = np.zeros((2, 2 * max_experts - 1), dtype=int) - 1

        self.model_params = {}
        self.model_optimizers = {}

    def init_model_params(self, model_id):
        log_sigma_f = nn.Parameter(torch.zeros(self.y_dim))
        log_sigma_n = nn.Parameter(torch.full((self.y_dim,), -2.0))
        log_lengthscale = nn.Parameter(torch.zeros((self.x_dim, self.y_dim)))
        self.model_params[model_id] = {
            'log_sigma_f': log_sigma_f,
            'log_sigma_n': log_sigma_n,
            'log_lengthscale': log_lengthscale
        }
        optimizer = torch.optim.Adam([log_sigma_f, log_sigma_n, log_lengthscale], lr=0.0005)
        self.model_optimizers[model_id] = optimizer

    def kernel(self, Xi, Xj, out, model_id):
        params = self.model_params[model_id]
        sigma_f = torch.exp(params['log_sigma_f'][out])
        lengthscale = torch.exp(params['log_lengthscale'][:, out])
        if Xi.ndim == 1:
            sqdist = torch.sum(((Xi - Xj) / lengthscale) ** 2)
            return sigma_f**2 * torch.exp(-0.5 * sqdist)
        else:
            diff = (Xi.T - Xj) / lengthscale
            return sigma_f**2 * torch.exp(-0.5 * torch.sum(diff ** 2, axis=1))

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

    def update_point(self, x, y):
        model = 0
        while self.children[0, model] != -1:
            pL = self.activation(x, model)
            if pL >= random.random() and pL != 0:
                model = self.children[0, model]
            else:
                model = self.children[1, model]
        self.add_point(x, y, model)

    def add_point(self, x, y, model):
        print(f"Adding point to model {model}: {x}, {y}")
        if model not in self.model_params:
            self.init_model_params(model)

        pos = self.auxUbic[model]
        idx = self.localCount[model]
        if (pos * self.max_data + idx) >= self.X.shape[1]:
            print("Index overflow, skipping point.")
            return

        self.X[:, pos * self.max_data + idx] = x
        self.Y[:, pos * self.max_data + idx] = y
        self.localCount[model] += 1
        self.update_kernel(x, y, model)

        if self.localCount[model] == self.max_data:
            self.divide(model)

        self.optimize_model(model)

    def update_kernel(self, x, y, model):
        pos = self.auxUbic[model]
        l = self.localCount[model]
        if l > self.max_data:
            print("Exceeded max_data in model, skipping kernel update.")
            return
        for p in range(self.y_dim):
            sigma_n = torch.exp(self.model_params[model]['log_sigma_n'][p])
            kx = self.kernel(torch.tensor(x), torch.tensor(x), p, model).item()
            if l == 1:
                self.K[p * self.max_data, pos * self.max_data] = kx + sigma_n.item()**2
                self.alpha[p * self.max_data, pos] = self.Y[p, pos * self.max_data] / kx
            else:
                auxX = self.X[:, pos * self.max_data: pos * self.max_data + l]
                auxY = self.Y[p, pos * self.max_data: pos * self.max_data + l]
                b = self.kernel(torch.tensor(auxX), torch.tensor(x), p, model).detach().numpy()
                b[-1] += sigma_n.item()**2
                auxOut = p * self.max_data
                self.K[auxOut + l - 1, pos * self.max_data: pos * self.max_data + l - 1] = b[0:-1]
                self.K[auxOut: auxOut + l, pos * self.max_data + l - 1] = b
                self.alpha[auxOut: auxOut + l, pos] = np.linalg.solve(
                    self.K[auxOut: auxOut + l, pos * self.max_data: pos * self.max_data + l],
                    auxY
                )

    def divide(self, model):
        if self.auxUbic.max() + 1 >= self.max_experts:
            print("No room for more divisions")
            return

        pos = self.auxUbic[model]
        data = self.X[:, pos * self.max_data: pos * self.max_data + self.max_data]
        widths = data.max(axis=1) - data.min(axis=1)
        cutD = np.argmax(widths)
        width = widths[cutD]
        mP = (data[cutD].max() + data[cutD].min()) / 2.0
        o = width / self.wo if width > 0 else 0.1

        iL, iR = [], []
        for i in range(self.max_data):
            xD = data[cutD, i]
            if xD < mP - 0.5 * o:
                iL.append(i)
            elif xD > mP + 0.5 * o:
                iR.append(i)
            else:
                pL = 0.5 + (xD - mP) / o
                if random.random() < pL:
                    iL.append(i)
                else:
                    iR.append(i)

        newL = self.count + 1 if self.count > 0 else 1
        newR = newL + 1
        self.children[0, model] = newL
        self.children[1, model] = newR
        self.medians[model] = mP
        self.overlapD[model] = cutD
        self.overlapW[model] = o

        self.localCount[newL] = len(iL)
        self.localCount[newR] = len(iR)
        self.auxUbic[newL] = self.auxUbic[model]
        self.auxUbic[newR] = self.auxUbic.max() + 1
        self.auxUbic[model] = -1

        for leaf in (newL, newR):
            if leaf not in self.model_params:
                self.init_model_params(leaf)

        def move_data(indices, leaf):
            newX = self.X[:, pos * self.max_data + np.array(indices)]
            newY = self.Y[:, pos * self.max_data + np.array(indices)]
            new_pos = self.auxUbic[leaf]
            self.X[:, new_pos * self.max_data: new_pos * self.max_data + len(indices)] = newX
            self.Y[:, new_pos * self.max_data: new_pos * self.max_data + len(indices)] = newY
            for p in range(self.y_dim):
                K_full = self.K[p * self.max_data: (p + 1) * self.max_data,
                                pos * self.max_data: pos * self.max_data + self.max_data]
                newK = K_full[np.ix_(indices, indices)]
                self.K[p * self.max_data: p * self.max_data + len(indices),
                       new_pos * self.max_data: new_pos * self.max_data + len(indices)] = newK
                self.alpha[p * self.max_data: p * self.max_data + len(indices), new_pos] = \
                    np.linalg.solve(newK, self.Y[p, new_pos * self.max_data: new_pos * self.max_data + len(indices)])

        move_data(iL, newL)
        move_data(iR, newR)
        self.count = newR

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
                kval = self.kernel(torch.tensor(X_sub[:, :min_len]), torch.tensor(x), p, model).detach().numpy()
                pred = np.dot(kval, alpha_sub[:min_len])
                out[p] += pred * probs[i]

                # Variance approximation (optional): sum of k(x,x) - k.T @ K_inv @ k
                k_xx = self.kernel(torch.tensor(x), torch.tensor(x), p, model).item()
                var = k_xx - np.dot(kval[:min_len], kval[:min_len])  # Very rough
                out_var[p] += max(var, 1e-6) * probs[i]

        if return_std:
            return out, np.sqrt(out_var)
        return out

    
    def optimize_model(self, model_id, steps=3):
        pos = self.auxUbic[model_id]
        n = self.localCount[model_id]
        if n < 5:
            return
        x_data = self.X[:, pos * self.max_data: pos * self.max_data + n]
        y_data = self.Y[:, pos * self.max_data: pos * self.max_data + n]
        for p in range(self.y_dim):
            optimizer = self.model_optimizers[model_id]
            for _ in range(steps):
                optimizer.zero_grad()
                loss = self.nll_loss(x_data, y_data, model_id, p)
                loss.backward()
                optimizer.step()

    def nll_loss(self, x_data, y_data, model_id, output_dim):
        params = self.model_params[model_id]
        sigma_f = torch.exp(params['log_sigma_f'][output_dim])
        sigma_n = torch.exp(params['log_sigma_n'][output_dim])
        lengthscale = torch.exp(params['log_lengthscale'][:, output_dim])
        X = torch.tensor(x_data.T, dtype=torch.float32)
        Y = torch.tensor(y_data[output_dim], dtype=torch.float32).unsqueeze(-1)
        dists = ((X[:, None, :] - X[None, :, :]) / lengthscale).pow(2).sum(-1)
        K = sigma_f**2 * torch.exp(-0.5 * dists)
        K += sigma_n**2 * torch.eye(K.size(0))
        try:
            L = torch.linalg.cholesky(K)
            alpha = torch.cholesky_solve(Y, L)
            log_det = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
            nll = 0.5 * Y.T @ alpha + 0.5 * log_det + 0.5 * X.shape[0] * torch.log(torch.tensor(2 * np.pi))
            return nll.squeeze()
        except RuntimeError:
            return torch.tensor(1e6)