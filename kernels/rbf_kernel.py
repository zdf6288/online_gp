import torch

class RBFKernel:
    def __init__(self, lengthscale=1.0, variance=1.0):
        self.lengthscale = torch.tensor(lengthscale, dtype=torch.float64)
        self.variance = torch.tensor(variance, dtype=torch.float64)

    def __call__(self, X1, X2):
        device = X1.device
        lengthscale = self.lengthscale.to(device)
        variance = self.variance.to(device)

        X1 = X1 / lengthscale
        X2 = X2 / lengthscale
        sqdist = torch.cdist(X1, X2).pow(2)
        return variance * torch.exp(-0.5 * sqdist)
