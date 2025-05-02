import torch

class RBFKernel:
    def __init__(self, lengthscale=1.0, variance=1.0):
        self.lengthscale = torch.tensor(lengthscale, dtype=torch.float64)
        self.variance = torch.tensor(variance, dtype=torch.float64)

    def __call__(self, X1, X2):
        """
        Compute RBF kernel matrix between X1 and X2
        :param X1: tensor of shape [n1, d]
        :param X2: tensor of shape [n2, d]
        :return: tensor of shape [n1, n2]
        """
        X1 = X1 / self.lengthscale
        X2 = X2 / self.lengthscale
        sqdist = torch.cdist(X1, X2).pow(2)
        return self.variance * torch.exp(-0.5 * sqdist)

    def set_params(self, lengthscale=None, variance=None):
        if lengthscale is not None:
            self.lengthscale = torch.tensor(lengthscale, dtype=torch.float64)
        if variance is not None:
            self.variance = torch.tensor(variance, dtype=torch.float64)