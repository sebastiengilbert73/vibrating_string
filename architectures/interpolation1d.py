import torch
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class BatchEvaluable1D(ABC):
    @abstractmethod
    def batch_evaluate(self, x_tsr):
        pass

class GaussianRBF1D(BatchEvaluable1D):
    def __init__(self, xy_tsr, sigma=None, reg=1.0e-8, device='cuda', minimum_x_gap=1e-6):  # xy_tsr.shape = (N, 2)
        if len(xy_tsr.shape) != 2:
            raise ValueError(f"GaussianRBF1D.__init__(): len(xy_tsr.shape) ({len(xy_tsr.shape)}) != 2")
        if xy_tsr.shape[1] != 2:
            raise ValueError(f"GaussianRBF1D.__init__(): xy_tsr.shape[1] ({xy_tsr.shape[1]}) != 2")
        if xy_tsr.shape[0] < 2:
            raise ValueError(f"GaussianRBF1D.__init__(): The number of points ({xy_tsr.shape[0]}) < 2")
        # Sort the by x's
        sorted_indices = torch.argsort(xy_tsr[:, 0])
        self.sorted_xy_tsr = xy_tsr[sorted_indices].to(device)  # (N, 2)

        # Check that all x's are sufficiently different
        delta_xs = torch.zeros(self.sorted_xy_tsr.shape[0] - 1)
        for row in range(1, self.sorted_xy_tsr.shape[0]):
            gap = self.sorted_xy_tsr[row, 0] - self.sorted_xy_tsr[row - 1, 0]
            if gap < minimum_x_gap:
                raise ValueError(f"GaussianRBF1D.__init__(): The minimum gap ({minimum_x_gap}) is not respected between x = {self.sorted_xy_tsr[row - 1, 0]} and x = {self.sorted_xy_tsr[row, 0]}")
            delta_xs[row - 1] = gap

        # Sigma, the standard deviation of the gaussians
        if sigma is None:
            self.sigma = torch.median(delta_xs).to(device)
        else:
            self.sigma = torch.tensor(sigma).to(device)

        N = self.sorted_xy_tsr.shape[0]
        # y = a x + b + sum_i [ c_i exp(-(x - mu_i)^2/(2 * sigma^2)) ]
        self.centers = self.sorted_xy_tsr[1: -1, 0].to(device)  # (N - 2,)

        D = torch.zeros((N, N))  # (N, N)
        e = torch.zeros(N)  # (N,)
        for row_ndx in range(N):
            x = self.sorted_xy_tsr[row_ndx, 0]
            y = self.sorted_xy_tsr[row_ndx, 1]
            D[row_ndx, 0] = x
            D[row_ndx, 1] = 1
            for center_ndx in range(len(self.centers)):
                D[row_ndx, center_ndx + 2] = torch.exp(-(x - self.centers[center_ndx])**2/(2 * self.sigma**2))
            e[row_ndx] = y
        # Ridge formulation: (DtD + lambda I) x = Dt e
        DtD = D.T @ D
        reg_term = reg * torch.eye(DtD.shape[0])
        lhs = DtD + reg_term
        rhs = D.T @ e
        abc = torch.linalg.solve(lhs, rhs)  # (N,)
        self.a = abc[0].to(device)
        self.b = abc[1].to(device)
        self.c = abc[2:].to(device)  # (N - 2,)

    def batch_evaluate(self, x_tsr):  # x_tsr.shape = (N, 1)
        if len(x_tsr.shape) != 2:
            raise ValueError(f"GaussianRBF1D.batch_evaluate(): len(x_tsr.shape) ({len(x_tsr.shape)}) != 2")
        y = torch.zeros_like(x_tsr)  # (N, 1)
        y += self.a * x_tsr + self.b + torch.sum( self.c * torch.exp(-(x_tsr - self.centers)**2/(2 * self.sigma**2)), dim=1).unsqueeze(1)
        return y


# Example d'utilisation
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    device = "cuda"

    xy_tsr = torch.tensor([[0, 0.0], [0.7, 1.0], [0.35, 0.5], [0.15, 1.2], [1.0, 0.8]], dtype=torch.float64, device=device)
    #xy_tsr = torch.tensor([[0, 0.0], [0.5, 1.0], [1., 0.]], dtype=torch.float64, device=device)
    rbf = GaussianRBF1D(xy_tsr, sigma=None, reg=1e-8, device=device, minimum_x_gap=1e-6)

    # Test batch_evaluate()
    q = torch.tensor([0.2, 0.7, 0.4], dtype=torch.float64, device=device)
    print(f"q.device = {q.device}; q.unsqueeze(1).device = {q.unsqueeze(1).device}")
    zq = rbf.batch_evaluate(q.unsqueeze(1))
    print("z(q):", zq)

    # Display the collocation points and the interpolation
    xs = np.linspace(0, 1, 101)
    ys = rbf.batch_evaluate(torch.from_numpy(xs).unsqueeze(1).to(device) )

    fig, ax = plt.subplots()
    ax.scatter(xy_tsr[:, 0].cpu(), xy_tsr[:, 1].cpu(), color='blue', marker='o', label='Collocation points')
    ax.plot(xs, ys.cpu(), label='Interpolation', color='red')
    ax.legend()
    plt.show()