import torch
import numpy as np
import architectures.pinn1d as pinn1d
import architectures.interpolation1d as interpolation1d
from typing import List

class HardConstrained1dResNet(torch.nn.Module):
    def __init__(self,
                 boundary_bubble: interpolation1d.BatchEvaluable1D,
                 number_of_blocks: int,
                 block_width: int,
                 number_of_outputs: int,
                 initial_profile_xuv: np.ndarray,
                 sigma: float,
                 rbf_reg: float=0.0001,
                 time_bubble_sigma: float=0.3,
                 device='cuda',
                 minimum_x_gap=1e-6):
        super().__init__()
        self.boundary_bubble = boundary_bubble
        self.z_predictor = pinn1d.ResidualNet(
            number_of_inputs=2,
            number_of_blocks=number_of_blocks,
            block_width=block_width,
            number_of_outputs=number_of_outputs
        ).to(device)
        self.initial_profile_xuv = initial_profile_xuv

        self.initial_position_interpolator = interpolation1d.GaussianRBF1D(
            xy_tsr=self.initial_profile_xuv[:, 0: 2],
            sigma=sigma,
            reg=rbf_reg,
            device=device,
            minimum_x_gap=minimum_x_gap
        )
        self.initial_velocity_interpolator = interpolation1d.GaussianRBF1D(
            xy_tsr=self.initial_profile_xuv[:, [0, 2]],
            sigma=sigma,
            reg=rbf_reg,
            device=device,
            minimum_x_gap=minimum_x_gap
        )

        self.time_bubble_sigma = time_bubble_sigma

    def forward(self, x_t):  # x_t.shape = (B, 2)
        x_tsr = x_t[:, 0].unsqueeze(1)  # (B, 1)
        t_tsr = x_t[:, 1].unsqueeze(1)  # (B, 1)
        bubble_t_tsr = self.time_bubble(t_tsr)  # (B, 1)
        bubble_x_tsr = self.boundary_bubble.batch_evaluate(x_tsr)  # (B, 1)
        z_tsr = self.z_predictor(x_t)  # (B, 1)
        initial_interpolation_tsr = self.initial_position_interpolator.batch_evaluate(x_tsr) + self.initial_velocity_interpolator.batch_evaluate(x_tsr) * t_tsr  # (B, 1)
        return initial_interpolation_tsr + bubble_t_tsr * bubble_x_tsr * z_tsr  # (B, 1)

    def time_bubble(self, t_tsr):  # t_tsr.shape = (B, 1)
        return 1.0 - torch.exp(-torch.pow(t_tsr, 2)/(2 * self.time_bubble_sigma**2))  # (B, 1)

class HardConstrained1dMLP(torch.nn.Module):
    def __init__(self,
                 boundary_bubble: interpolation1d.BatchEvaluable1D,
                 layer_widths: List[int],
                 number_of_outputs: int,
                 initial_profile_xuv: np.ndarray,
                 sigma: float,
                 rbf_reg: float=0.0001,
                 time_bubble_sigma: float=0.3,
                 device='cuda',
                 minimum_x_gap=1e-6):
        super().__init__()
        self.boundary_bubble = boundary_bubble
        self.z_predictor = pinn1d.MLP(
            number_of_inputs=2,
            layer_widths=layer_widths,
            number_of_outputs=number_of_outputs
        ).to(device)
        self.initial_profile_xuv = initial_profile_xuv

        self.initial_position_interpolator = interpolation1d.GaussianRBF1D(
            xy_tsr=self.initial_profile_xuv[:, 0: 2],
            sigma=sigma,
            reg=rbf_reg,
            device=device,
            minimum_x_gap=minimum_x_gap
        )
        self.initial_velocity_interpolator = interpolation1d.GaussianRBF1D(
            xy_tsr=self.initial_profile_xuv[:, [0, 2]],
            sigma=sigma,
            reg=rbf_reg,
            device=device,
            minimum_x_gap=minimum_x_gap
        )

        self.time_bubble_sigma = time_bubble_sigma

    def forward(self, x_t):  # x_t.shape = (B, 2)
        x_tsr = x_t[:, 0].unsqueeze(1)  # (B, 1)
        t_tsr = x_t[:, 1].unsqueeze(1)  # (B, 1)
        bubble_t_tsr = self.time_bubble(t_tsr)  # (B, 1)
        bubble_x_tsr = self.boundary_bubble.batch_evaluate(x_tsr)  # (B, 1)
        z_tsr = self.z_predictor(x_t)  # (B, 1)
        initial_interpolation_tsr = self.initial_position_interpolator.batch_evaluate(x_tsr) + self.initial_velocity_interpolator.batch_evaluate(x_tsr) * t_tsr  # (B, 1)
        return initial_interpolation_tsr + bubble_t_tsr * bubble_x_tsr * z_tsr  # (B, 1)

    def time_bubble(self, t_tsr):  # t_tsr.shape = (B, 1)
        return 1.0 - torch.exp(-torch.pow(t_tsr, 2)/(2 * self.time_bubble_sigma**2))  # (B, 1)

if __name__ == '__main__':
    device = 'cuda'
    boundary_bubble = interpolation1d.GaussianRBF1D(torch.tensor([[0, 0], [0.5, 1], [1, 0]]))
    initial_xs = torch.linspace(0, 1, 11).to(device)
    initial_us = torch.sin(torch.pi * initial_xs)
    initial_vs = torch.sin(2 * torch.pi * initial_xs)
    initial_profile_xuv = torch.cat([initial_xs.unsqueeze(1), initial_us.unsqueeze(1), initial_vs.unsqueeze(1)], dim=1)

    net = HardConstrained1dResNet(
        boundary_bubble=boundary_bubble,
        number_of_blocks=2,
        block_width=32,
        number_of_outputs=1,
        initial_profile_xuv=initial_profile_xuv,
        sigma=None,
        rbf_reg=0.0001,
        time_bubble_sigma=0.3,
        device=device,
        minimum_x_gap=1e-6
    )

    input_tsr = torch.randn(8, 2).to(device)
    output_tsr = net(input_tsr)