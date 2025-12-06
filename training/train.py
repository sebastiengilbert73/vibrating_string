import ast
import copy
import logging
import argparse
import os
import random
import pandas as pd
import torch
import sys
sys.path.append("..")
import architectures.constrained1d as arch
import utilities.scheduling as scheduling
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
from differentiation.differentiate import first_derivative, second_derivative
import architectures.interpolation1d as interpolation1d

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    outputDirectory,
    randomSeed,
    initialProfileXUV,
    architecture,
    startingNeuralNetwork,
    duration,
    tension,
    rho,
    gamma,
    scheduleFilepath,
    numberOfDiffEquResPoints,
    displayResults
):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    logging.info(f"train.main(); device = {device}; architecture = {architecture}")

    outputDirectory += "_" + architecture
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    random.seed(randomSeed)
    torch.manual_seed(randomSeed)

    # Load the initial string profile
    initial_profile_df = pd.read_csv(initialProfileXUV)
    xuv = initial_profile_df.values  # (N_initial, 3)
    if xuv.shape[1] != 3:
        raise ValueError(f"train.main(): xuv.shape[1] ({xuv.shape}) != 3")
    xs = xuv[:, 0]
    min_x, max_x = min(xs), max(xs)
    normalized_xs = np.zeros(len(xs))
    for i in range(normalized_xs.shape[0]):
        normalized_xs[i] = (xs[i] - min_x)/(max_x - min_x)
    x_tsr = torch.tensor(normalized_xs).unsqueeze(1).float().to(device)  # (N_initial, 1), normalized_values [0, 1]

    #x0_tsr = torch.cat([x_tsr, torch.zeros(x_tsr.shape[0], 1).to(device)], dim=1).float().to(device)  # (N_initial, 2)
    u0 = xuv[:, 1]  # (N_initial)
    u0_tsr = torch.tensor(u0).float().unsqueeze(1).to(device)  # (N_initial, 1)
    v0 = xuv[:, 2]  # (N_initial)
    v0_tsr = torch.tensor(v0).float().unsqueeze(1).to(device)  # (N_initial, 1)
    initial_profile_xuv_tsr = torch.cat([x_tsr, u0_tsr, v0_tsr], dim=1)  # (N_initial, 3)  normalized x

    # Create the boundary bubble
    boundary_bubble = interpolation1d.GaussianRBF1D(torch.tensor([[0, 0], [0.5, 1], [1, 0]]))  # Parabola-like shape

    # Create the neural network
    neural_net = None
    architecture_tokens = architecture.split('_')
    if architecture_tokens[0] == 'HardConstrained1dResNet':
        neural_net = arch.HardConstrained1dResNet(
            boundary_bubble=boundary_bubble,
            number_of_blocks=int(architecture_tokens[1]),
            block_width=int(architecture_tokens[2]),
            number_of_outputs=int(architecture_tokens[3]),
            initial_profile_xuv=initial_profile_xuv_tsr,
            sigma=None,
            rbf_reg=float(architecture_tokens[4]),
            time_bubble_sigma=float(architecture_tokens[5]),
            device=device,
            minimum_x_gap=float(architecture_tokens[6])
            )
    elif architecture_tokens[0] == 'HardConstrained1dMLP':
        neural_net = arch.HardConstrained1dMLP(
            boundary_bubble=boundary_bubble,
            layer_widths=ast.literal_eval(architecture_tokens[1]),
            number_of_outputs=int(architecture_tokens[2]),
            initial_profile_xuv=initial_profile_xuv_tsr,
            sigma=None,
            rbf_reg=float(architecture_tokens[3]),
            time_bubble_sigma=float(architecture_tokens[4]),
            device=device,
            minimum_x_gap=float(architecture_tokens[5])
        )
    else:
        raise NotImplementedError(f"train.main(): Not implemented architecture '{architecture}'")
    if startingNeuralNetwork is not None:
        neural_net.load_state_dict(torch.load(startingNeuralNetwork))
    neural_net.to(device)

    # Load the schedule
    schedule_df = pd.read_csv(scheduleFilepath)
    schedule = scheduling.Schedule(schedule_df)

    # Training parameters
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=schedule.parameters(1)['learning_rate'], weight_decay=0)

    # Training loop
    minimum_loss = float('inf')
    champion_neural_net = None
    phase = schedule.parameters(1)['phase']
    diff_eqn_residual_xt_tsr = 1.0 * torch.rand((numberOfDiffEquResPoints, 2), requires_grad=True).to(
        device)  # (N_res, 2), normalized dimensions
    duration_alpha = np.clip(schedule.parameters(1)['duration_alpha'], 0, 1.0)
    diff_eqn_residual_xt_tsr[:, 1] = duration_alpha * diff_eqn_residual_xt_tsr[:, 1]

    with open(os.path.join(outputDirectory, "epochLoss.csv"), 'w', buffering=1) as epoch_loss_file:
        epoch_loss_file.write("epoch,loss,is_champion\n")
        for epoch in range(1, schedule.last_epoch() + 1):
            # Set the neural network to training mode
            neural_net.train()
            neural_net.zero_grad()

            current_parameters = schedule.parameters(epoch)
            if current_parameters['phase'] != phase:
                phase = current_parameters['phase']
                logging.info(f" ---- Phase {phase} ----")
                optimizer = torch.optim.Adam(neural_net.parameters(), lr=current_parameters['learning_rate'],
                                             weight_decay=0)
                duration_alpha = np.clip(current_parameters['duration_alpha'], 0, 1.0)
                diff_eqn_residual_xt_tsr = 1.0 * torch.rand((numberOfDiffEquResPoints, 2), requires_grad=True).to(
                    device)  # (N_res, 2), normalized dimensions
                diff_eqn_residual_xt_tsr[:, 1] = duration_alpha * diff_eqn_residual_xt_tsr[:, 1]
                minimum_loss = float('inf')

            # Differential equation residual loss
            diff_eqn_residual_xt_tsr.detach_()
            diff_eqn_residual_xt_tsr.requires_grad = True

            d2u_dx2__d2u_dtdx = second_derivative(neural_net, diff_eqn_residual_xt_tsr, 0)  # (N_res, 2)
            d2u_dx2 = d2u_dx2__d2u_dtdx[:, 0]  # (N_res)

            d2u_dxdt__d2u_dt2 = second_derivative(neural_net, diff_eqn_residual_xt_tsr, 1)
            d2u_dt2 = d2u_dxdt__d2u_dt2[:, 1]  # (N_res)

            diff_eqn_residual = 1.0/duration**2 * d2u_dt2 - tension/rho/( (max_x - min_x)**2) * d2u_dx2  # (N_res)
            diff_eqn_residual_loss = criterion(diff_eqn_residual, torch.zeros_like(diff_eqn_residual))

            loss = diff_eqn_residual_loss
            is_champion = False
            if loss.item() < minimum_loss:
                minimum_loss = loss.item()
                champion_neural_net = copy.deepcopy(neural_net)
                is_champion = True
                champion_filepath = os.path.join(outputDirectory, f"{architecture}.pth")
                torch.save(champion_neural_net.state_dict(), champion_filepath)

            logging.info(f"Epoch {epoch}: loss = {loss.item()}")
            if is_champion:
                logging.info(f" **** Champion! ****")
            epoch_loss_file.write(f"{epoch},{loss.item()},{is_champion}\n")

            loss.backward()
            optimizer.step()


    logging.info(f"minimum_loss = {minimum_loss}")

    if displayResults:
        champion_neural_net.eval()
        u = np.zeros((normalized_xs.shape[0], 256), dtype=float)  # (N_initial, 256)
        delta_T = 1.0/(256 - 1)
        images = []
        for t_ndx in range(256):
            t = t_ndx * delta_T  # [0 ... 1.0]
            xt = torch.zeros(normalized_xs.shape[0], 2).to(device)  # (N_initial, 2)
            xt[:, 0] = torch.from_numpy(normalized_xs).to(device)  # Normalized x, t
            xt[:, 1] = t
            u_t = champion_neural_net(xt)  # (N_initial, 1)
            u[:, t_ndx] = u_t[:, 0].cpu().detach().numpy()

            #u_t = u_t.reshape(len(ys), len(xs)).cpu().detach().numpy()
            fig, ax = plt.subplots()
            #ax.imshow(u_t, extent=[min_x, max_x, min_y, max_y], origin='lower', aspect='auto', cmap='viridis',
            #              interpolation='nearest', vmin=-0.01, vmax=0.01)
            ax.plot(xs, u_t.detach().cpu(), color='red')
            #ax.legend()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(f't = {(t * duration):.2f} s')
            ax.set_ylim(-1.1, 1.1)
            ax.grid(True)

            image_filepath = os.path.join(outputDirectory, f"prediction_{t_ndx}.png")
            plt.savefig(image_filepath)
            plt.close()  # Prevents the figure from being displayed later
            images.append(imageio.imread(image_filepath))
            try:
                os.remove(image_filepath)
            except Exception as e:
                raise RuntimeError(f"train.main(): Error caught while trying to erase {image_filepath}: {e}")


        imageio.mimsave(os.path.join(outputDirectory, "predictions.gif"), images, duration=30, loop=0)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_train'", default='./output_train')
    parser.add_argument('--randomSeed', help="The random seed. Default: 0", type=int, default=0)
    parser.add_argument('--initialProfileXUV', help="The initial string profile, with x, u, v columns. Default: './initial_profile.csv'", default='./initial_profile.csv')
    parser.add_argument('--architecture', help="The neural network architecture. Default: 'HardConstrained1dResNet_2_32_1_0.0001_0.3_0.000001'", default='HardConstrained1dResNet_2_32_1_0.0001_0.3_0.000001')
    parser.add_argument('--startingNeuralNetwork', help="The filepath to the starting neural network. Default: 'None'", default='None')
    parser.add_argument('--duration', help="The simulation duration, in seconds. Default: 4.0", type=float, default=4.0)
    parser.add_argument('--tension', help="The tension in the string, in N. Default: 10.0", type=float, default=10.0)
    parser.add_argument('--rho', help="The string density, in kg/m. Default: 0.5", type=float, default=0.5)
    parser.add_argument('--gamma', help="The friction coefficient, in Pa-s/m. Default: 0.0", type=float, default=0.0)
    parser.add_argument('--scheduleFilepath', help="The filepath to the training schedule. Default: './schedule.csv'", default='./schedule.csv')
    parser.add_argument('--numberOfDiffEquResPoints', help="The number of points for the differential equation residual. Default: 32768", type=int, default=32768)
    parser.add_argument('--displayResults', help="Display the results", action='store_true')
    args = parser.parse_args()
    if args.startingNeuralNetwork.upper() == 'NONE':
        args.startingNeuralNetwork = None

    main(
        args.outputDirectory,
        args.randomSeed,
        args.initialProfileXUV,
        args.architecture,
        args.startingNeuralNetwork,
        args.duration,
        args.tension,
        args.rho,
        args.gamma,
        args.scheduleFilepath,
        args.numberOfDiffEquResPoints,
        args.displayResults
    )