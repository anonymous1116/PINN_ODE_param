import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import argparse
import os
import time
from abc import ABC
from scipy.integrate import solve_ivp
from solvers_utils import PretrainedSolver
from networks import FCNN, SinActv
from generators import SamplerGenerator, Generator1D
from neurodiffeq import safe_diff as diff
from utils import h1_error_trajectory

class ODESystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(0.3))
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.S0 = torch.tensor(90.)
        self.I0 = torch.tensor(10.)
        self.R0 = torch.tensor(0.)
        self.initial_conditions = [self.S0, self.I0, self.R0]

        
    def compute_derivative(self, S, I, R, t):
        """S.shape = [batch, 1]
        I.shape = [batch, 1]
        R.shape = [batch, 1]
        t.shape = [batch, 1]
        """
        N = 100
        return [
            diff(S, t) + self.beta * S * I / 100,
            diff(I, t) - self.beta * S * I / 100 + self.gamma * I,
            diff(R, t) - self.gamma * I,
        ]
            

    def compute_func_val(self, nets, derivative_batch_t):
        t_0 = 0.0
        rslt = []
        for idx, net in enumerate(nets):
            u_0 = self.initial_conditions[idx]
            network_output = net(torch.cat(derivative_batch_t, dim=1))
            new_network_output = u_0 + (1 - torch.exp(-torch.cat(derivative_batch_t, dim=1) + t_0)) * network_output
            rslt.append(new_network_output)
        return rslt


class BaseSolver(ABC, PretrainedSolver, nn.Module):
    def __init__(self, diff_eqs, net1, net2, net3):
        super().__init__()
        self.diff_eqs = diff_eqs
        self.net1 = net1
        self.net1 = net2
        self.net1 = net3
        self.nets = [net1, net2, net3]

    def compute_loss(self, derivative_batch_t, variable_batch_t, batch_y, derivative_weight=0.5, return_parts = False):
        """derivative_batch_t can be sampled in any distribution and sample size.
        derivative_batch_t= list([derivative_batch_size, 1])
        """
        derivative_loss = 0.0
        derivative_funcs = self.diff_eqs.compute_func_val(self.nets, derivative_batch_t)
        derivative_residuals = self.diff_eqs.compute_derivative(*derivative_funcs,
                                                                *derivative_batch_t)
        derivative_residuals = torch.cat(derivative_residuals, dim=1)  # [100, 5]
        derivative_loss += (derivative_residuals ** 2).mean()

        """(variable_batch_t, batch_y) is sampled from data
         variable_batch_t =list([variable_batch_size, 1])
        batch_y.shape = [variable_batch_size, 1]
        """
        variable_loss = 0.0
        variable_funcs = self.diff_eqs.compute_func_val(self.nets, variable_batch_t)
        variable_funcs = torch.cat(variable_funcs, dim=1)  # [10, 5]
        variable_loss += ((variable_funcs - batch_y) ** 2).mean()
        
        total_loss = derivative_weight * derivative_loss + variable_loss

        if return_parts:
            return total_loss, derivative_loss, variable_loss
        return total_loss


# 100 simulations
def fOde(theta, x, tvec, N=100):
    S = x[:, 0]
    I = x[:, 1]
    R = x[:, 2]

    beta = theta[0]
    gamma = theta[1]

    Sdt = -beta * S * I / N
    Idt = beta * S * I / N - gamma * I
    Rdt = gamma * I

    result = np.stack([Sdt, Idt, Rdt], axis=1)
    return result


def main(args):
    true_theta = [0.3, 0.1]
    true_x0 = [90, 10, 0]
    true_sigma = args.true_sigma
    sci_str = format(args.true_sigma, ".0e")
    penalty = format(args.penalty, ".1e").replace(".", "_")
    
    print("sigma: ", sci_str, "penalty: ", penalty)
    n = 101
    tvecObs = np.linspace(0, 100, num=n)
    sol = solve_ivp(lambda t, y: fOde(true_theta, y.transpose(), t).transpose(),
                    t_span=[0, tvecObs[-1]], y0=true_x0, t_eval=tvecObs, vectorized=True)
    
    print("sol:", sol)

    ydataTruth = sol.y
    ydataTruth = np.array(ydataTruth).transpose()

    tvecFull = np.linspace(0, 20, num=2001)
    solFull = solve_ivp(lambda t, y: fOde(true_theta, y.transpose(), t).transpose(),
                        t_span=[0, tvecFull[-1]], y0=true_x0, t_eval=tvecFull, vectorized=True)
    ydataTruthFull = solFull.y
    ydataTruthFull = np.array(ydataTruthFull).transpose()

    SEED = pd.read_table("./Experiments/FN_seed.txt", delim_whitespace=True, header=None)
    SEED = torch.tensor(data=SEED.values, dtype=torch.int)
    s = args.seed
    np.random.seed(SEED[s, 0].data)
    torch.manual_seed(SEED[s, 0].data)
    
    ydata = ydataTruth[:, 1] + np.random.normal(0, true_sigma, ydataTruth[:, 0].size)
    
    output_dir = f"../depot_hyun/hyun/ODE_param/SIR/sig{sci_str}/lambda_{penalty}"
    
    os.makedirs(f"{output_dir}/ydata", exist_ok=True)
    os.makedirs(f"{output_dir}/results", exist_ok=True)
    
    t = torch.linspace(0., 20., n)  # torch.float32
    true_y = torch.from_numpy(ydata)  # torch.float64
    true_y = torch.reshape(true_y, (true_y.size(0), 1))
    t_min = 0.0
    t_max = 100.0
    variable_batch_size = 7
    derivative_batch_size = 1000
    train_generator = SamplerGenerator(
        Generator1D(size=derivative_batch_size, t_min=t_min, t_max=t_max, method='equally-spaced-noisy'))
    model = BaseSolver(diff_eqs=ODESystem(),
                    net1=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=SinActv),
                    net2=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=SinActv),
                    net3=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=SinActv)
                    )
    best_model = BaseSolver(diff_eqs=ODESystem(),
                    net1=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=SinActv),
                    net2=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=SinActv),
                    net3=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=SinActv)
                    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #            optimizer,
    #            mode="min",
    #            factor=0.5,
    #            patience=3000,
    #            min_lr=1e-6
    #            )
    
    y_ind = np.arange(n)
    train_epochs = 15000  # 10000
    loss_history = []
    #num_pilot = train_epochs/10
    for epoch in range(train_epochs):
        np.random.shuffle(y_ind)
        epoch_loss = 0.0
        batch_loss = 0.0
        # model.train()
        optimizer.zero_grad()
        for i in range(0, n, variable_batch_size):
            variable_batch_id = y_ind[i:(i + variable_batch_size)]
            # optimizer.zero_grad()
            batch_loss = model.compute_loss(
                derivative_batch_t=[s.reshape(-1, 1) for s in train_generator.get_examples()],  # list([100, 1])
                variable_batch_t=[t[variable_batch_id].view(-1, 1)],  # list([7, 1])
                batch_y=true_y[variable_batch_id],  # [7, 2]
                derivative_weight=args.penalty)
            
            batch_loss.backward()
            epoch_loss += batch_loss.item()
        if epoch % 100 == 0:
            print(f'Train Epoch: {epoch} '
                f'\tLoss: {batch_loss.item():.6f}')
        optimizer.step()
        #scheduler.step(batch_loss)
        loss_history.append(epoch_loss)
        if loss_history[-1] == min(loss_history):
            best_model.load_state_dict(model.state_dict())
            best_model.eval()
            param_results = np.array([best_model.diff_eqs.beta.data, best_model.diff_eqs.gamma.data])
            print(f"param_results at: {epoch}, {param_results}")
    

    # check estimated parameters
    best_model.eval()
    param_results = np.array([best_model.diff_eqs.beta.data, best_model.diff_eqs.gamma.data])
    print("param_results:", param_results)
    # check estimated path
    with torch.no_grad():
        estimate_t = torch.linspace(0., 20., 2001)
        estimate_funcs = best_model.diff_eqs.compute_func_val(best_model.nets, [estimate_t.view(-1, 1)])
        estimate_funcs = torch.cat(estimate_funcs, dim=1)
    estimate_funcs = estimate_funcs.numpy()
    #trajectory_RMSE = np.sqrt(np.mean((estimate_funcs[observed_ind, :] - ydataTruthFull[observed_ind, :]) ** 2,
    #                                        axis=0))
    trajectory_RMSE = np.sqrt(np.mean((estimate_funcs[observed_ind, :] - ydataTruthFull[observed_ind, :]) ** 2,
                                            axis=0))
    
    dt = tvecObs[1] - tvecObs[0]

    val_term = np.sum((estimate_funcs[observed_ind, :] - ydataTruthFull[observed_ind, :]) ** 2) * dt
    print("val_term_part:", val_term)
    tmp = fOde(theta = param_results, x = estimate_funcs[observed_ind,:], tvec = tvecObs)
    dtrue = fOde(theta = true_theta, x = ydataTruth, tvec = estimate_t)
    der_term = np.sum((tmp  - dtrue) ** 2)* dt
    print("der_term_part:", der_term)
    h1_part = np.sqrt(val_term + der_term)
    print("H1: ", h1_part)
    dt = estimate_t[1] - estimate_t[0]
    
    l2 = np.sqrt(np.mean((ydata - estimate_funcs[observed_ind, :]) ** 2))
    print("l2: ", l2)
    best_model.eval()
    #with torch.no_grad():  # <-- IMPORTANT: only if you *don't* need gradients here

    new_derivative_batch_size = 2000
    new_train_generator = SamplerGenerator(
        Generator1D(size=new_derivative_batch_size, t_min=t_min, t_max=t_max, method='equally-spaced-noisy'))

    total, dloss, vloss = best_model.compute_loss(
        derivative_batch_t=[s.reshape(-1, 1) for s in new_train_generator.get_examples()],
        variable_batch_t=[t.view(-1, 1)],
        batch_y=true_y,
        derivative_weight=0.5,
        return_parts=True
    )
    print("derivative_loss =", float(dloss ** (1/2)), "l2: ", vloss ** (1/2), "total: ", total)

    
    print(f"Simulation {s} completed")
    #np.save(f"{output_dir}/results/trajectory_RMSE_{s}.npy", trajectory_RMSE)
    #np.save(f"{output_dir}/results/param_results_{s}.npy", param_results)
    #np.save(f"{output_dir}/results/trajectory_{s}.npy", trajectory_RMSE)
    #np.save(f"{output_dir}/results/h1_errors_{s}.npy", np.array(h1_part))
    #np.save(f"{output_dir}/results/l2_{s}.npy", l2)
    #np.save(f"{output_dir}/results/derivative_loss_{s}.npy", float(dloss ** (1/2)))
    
    print(f"Simulation {s} saved completed")
    

def get_args():
    parser = argparse.ArgumentParser(description="Run simulation with customizable parameters.")
    parser.add_argument("--seed", type = int, default = 1,
                        help = "See number (default: 1)")
    parser.add_argument("--true_sigma", type = float, default = 0.2,
                        help = "observation errors (default: 0.2)")
    parser.add_argument("--penalty", type = float, default = 0.8,
                        help = "observation errors (default: 0.8)")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)
    
