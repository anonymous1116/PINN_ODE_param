import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import argparse
import os,sys
import time
from abc import ABC
import copy
from scipy.integrate import solve_ivp
from networks import SinActv
from solvers_utils import PretrainedSolver
from generators import SamplerGenerator, Generator1D
from neurodiffeq import safe_diff as diff
from warnings import warn



class FCNN(nn.Module):
    """A fully connected neural network.
    :param n_input_units: Number of units in the input layer, defaults to 1.
    :type n_input_units: int
    :param n_output_units: Number of units in the output layer, defaults to 1.
    :type n_output_units: int
    :param n_hidden_units: [DEPRECATED] Number of hidden units in each layer
    :type n_hidden_units: int
    :param n_hidden_layers: [DEPRECATED] Number of hidden mappsings (1 larger than the actual number of hidden layers)
    :type n_hidden_layers: int
    :param actv: The activation layer constructor after each hidden layer, defaults to `torch.nn.Tanh`.
    :type actv: class
    :param hidden_units: Number of hidden units in each hidden layer. Defaults to (32, 32).
    :type hidden_units: Tuple[int]
    .. note::
        The arguments "n_hidden_units" and "n_hidden_layers" are deprecated in favor of "hidden_units".
    """

    def __init__(self, n_input_units=1, n_output_units=1, n_hidden_units=None, n_hidden_layers=None,
                 actv=nn.Tanh, hidden_units=None):
        r"""Initializer method.
        """
        super(FCNN, self).__init__()

        # FORWARD COMPATIBILITY
        # If only one of {n_hidden_unit, n_hidden_layers} is specified, fill-in the other one
        if n_hidden_units is None and n_hidden_layers is not None:
            n_hidden_units = 32
        elif n_hidden_units is not None and n_hidden_layers is None:
            n_hidden_layers = 1

        # FORWARD COMPATIBILITY
        # When {n_hidden_unit, n_hidden_layers} are specified, construct an equivalent hidden_units if None is provided
        if n_hidden_units is not None or n_hidden_layers is not None:
            if hidden_units is None:
                hidden_units = tuple(n_hidden_units for _ in range(n_hidden_layers + 1))
                warn(f"`n_hidden_units` and `n_hidden_layers` are deprecated, "
                     f"pass `hidden_units={hidden_units}` instead",
                     FutureWarning)
            else:
                warn(f"Ignoring `n_hidden_units` and `n_hidden_layers` in favor of `hidden_units={hidden_units}`",
                     FutureWarning)

        # If none of {n_hidden_units, n_hidden_layers, hidden_layers} are specified, use (32, 32) by default
        if hidden_units is None:
            hidden_units = (32, 32)

        # If user passed in a list, iterator, etc., convert it to tuple
        if not isinstance(hidden_units, tuple):
            hidden_units = tuple(hidden_units)

        units = (n_input_units,) + hidden_units
        layers = []
        for i in range(len(units) - 1):
            layers.append(nn.Linear(units[i], units[i + 1]))
            layers.append(actv())
        # There's not activation in after the last layer
        layers.append(nn.Linear(units[-1], n_output_units))
        self.NN = torch.nn.Sequential(*layers)

    def forward(self, t):
        x = self.NN(t)
        return x


class SharedFCNN(nn.Module):
    def __init__(self, n_input_units=1, n_output_units=2, hidden_units=[64, 64], actv=nn.Tanh):
        super().__init__()

        layers = []
        prev_units = n_input_units
        for h in hidden_units:
            layers.append(nn.Linear(prev_units, h))
            layers.append(actv())
            prev_units = h

        layers.append(nn.Linear(prev_units, n_output_units))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)   # output shape: [batch, 2]


class ODESystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.5))
        self.V0 = nn.Parameter(torch.tensor(-1.))
        self.R0 = nn.Parameter(torch.tensor(1.))

    def compute_derivative(self, V, R, t):
        """v.shape = [batch, 1]
        t.shape = [batch, 1]
        """
        return [diff(V, t) - self.c * (V - V ** 3 / 3 + R), diff(R, t) + (V - self.a + self.b * R) / self.c]

    def compute_func_val(self, shared_net, batch_t):
        # batch_t: list of tensors, e.g. [t] with shape [B, 1]
        t = torch.cat(batch_t, dim=1)  # [B, 1]

        out = shared_net(t)            # [B, 2]

        # learnable initial conditions
        init = torch.stack([self.V0, self.R0]).to(out.device)  # [2]
        init = init.unsqueeze(0)                               # [1, 2]

        phi = 1 - torch.exp(-t)        # [B, 1], will broadcast to [B, 2]
        new_out = init + phi * out     # [B, 2]

        return new_out

class BaseSolver(ABC, PretrainedSolver, nn.Module):
    def __init__(self, diff_eqs, shared_net):
        super().__init__()
        self.diff_eqs = diff_eqs
        self.shared_net = shared_net

    def compute_loss(self, derivative_batch_t, variable_batch_t, batch_y,
                     derivative_weight=0.5):

        # --------------------------
        # 1. Derivative loss
        # --------------------------
        derivative_loss = 0.0
        derivative_funcs = self.diff_eqs.compute_func_val(self.shared_net, derivative_batch_t)
        derivative_residuals = self.diff_eqs.compute_derivative(*derivative_funcs,
                                                                *derivative_batch_t)
        derivative_residuals = torch.cat(derivative_residuals, dim=1)  # [100, 5]
        derivative_loss += (derivative_residuals ** 2).mean()
        
        # --------------------------
        # 2. Variable loss (data fit)
        # --------------------------
        """(variable_batch_t, batch_y) is sampled from data
         variable_batch_t =list([variable_batch_size, 1])
        batch_y.shape = [variable_batch_size, 1]
        """
        variable_loss = 0.0
        variable_funcs = self.diff_eqs.compute_func_val(self.shared_net, variable_batch_t)
        variable_funcs = torch.cat(variable_funcs, dim=1)  # [10, 5]
        variable_loss += ((variable_funcs - batch_y) ** 2).mean()
        return derivative_weight * derivative_loss + variable_loss

        


# 100 simulations
def fOde(theta, x, tvec):
    V = x[:, 0]
    R = x[:, 1]
    Vdt = theta[2] * (V - pow(V,3) / 3.0 + R)
    Rdt = -1.0/theta[2] * ( V - theta[0] + theta[1] * R)
    result = np.stack([Vdt, Rdt], axis=1)
    return result


def fOdeDx(theta, x, tvec):
    resultDx = np.zeros(shape=[np.shape(x)[0], np.shape(x)[1], np.shape(x)[1]])
    V = x[:, 0]
    R = x[:, 1]
    resultDx[:, 0, 0] = theta[2] * (1 - np.square(V))
    resultDx[:, 1, 0] = theta[2]
    resultDx[:, 0, 1] = -1.0 / theta[2]
    resultDx[:, 1, 1] = -1.0*theta[1]/theta[2]
    return resultDx


def fOdeDtheta(theta, x, tvec):
    resultDtheta = np.zeros(shape=[np.shape(x)[0], np.shape(theta)[0], np.shape(x)[1]])
    V = x[:, 0]
    R = x[:, 1]
    resultDtheta[:, 2, 0] = V - pow(V, 3) / 3.0 + R
    resultDtheta[:, 0, 1] = 1.0 / theta[2]
    resultDtheta[:, 1, 1] = -R / theta[2]
    resultDtheta[:, 2, 1] = 1.0/pow(theta[2], 2) * (V - theta[0] + theta[1] * R)
    return resultDtheta


def main(args):
    true_theta = [0.2, 0.2, 3]
    true_x0 = [-1, 1]
    true_sigma = [0.2, 0.2]
    n = 41
    tvecObs = np.linspace(0, 20, num=n)
    sol = solve_ivp(lambda t, y: fOde(true_theta, y.transpose(), t).transpose(),
                    t_span=[0, tvecObs[-1]], y0=true_x0, t_eval=tvecObs, vectorized=True)
    ydataTruth = sol.y
    ydataTruth = np.array(ydataTruth).transpose()

    tvecFull = np.linspace(0, 20, num=2001)
    solFull = solve_ivp(lambda t, y: fOde(true_theta, y.transpose(), t).transpose(),
                        t_span=[0, tvecFull[-1]], y0=true_x0, t_eval=tvecFull, vectorized=True)
    ydataTruthFull = solFull.y
    ydataTruthFull = np.array(ydataTruthFull).transpose()

    SEED = pd.read_table("./Experiments/FN_seed.txt", delim_whitespace=True, header=None)
    SEED = torch.tensor(data=SEED.values, dtype=torch.int)
    
    observed_ind = np.linspace(0, 2000, num=41, dtype=int)
    #observed_ind = np.concatenate((observed_ind, np.array([1100, 1200, 1300, 1400, 1500, 1700, 2000])))


    s = args.seed

    np.random.seed(SEED[s, 0].data)
    torch.manual_seed(SEED[s, 0].data)
    ydataV = ydataTruth[:, 0] + np.random.normal(0, true_sigma[0], ydataTruth[:, 0].size)
    ydataR = ydataTruth[:, 1] + np.random.normal(0, true_sigma[1], ydataTruth[:, 1].size)
    ydata = np.stack([np.array(ydataV), np.array(ydataR)], axis=1)

    output_dir = f"../depot_hyun/hyun/ODE_param/FN_SA"
    os.makedirs(f"{output_dir}/ydata", exist_ok=True)
    os.makedirs(f"{output_dir}/results", exist_ok=True)
    
    
    np.save(f"{output_dir}/ydata/ydata_{s}.npy", ydata)
    if s == 1:
        np.save(f"{output_dir}/ydata/ydataTruthFull.npy", ydataTruthFull)
        np.save(f"{output_dir}/ydata/ydataTruth.npy", ydataTruth)
        np.save(f"{output_dir}/ydata/observed_ind.npy", observed_ind)
        print("ydataTruthFull, ydataTruth, observed_ind saved", flush=True)    
    
    t = torch.linspace(0., 20., n)  # torch.float32
    true_y = torch.from_numpy(ydata)  # torch.float64
    t_min = 0.0
    t_max = 20.0
    variable_batch_size = 7
    derivative_batch_size = 100
    train_generator = SamplerGenerator(
        Generator1D(size=derivative_batch_size, t_min=t_min, t_max=t_max, method='equally-spaced-noisy'))
    model = BaseSolver(
    diff_eqs = ODESystem(),
    shared_net = SharedFCNN(
        n_input_units=1,
        n_output_units=2,
        hidden_units=[64, 64],
        actv=SinActv
    )
    )
    best_model = copy.deepcopy(model)  # initialize
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    y_ind = np.arange(n)
    train_epochs = 15000  # 10000
    loss_history = []
    best_loss = float('inf')
    for epoch in range(train_epochs):

        # ---- 1) Shuffle data indices ----
        y_ind = np.random.permutation(n)

        # ---- 2) Sample derivative batch once per epoch ----
        derivative_batch_t = [
            s.reshape(-1, 1) for s in train_generator.get_examples()
        ]  # e.g. list([100,1])

        epoch_loss = 0.0

        model.train()

        # ---- 3) Loop through variable data in mini-batches ----
        for i in range(0, n, variable_batch_size):

            variable_batch_id = y_ind[i: i + variable_batch_size]
            variable_batch_t = [t[variable_batch_id].view(-1, 1)]
            batch_y = true_y[variable_batch_id]

            # ---- forward + backward ----
            optimizer.zero_grad()
            batch_loss = model.compute_loss(
                derivative_batch_t=derivative_batch_t,
                variable_batch_t=variable_batch_t,
                batch_y=batch_y,
                derivative_weight=0.8,
            )
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()

        loss_history.append(epoch_loss)

        # ---- 4) Save best model ----
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model = copy.deepcopy(model)

        # optional: print
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss = {epoch_loss:.6f}")
        
    
    # check estimated parameters
    with torch.no_grad():
        param_results = np.array([best_model.diff_eqs.a.data, best_model.diff_eqs.b.data, best_model.diff_eqs.c.data])

    # check estimated path
    with torch.no_grad():
        estimate_t = torch.linspace(0., 20., 2001)
        estimate_funcs = best_model.diff_eqs.compute_func_val(best_model.shared_net, [estimate_t.view(-1, 1)])
        estimate_funcs = estimate_funcs.numpy()
    trajectory_RMSE = np.sqrt(np.mean((estimate_funcs[observed_ind, :] - ydataTruthFull[observed_ind, :]) ** 2,
                                            axis=0))
    print(f"Simulation {s} finished")
    np.save(f"{output_dir}/results/trajectory_RMSE_{s}.npy", trajectory_RMSE)
    np.save(f"{output_dir}/results/param_results_{s}.npy", param_results)
    np.save(f"{output_dir}/results/trajectory_{s}.npy", trajectory_RMSE)
    
    

def get_args():
    parser = argparse.ArgumentParser(description="Run simulation with customizable parameters.")
    parser.add_argument("--seed", type = int, default = 1,
                        help = "See number (default: 1)")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)
    
