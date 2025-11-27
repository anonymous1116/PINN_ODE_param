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
        self.initial_conditions = torch.tensor([-1., 1.])  # (V0, R0)

    def compute_func_val(self, shared_net, derivative_batch_t):
        # derivative_batch_t = list([tensor(batch,1)]) --> concatenate
        t = torch.cat(derivative_batch_t, dim=1)

        # shared prediction: [batch, 2]
        out = shared_net(t)

        # apply your residual modification
        # initial offset V0, R0 added via broadcasting
        new_out = self.initial_conditions + (1 - torch.exp(-(t - 0.0))) * out
        # new_out shape = [batch, 2]
        return new_out   # returns tensor

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
        funcs = self.diff_eqs.compute_func_val(self.shared_net, derivative_batch_t)
        V_pred = funcs[:, 0:1]
        R_pred = funcs[:, 1:2]
        t_pred = torch.cat(derivative_batch_t, dim=1)

        dVdt = diff(V_pred, t_pred)
        dRdt = diff(R_pred, t_pred)

        res1 = dVdt - self.diff_eqs.c * (V_pred - V_pred**3 / 3 + R_pred)
        res2 = dRdt + (V_pred - self.diff_eqs.a + self.diff_eqs.b * R_pred) / self.diff_eqs.c

        derivative_loss = (torch.cat([res1, res2], dim=1)**2).mean()

        # --------------------------
        # 2. Variable loss (data fit)
        # --------------------------
        funcs_var = self.diff_eqs.compute_func_val(self.shared_net, variable_batch_t)
        variable_loss = ((funcs_var - batch_y)**2).mean()

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
    
