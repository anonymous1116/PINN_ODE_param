import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import argparse
import os
import time, copy

from abc import ABC
from scipy.integrate import solve_ivp
from sklearn.model_selection import KFold
from solvers_utils import PretrainedSolver
from networks import FCNN, SinActv
from generators import SamplerGenerator, Generator1D
from neurodiffeq import safe_diff as diff
from utils import h1_error_trajectory

class ODESystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.5))
        self.V0 = nn.Parameter(torch.tensor(-1.))
        self.R0 = nn.Parameter(torch.tensor(1.))
        self.initial_conditions = [self.V0, self.R0]

    def compute_derivative(self, V, R, t):
        """v.shape = [batch, 1]
        t.shape = [batch, 1]
        """
        return [diff(V, t) - self.c * (V - V ** 3 / 3 + R), diff(R, t) + (V - self.a + self.b * R) / self.c]

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
    def __init__(self, diff_eqs, net1, net2):
        super().__init__()
        self.diff_eqs = diff_eqs
        self.net1 = net1
        self.net2 = net2
        self.nets = [net1, net2]

    def compute_loss(self, derivative_batch_t, variable_batch_t, batch_y, derivative_weight=0.5):
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


def FN_CV(penalty, obs, t, model, train_generator, train_idx, val_idx, variable_batch_size = 7, train_epochs = 10000):
    model_copy = copy.copy(model)
    best_model_copy = copy.copy(model)
    optimizer = torch.optim.Adam(model_copy.parameters(), lr=5e-3)
    y_ind = np.arange(len(train_idx))
    obs_train = obs[train_idx]
    obs_val = obs[val_idx]
    loss_history = []

    time_train = t[train_idx]

    # adfasdfasfdfdf 
    for epoch in range(train_epochs):
        np.random.shuffle(y_ind)
        epoch_loss = 0.0
        batch_loss = 0.0
        model_copy.train()
        optimizer.zero_grad()
        for i in range(0, len(y_ind), variable_batch_size):
            variable_batch_id = y_ind[i:(i + variable_batch_size)]
            # optimizer.zero_grad()
            batch_loss = model_copy.compute_loss(
                derivative_batch_t=[s.reshape(-1, 1) for s in train_generator.get_examples()],  # list([100, 1])
                variable_batch_t=[time_train[variable_batch_id].view(-1, 1)],  # list([7, 1])
                batch_y=obs_train[variable_batch_id],  # [7, 2]
                derivative_weight=penalty
                )
            
            batch_loss.backward()
            epoch_loss += batch_loss.item()
        if epoch % 100 == 0:
            print(f'Train Epoch: {epoch} '
                f'\tLoss: {batch_loss.item():.6f}')
        optimizer.step()
        #scheduler.step(batch_loss)
        loss_history.append(epoch_loss)
        if loss_history[-1] == min(loss_history):
            best_model_copy.load_state_dict(model_copy.state_dict())

    # check estimated path
    best_model_copy.eval()
    #param_results = np.array([best_model_copy.diff_eqs.a.data, best_model_copy.diff_eqs.b.data, best_model_copy.diff_eqs.c.data])

    with torch.no_grad():
        estimate_t = torch.linspace(0., 20., 41)
        estimate_funcs = best_model_copy.diff_eqs.compute_func_val(best_model_copy.nets, [estimate_t.view(-1, 1)])
        estimate_funcs = torch.cat(estimate_funcs, dim=1)
    estimate_funcs = estimate_funcs.numpy()

    CV_error = np.mean((estimate_funcs[val_idx,:] - obs_val.numpy()) ** 2, axis =0 )
    del model_copy, best_model_copy
    return CV_error

def main(args):
    true_theta = [0.2, 0.2, 3]
    true_x0 = [-1, 1]
    true_sigma = [args.true_sigma, args.true_sigma]
    sci_str = format(args.true_sigma, ".0e")
    
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

    s = args.seed

    np.random.seed(SEED[s, 0].data)
    torch.manual_seed(SEED[s, 0].data)
    ydataV = ydataTruth[:, 0] + np.random.normal(0, true_sigma[0], ydataTruth[:, 0].size)
    ydataR = ydataTruth[:, 1] + np.random.normal(0, true_sigma[1], ydataTruth[:, 1].size)
    ydata = np.stack([np.array(ydataV), np.array(ydataR)], axis=1)

    output_dir = f"../depot_hyun/hyun/ODE_param/FN_sig{sci_str}/CV"
    os.makedirs(f"{output_dir}/ydata", exist_ok=True)
    os.makedirs(f"{output_dir}/results", exist_ok=True)
    
    
    t = torch.linspace(0., 20., n)  # torch.float32
    true_y = torch.from_numpy(ydata)  # torch.float64
    t_min = 0.0
    t_max = 20.0
    variable_batch_size = 7
    derivative_batch_size = 100
    train_generator = SamplerGenerator(
        Generator1D(size=derivative_batch_size, t_min=t_min, t_max=t_max, method='equally-spaced-noisy'))
    model = BaseSolver(diff_eqs=ODESystem(),
                    net1=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=SinActv),
                    net2=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=SinActv))
    best_model = BaseSolver(diff_eqs=ODESystem(),
                    net1=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=SinActv),
                    net2=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=SinActv))
    

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=2726)
    penalty_list = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    CV_error_list = []
    for penalty in penalty_list:
        CV_error = 0
        num = 0
        for train_idx, val_idx in kfold.split(true_y):
            print(f"penalty: {penalty}, CV: {num}/{k_folds}")
            CV_error += FN_CV(penalty, true_y, t, copy.copy(model), train_generator, train_idx, val_idx, variable_batch_size = 7, train_epochs = 300)
            num+=1
        print("CV_error: ", CV_error)
        CV_error_list.append(CV_error)
    
    CV_error_list = np.array(CV_error_list)

    penalty_CV = CV_error_list[np.argmin(CV_error_list)]

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
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
                derivative_weight=penalty_CV
                )
            
            batch_loss.backward()
            epoch_loss += batch_loss.item()
            if i % 100 == 0:
                print(f'Train Epoch: {epoch} '
                    f'[{i:05}/{n} '
                    f'\tLoss: {batch_loss.item():.6f}')
        optimizer.step()
        #scheduler.step(batch_loss)
        loss_history.append(epoch_loss)
        if loss_history[-1] == min(loss_history):
            best_model.load_state_dict(model.state_dict())

    # check estimated parameters
    best_model.eval()
    param_results = np.array([best_model.diff_eqs.a.data, best_model.diff_eqs.b.data, best_model.diff_eqs.c.data])

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
    

    print(f"Simulation {s} completed")
    np.save(f"{output_dir}/results/trajectory_RMSE_{s}.npy", trajectory_RMSE)
    np.save(f"{output_dir}/results/param_results_{s}.npy", param_results)
    np.save(f"{output_dir}/results/trajectory_{s}.npy", trajectory_RMSE)
    np.save(f"{output_dir}/results/h1_errors_{s}.npy", np.array(h1_part))
    np.save(f"{output_dir}/results/CV_errors_{s}.npy", np.array(CV_error_list))
    np.save(f"{output_dir}/results/lambda_{s}.npy", penalty_CV)
    
    print(f"Simulation {s} saved completed")
    

def get_args():
    parser = argparse.ArgumentParser(description="Run simulation with customizable parameters.")
    parser.add_argument("--seed", type = int, default = 1,
                        help = "See number (default: 1)")
    parser.add_argument("--true_sigma", type = float, default = 0.2,
                        help = "observation errors (default: 0.2)")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)
    
