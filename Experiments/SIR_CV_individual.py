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
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.2))
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
        N = 100.0
        return [
            diff(S, t) + self.beta * S * I / N,
            diff(I, t) - self.beta * S * I / N + self.gamma * I,
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
        self.net2 = net2
        self.net3 = net3
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
        I_pred = variable_funcs[1]   # second network = I(t)
        variable_loss += ((I_pred - batch_y) ** 2).mean()
        
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



def SIR_CV(penalty, obs, t, model, train_generator, train_idx, val_idx, variable_batch_size = 7, train_epochs = 10000):
    model_copy = copy.deepcopy(model)
    best_model_copy = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model_copy.parameters(), lr=1e-3)
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
        estimate_t = torch.linspace(0., 100., 101)
        estimate_funcs = best_model_copy.diff_eqs.compute_func_val(best_model_copy.nets, [estimate_t.view(-1, 1)])
        estimate_funcs = torch.cat(estimate_funcs, dim=1)
    estimate_funcs = estimate_funcs.numpy()

    t_min = min(t)
    t_max = max(t)    
    new_derivative_batch_size = 2000
    new_train_generator = SamplerGenerator(
        Generator1D(size=new_derivative_batch_size, t_min=t_min, t_max=t_max, method='equally-spaced-noisy'))

    total, dloss, vloss = best_model_copy.compute_loss(
        derivative_batch_t=[s.reshape(-1, 1) for s in new_train_generator.get_examples()],
        variable_batch_t=[t[val_idx].view(-1, 1)],
        batch_y=obs_val,
        derivative_weight=penalty,
        return_parts=True
    )
    return total, dloss, vloss


def main(args):
    true_theta = [0.3, 0.1]
    true_x0 = [90, 10, 0]
    true_sigma = args.true_sigma
    sci_str = format(args.true_sigma, ".1e").replace(".", "_")
    penalty = format(args.penalty, ".1e").replace(".", "_")
    
    
    print("sigma: ", sci_str)
    n = 101
    tvecObs = np.linspace(0, 100, num=n)
    sol = solve_ivp(lambda t, y: fOde(true_theta, y.transpose(), t).transpose(),
                    t_span=[0, tvecObs[-1]], y0=true_x0, t_eval=tvecObs, vectorized=True)
    
    print("sol:", sol)

    ydataTruth = sol.y
    ydataTruth = np.array(ydataTruth).transpose()

    tvecFull = np.linspace(0, 100, num=2001)
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
    
    output_dir = f"../depot_hyun/hyun/ODE_param/SIR/sig_{sci_str}/lambda_{penalty}_CV"
    os.makedirs(f"{output_dir}/ydata", exist_ok=True)
    os.makedirs(f"{output_dir}/results", exist_ok=True)
    
    
    t = torch.linspace(0., 100., n)  # torch.float32
    true_y = torch.from_numpy(ydata)  # torch.float64
    true_y = torch.reshape(true_y, (true_y.size(0), 1))
    t_min = 0.0
    t_max = 100.0
    variable_batch_size = 7
    derivative_batch_size = 1000
    train_generator = SamplerGenerator(
        Generator1D(size=derivative_batch_size, t_min=t_min, t_max=t_max, method='equally-spaced-noisy'))
    hidden_units = [32, 32]
    model = BaseSolver(diff_eqs=ODESystem(),
                    net1=FCNN(n_input_units=1, n_output_units=1, hidden_units=hidden_units, actv=SinActv),
                    net2=FCNN(n_input_units=1, n_output_units=1, hidden_units=hidden_units, actv=SinActv),
                    net3=FCNN(n_input_units=1, n_output_units=1, hidden_units=hidden_units, actv=SinActv)
                    )
    best_model = BaseSolver(diff_eqs=ODESystem(),
                    net1=FCNN(n_input_units=1, n_output_units=1, hidden_units=hidden_units, actv=SinActv),
                    net2=FCNN(n_input_units=1, n_output_units=1, hidden_units=hidden_units, actv=SinActv),
                    net3=FCNN(n_input_units=1, n_output_units=1, hidden_units=hidden_units, actv=SinActv)
                    )
    
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=2726)


    start_time = time.time()
    cumulative_time = 0
    CV_l2_error = 0
    CV_deri_error = 0
    num = 0
    penalty = args.penalty
    for train_idx, val_idx in kfold.split(true_y):
        print(f"penalty: {penalty}, CV: {num}/{k_folds}")
        #_, CV_l2_error, CV_deri_error += FN_CV(penalty, true_y, t, model, train_generator, train_idx, val_idx, variable_batch_size = 7, train_epochs = 10000)
        results = SIR_CV(penalty, true_y, t, model, train_generator, train_idx, val_idx, variable_batch_size = 7, train_epochs = 10)
        CV_deri_error += results[1] 
        CV_l2_error += results[2]

        num+=1
        end_time = time.time()
        cumulative_time+= end_time-start_time
        print(f"cumulative time: {cumulative_time:.3f}" )

    CV_l2_error_final =  CV_l2_error/k_folds ** (1/2)
    CV_deri_error_final = (CV_deri_error/k_folds) ** (1/2)
    print("CV_l2_error: ", CV_l2_error_final)
    print("CV_deri_error: ", CV_deri_error_final)

    np.save(f"{output_dir}/results/CV_l2_{s}.npy", float(CV_l2_error_final))
    np.save(f"{output_dir}/results/CV_derivative_loss_{s}.npy", float(CV_deri_error_final))
    
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
    
