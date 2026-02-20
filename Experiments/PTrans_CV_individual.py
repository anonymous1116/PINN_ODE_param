import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import copy, time
from abc import ABC
from solvers_utils import PretrainedSolver
from networks import FCNN
from generators import SamplerGenerator, Generator1D
from neurodiffeq import safe_diff as diff
from sklearn.model_selection import KFold

import argparse, os


class ODESystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.k1 = nn.Parameter(torch.tensor(0.5))
        self.k2 = nn.Parameter(torch.tensor(0.5))
        self.k3 = nn.Parameter(torch.tensor(0.5))
        self.k4 = nn.Parameter(torch.tensor(0.5))
        self.V = nn.Parameter(torch.tensor(0.5))
        self.Km = nn.Parameter(torch.tensor(0.5))
        self.S0 = nn.Parameter(torch.tensor(1.))
        self.Sd0 = nn.Parameter(torch.tensor(0.))
        self.R0 = nn.Parameter(torch.tensor(1.))
        self.SR0 = nn.Parameter(torch.tensor(0.))
        self.Rpp0 = nn.Parameter(torch.tensor(0.))
        self.initial_conditions = [self.S0, self.Sd0, self.R0, self.SR0, self.Rpp0]

    def compute_derivative(self, S, Sd, R, SR, Rpp, t):
        """S.shape = [batch, 1]
        t.shape = [batch, 1]
        """
        return [diff(S, t) + self.k1 * S + self.k2 * S * R - self.k3 * SR, diff(Sd, t) - self.k1 * S,
                diff(R, t) + self.k2 * S * R - self.k3 * SR - self.V * Rpp / (self.Km + Rpp),
                diff(SR, t) - self.k2 * S * R + self.k3 * SR + self.k4 * SR,
                diff(Rpp, t) - self.k4 * SR + self.V * Rpp / (self.Km + Rpp)]

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
    def __init__(self, diff_eqs, net1, net2, net3, net4, net5):
        super().__init__()
        self.diff_eqs = diff_eqs
        self.net1 = net1
        self.net2 = net2
        self.net3 = net3
        self.net4 = net4
        self.net5 = net5
        self.nets = [net1, net2, net3, net4, net5]

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
         variable_batch_t = list([variable_batch_size, 1])
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



def fOde(theta, x, tvec=None, eps=1e-12):
    """
    PTrans RHS f(x; theta) for 100 (or any) simulations in parallel.

    Parameters
    ----------
    theta : array-like of length 6
        [k1, k2, k3, k4, V, Km]
    x : ndarray, shape (N, 5)
        Columns = [S, Sd, R, SR, Rpp]
    tvec : unused (kept for API symmetry with your FN version)
    eps : small float to guard division by zero in Km + Rpp

    Returns
    -------
    dxdt : ndarray, shape (N, 5)
        [dS/dt, dSd/dt, dR/dt, dSR/dt, dRpp/dt]
    """
    k1, k2, k3, k4, V, Km = np.asarray(theta, dtype=float)
    S, Sd, R, SR, Rpp = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]

    denom = Km + Rpp
    denom = np.where(denom > 0.0, denom, eps)

    dS   = -k1*S - k2*S*R + k3*SR
    dSd  =  k1*S
    dR   = -k2*S*R + k3*SR + V*Rpp/denom
    dSR  =  k2*S*R - k3*SR - k4*SR
    dRpp =  k4*SR   - V*Rpp/denom

    return np.stack([dS, dSd, dR, dSR, dRpp], axis=1)



def PTrans_pretrain(penalty, extended_obs, t, model, train_generator, variable_batch_size=10, train_epochs = 5000):
    model_copy = copy.deepcopy(model)
    best_model_copy = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model_copy.parameters(), lr=9e-3)
    n = len(t)
    y_ind = np.arange(n)
    
    loss_history = []
    for epoch in range(train_epochs):
        np.random.shuffle(y_ind)
        epoch_loss = 0.0
        batch_loss = 0.0
        model_copy.train()
        optimizer.zero_grad()
        for i in range(0, n, variable_batch_size):
            variable_batch_id = y_ind[i:(i + variable_batch_size)]
            # optimizer.zero_grad()
            batch_loss = model_copy.compute_loss(
                derivative_batch_t=[s.reshape(-1, 1) for s in train_generator.get_examples()],  # list([100, 1])
                variable_batch_t=[t[variable_batch_id].view(-1, 1)],  # list([10, 1])
                batch_y=extended_obs[variable_batch_id],  # [10, 5]
                derivative_weight=penalty)  # 0.05
            batch_loss.backward()
            epoch_loss += batch_loss.item()
        if epoch % 100 == 0:
            print(f'Train Epoch: {epoch} '
                    f'[{epoch}/{train_epochs} '
                    f'\tLoss: {batch_loss.item():.6f}')
        optimizer.step()
        loss_history.append(epoch_loss)
        if loss_history[-1] == min(loss_history):
            best_model_copy.load_state_dict(model_copy.state_dict())

    return best_model_copy.state_dict()
    

def PTrans_CV(penalty, obs, t, model, train_generator, train_idx, val_idx, variable_batch_size = 10, train_epochs = 10000):
    model_copy = copy.deepcopy(model)
    best_model_copy = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model_copy.parameters(), lr=9e-3)
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
    ydataTruth = [[1, 0.588261834720057, 0.405587021811379,
                0.233954596382738, 0.185824926227245, 0.121529475508475, 0.0660579216704765,
                0.0232239721559163, 0.00753621476608807, 0.000635757067732186,
                4.4828522151875e-05, 2.92691291637857e-06, 1.85430809432099e-07,
                7.28853967992039e-10, 2.90513174227738e-12],
                [0, 0.053266895650711,
                0.0873622910225387, 0.130427267370046, 0.145032917209717, 0.166173447332274,
                0.185270502887831, 0.199691529407793, 0.204604196852704, 0.20659618691378,
                0.206753576566759, 0.206764363427542, 0.206765059920321, 0.206765106622966,
                0.206765106806669],
                [1, 0.642586847997489, 0.498289607509476,
                0.384851880112798, 0.360672689559933, 0.337963962897698, 0.334437371299282,
                0.362606647434368, 0.408318304747127, 0.512250740799807, 0.61245271751103,
                0.702776887221291, 0.78106230356887, 0.896447938708228, 0.958939507477765
                ],
                [0, 0.301777886330572, 0.349662193053065, 0.28406917802038,
                0.239159189174826, 0.162847399043611, 0.0890984548705512, 0.0329795416265298,
                0.0122844593001908, 0.00151121723113409, 0.000149977389483994,
                1.26910389636527e-05, 9.71682989611335e-07, 4.82588798220601e-09,
                2.14807760018722e-11],
                [0, 0.0556352656719387, 0.152048199437459,
                0.331078941866822, 0.400168121265241, 0.499188638058692, 0.576464173830167,
                0.604413810939102, 0.579397235952683, 0.48623804196906, 0.387397305099487,
                0.297210421739746, 0.218936724748142, 0.103552056465885, 0.0410604925007539
                ]]
    ydataTruth = np.array(ydataTruth).transpose()
    theta_true = torch.tensor([0.07, 0.6,0.05,0.3,0.017,0.3])
    # run 100 simulations
    if args.true_sigma == 1e-2:
        sigma_cha = "001"
    else:
        sigma_cha = "0001"

    SEED = pd.read_table(f"./Experiments/PTrans_noise{sigma_cha}_seed.txt", delim_whitespace=True, header=None)
    SEED = torch.tensor(data=SEED.values, dtype=torch.int)
    n = 101
    tvecObs = [0, 1, 2, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60, 80, 100]
    tvecFull = np.linspace(0, 100, num=n)
    
    s = args.seed

    np.random.seed(SEED[s, 0].data)
    torch.manual_seed(SEED[s, 0].data)

    sci_str = format(args.true_sigma, ".0e")
    penalty = format(args.penalty, ".1e").replace(".", "_")
    
    output_dir = f"../depot_hyun/hyun/ODE_param/PTrans_sig{sci_str}/lambda_{penalty}_CV"
    os.makedirs(f"{output_dir}/ydata", exist_ok=True)
    os.makedirs(f"{output_dir}/results", exist_ok=True)
    
    ydata = ydataTruth + np.random.normal(0, args.true_sigma, ydataTruth.shape)  # [15, 5]
    np.save(f"{output_dir}/ydata/ydata_{s}.npy", ydata)
    
    ydataFull = np.zeros((n, 5))
    for j in range(5):
        ydataFull[:, j] = np.interp(tvecFull, tvecObs, ydata[:, j])  # [101, 5]
    t = torch.linspace(0., 100., n)  # torch.float32
    y_interpolate = torch.from_numpy(ydataFull)  # torch.float64
    
    t_min = 0.0
    t_max = 100.0
    variable_batch_size = 10
    derivative_batch_size = 1000
    train_generator = SamplerGenerator(
        Generator1D(size=derivative_batch_size, t_min=t_min, t_max=t_max, method='equally-spaced-noisy'))
    model = BaseSolver(diff_eqs=ODESystem(),
                       net1=FCNN(n_input_units=1, n_output_units=1, actv=nn.Tanh),
                       net2=FCNN(n_input_units=1, n_output_units=1, actv=nn.Tanh),
                       net3=FCNN(n_input_units=1, n_output_units=1, actv=nn.Tanh),
                       net4=FCNN(n_input_units=1, n_output_units=1, actv=nn.Tanh),
                       net5=FCNN(n_input_units=1, n_output_units=1, actv=nn.Tanh))
    best_model = copy.deepcopy(model)

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=2726)
    start_time = time.time()
    cumulative_time = 0
    CV_l2_error = 0
    CV_deri_error = 0
    num = 0
    
    # pretrain
    model_pretrain = copy.deepcopy(model)
    pretrain_dict = PTrans_pretrain(args.penalty, y_interpolate, t, model_pretrain, train_generator, variable_batch_size = 10, train_epochs = 50) #5000
    model_pretrain.load_state_dict(pretrain_dict)
    
    for train_idx, val_idx in kfold.split(ydata):
        print(f"penalty: {args.penalty}, CV: {num}/{k_folds}")
        results = PTrans_CV(args.penalty, torch.from_numpy(ydata), t, model_pretrain, train_generator, train_idx, val_idx, variable_batch_size = 10, train_epochs = 100) #10000
        num+=1

        CV_deri_error += results[1] 
        CV_l2_error += results[2]

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
    parser.add_argument("--true_sigma", type = float, default = 0.01,
                        help = "observation errors (default: 0.01)")
    parser.add_argument("--penalty", type = float, default = 0.1,
                        help = "observation errors (default: 0.1)")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)
