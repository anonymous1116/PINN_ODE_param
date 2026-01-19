import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from abc import ABC
from solvers_utils import PretrainedSolver
from networks import FCNN
from generators import SamplerGenerator, Generator1D
from neurodiffeq import safe_diff as diff
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
         variable_batch_t = list([variable_batch_size, 1])
        batch_y.shape = [variable_batch_size, 1]
        """
        variable_loss = 0.0
        variable_funcs = self.diff_eqs.compute_func_val(self.nets, variable_batch_t)
        variable_funcs = torch.cat(variable_funcs, dim=1)  # [10, 5]
        variable_loss += ((variable_funcs - batch_y) ** 2).mean()
        return derivative_weight * derivative_loss + variable_loss


def fOde(t, y, theta, eps=1e-12):
    """
    RHS f(t, y; theta) for the 5-state system.
    y: array-like (5,) -> [S, Sd, R, SR, Rpp]
    theta: dict with keys {'k1','k2','k3','k4','V','Km'} or a tuple in that order
    returns dy/dt as np.ndarray (5,)
    """
    if isinstance(theta, dict):
        k1, k2, k3, k4, V, Km = theta['k1'], theta['k2'], theta['k3'], theta['k4'], theta['V'], theta['Km']
    else:
        k1, k2, k3, k4, V, Km = theta  # assume tuple/list in this order

    S, Sd, R, SR, Rpp = y

    denom = Km + Rpp
    denom = denom if denom > 0 else eps  # guard against division by zero

    dS   = -k1*S - k2*S*R + k3*SR
    dSd  =  k1*S
    dR   = -k2*S*R + k3*SR + V*Rpp/denom
    dSR  =  k2*S*R - k3*SR - k4*SR
    dRpp =  k4*SR   - V*Rpp/denom

    return np.array([dS, dSd, dR, dSR, dRpp], dtype=float)



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
    elif args.true_sigma == 1e-3:
        sigma_cha = "0001"     
    #else:
    #    sigma_cha = "001"

    SEED = pd.read_table(f"./Experiments/PTrans_noise{sigma_cha}_seed.txt", delim_whitespace=True, header=None)
    SEED = torch.tensor(data=SEED.values, dtype=torch.int)
    n = 101
    tvecObs = [0, 1, 2, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60, 80, 100]
    tvecFull = np.linspace(0, 100, num=n)
    ydataTruthFull = np.zeros((n, 5))
    for j in range(5):
        ydataTruthFull[:, j] = np.interp(tvecFull, tvecObs, ydataTruth[:, j])
    trajectory_RMSE = np.zeros((100, 5))
    trajectory = np.zeros((100, n, 5))


    s = args.seed

    np.random.seed(SEED[s, 0].data)
    torch.manual_seed(SEED[s, 0].data)
    ydata = ydataTruth + np.random.normal(0, args.true_sigma, ydataTruth.shape)  # [15, 5]
    ydataFull = np.zeros((n, 5))
    for j in range(5):
        ydataFull[:, j] = np.interp(tvecFull, tvecObs, ydata[:, j])  # [101, 5]
    t = torch.linspace(0., 100., n)  # torch.float32
    true_y = torch.from_numpy(ydataFull)  # torch.float64
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
    best_model = BaseSolver(diff_eqs=ODESystem(),
                            net1=FCNN(n_input_units=1, n_output_units=1, actv=nn.Tanh),
                            net2=FCNN(n_input_units=1, n_output_units=1, actv=nn.Tanh),
                            net3=FCNN(n_input_units=1, n_output_units=1, actv=nn.Tanh),
                            net4=FCNN(n_input_units=1, n_output_units=1, actv=nn.Tanh),
                            net5=FCNN(n_input_units=1, n_output_units=1, actv=nn.Tanh))
    optimizer = torch.optim.Adam(model.parameters(), lr=9e-3)  # 12e-3
    y_ind = np.arange(n)
    train_epochs = 2000
    loss_history = []
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
                variable_batch_t=[t[variable_batch_id].view(-1, 1)],  # list([10, 1])
                batch_y=true_y[variable_batch_id],  # [10, 5]
                derivative_weight=0.07)  # 0.05
            batch_loss.backward()
            epoch_loss += batch_loss.item()
        if epoch % 100 == 0:
            print(f'Train Epoch: {epoch} '
                    f'[{epoch}/{train_epochs} '
                    f'\tLoss: {batch_loss.item():.6f}')
        optimizer.step()
        loss_history.append(epoch_loss)
        if loss_history[-1] == min(loss_history):
            best_model.load_state_dict(model.state_dict())

    # check estimated path using 101 points
    with torch.no_grad():
        estimate_t = torch.linspace(0., 100., n)
        estimate_funcs = best_model.diff_eqs.compute_func_val(best_model.nets, [estimate_t.view(-1, 1)])
        estimate_funcs = torch.cat(estimate_funcs, dim=1).numpy()

        estimate_t_1000 = torch.linspace(0., 100., 1001)
        estimate_funcs_1000 = best_model.diff_eqs.compute_func_val(best_model.nets, [estimate_t_1000.view(-1, 1)])
        estimate_funcs_1000 = torch.cat(estimate_funcs_1000, dim=1).numpy()

    trajectory_RMSE = np.sqrt(np.mean((estimate_funcs-ydataTruthFull)**2, axis=0))
    trajectory[s, :, :] = estimate_funcs
    print(f"Simulation {s} finished")
    print(trajectory_RMSE)

    true_trajectory_100 = pd.read_table(f"../depot_hyun/hyun/ODE_param/PTrans_trajectory_100.txt", header=None)
    true_trajectory_100 = torch.tensor(true_trajectory_100.to_numpy()[:,1:6], dtype = torch.float32)
    #estimate_funcs = torch.tensor(estimate_funcs, dtype = torch.flaot32)
    trajectory_RMSE_100 = np.sqrt(np.mean((estimate_funcs-true_trajectory_100.numpy())**2, axis=0))
    
    true_trajectory_1000 = pd.read_table(f"../depot_hyun/hyun/ODE_param/PTrans_trajectory_1000.txt", header=None)
    true_trajectory_1000 = torch.tensor(true_trajectory_1000.to_numpy()[:,1:6], dtype = torch.float32)
    #estimate_funcs = torch.tensor(estimate_funcs, dtype = torch.flaot32)
    trajectory_RMSE_1000 = np.sqrt(np.mean((estimate_funcs_1000-true_trajectory_1000.numpy())**2, axis=0))
    

    # save
    sci_str = format(args.true_sigma, ".0e")
    output_dir = f"../depot_hyun/hyun/ODE_param/PTrans_base_{sci_str}"
    os.makedirs(f"{output_dir}/results", exist_ok=True)
    
    print(f"Simulation {s} finished")
    np.save(f"{output_dir}/results/trajectory_RMSE_{s}.npy", trajectory_RMSE)
    np.save(f"{output_dir}/results/trajectory_RMSE100_{s}.npy", trajectory_RMSE_100)
    np.save(f"{output_dir}/results/trajectory_RMSE1000_{s}.npy", trajectory_RMSE_1000)
    print(f"trajectory_RMSE: {trajectory_RMSE}", flush=True)
    print(f"trajectory_RMSE_100: {trajectory_RMSE_100}", flush=True)
    print(f"trajectory_RMSE_1000: {trajectory_RMSE_1000}", flush=True)
    
    os.makedirs(f"{output_dir}/ydata", exist_ok=True)
    np.save(f"{output_dir}/ydata/ydata_{s}.npy", ydata)
    

    S, Sd, R, SR, Rpp = best_model.diff_eqs.compute_func_val(best_model.nets, [estimate_t])
    param_results = torch.cat([S, Sd, R, SR, Rpp], dim=1)  # (N,5)
    
    dt = estimate_t[1] - estimate_t[0]

    val_term = np.sum((estimate_funcs - true_trajectory_100) ** 2) * dt
    print("val_term_part:", val_term)
    dX_hat = fOde(theta = param_results, x = estimate_funcs, tvec = estimate_t)
    dtrue = fOde(theta= theta_true, x=true_trajectory_100, tvec=estimate_t)

    der_term = np.sum((dX_hat  - dtrue) ** 2)* dt
    print("der_term_part:", der_term)
    h1_error = np.sqrt(val_term + der_term)
    print("h1_part: ", h1_error)
    print("h1_error: ", h1_error)
    

    #======================================= with only ydata ==============================#=============================
    #model2 = BaseSolver(diff_eqs=ODESystem(),
    #                   net1=FCNN(n_input_units=1, n_output_units=1, actv=nn.Tanh),
    #                   net2=FCNN(n_input_units=1, n_output_units=1, actv=nn.Tanh),
    #                   net3=FCNN(n_input_units=1, n_output_units=1, actv=nn.Tanh),
    #                   net4=FCNN(n_input_units=1, n_output_units=1, actv=nn.Tanh),
    #                   net5=FCNN(n_input_units=1, n_output_units=1, actv=nn.Tanh))
    model.load_state_dict(best_model.state_dict())
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 12e-3
    y_ind = np.arange(len(tvecObs))
    loss_history = []
    train_epochs = 10000
    
    for epoch in range(train_epochs):
        np.random.shuffle(y_ind)
        epoch_loss = 0.0
        batch_loss = 0.0
        # model.train()
        
        optimizer.zero_grad()
        for i in range(0, len(y_ind), variable_batch_size):
            variable_batch_id = y_ind[i:(i + variable_batch_size)]
            #print(f"derivative_batch_t: {[s.reshape(-1, 1) for s in train_generator.get_examples()]}")
            #print(f"variable_batch_t: {[torch.tensor(tvecObs,dtype = torch.float32)[variable_batch_id].view(-1, 1)]}")
            #print(f"batch_y: {ydata[variable_batch_id]}")
    
            batch_loss = model.compute_loss(
                derivative_batch_t=[s.reshape(-1, 1) for s in train_generator.get_examples()],  
                variable_batch_t=[torch.tensor(tvecObs,dtype = torch.float32)[variable_batch_id].view(-1, 1)], 
                batch_y=torch.from_numpy(ydata)[variable_batch_id],  # [10, 5]
                derivative_weight=0.07)  # 0.05
            batch_loss.backward()
            epoch_loss += batch_loss.item()
            #if i % 100 == 0:
            #     print(f'Train Epoch: {epoch} '
            #           f'[{i:05}/{n} '
            #           f'\tLoss: {batch_loss.item():.6f}')
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Train Epoch: {epoch} '
                f'[{epoch}/{train_epochs}] '
                f'\tLoss: {batch_loss.item():.6f}')
        optimizer.step()
        loss_history.append(epoch_loss)
        if loss_history[-1] == min(loss_history):
            best_model.load_state_dict(model.state_dict())


    # check estimated path using 101 points
    with torch.no_grad():
        estimate_t = torch.linspace(0., 100., n)
        estimate_funcs = best_model.diff_eqs.compute_func_val(best_model.nets, [estimate_t.view(-1, 1)])
        estimate_funcs = torch.cat(estimate_funcs, dim=1).numpy()

        estimate_t_1000 = torch.linspace(0., 100., 1001)
        estimate_funcs_1000 = best_model.diff_eqs.compute_func_val(best_model.nets, [estimate_t_1000.view(-1, 1)])
        estimate_funcs_1000 = torch.cat(estimate_funcs_1000, dim=1).numpy()

    #trajectory_RMSE = np.sqrt(np.mean((estimate_funcs-ydataTruthFull)**2, axis=0))
    #trajectory[s, :, :] = estimate_funcs
    #print(f"Simulation {s} finished")
    #print(trajectory_RMSE)

    true_trajectory_100 = pd.read_table(f"../depot_hyun/hyun/ODE_param/PTrans_trajectory_100.txt", header=None)
    true_trajectory_100 = torch.tensor(true_trajectory_100.to_numpy()[:,1:6], dtype = torch.float32)
    #estimate_funcs = torch.tensor(estimate_funcs, dtype = torch.flaot32)
    trajectory_RMSE_100 = np.sqrt(np.mean((estimate_funcs-true_trajectory_100.numpy())**2, axis=0))
    
    true_trajectory_1000 = pd.read_table(f"../depot_hyun/hyun/ODE_param/PTrans_trajectory_1000.txt", header=None)
    true_trajectory_1000 = torch.tensor(true_trajectory_1000.to_numpy()[:,1:6], dtype = torch.float32)
    #estimate_funcs = torch.tensor(estimate_funcs, dtype = torch.flaot32)
    trajectory_RMSE_1000 = np.sqrt(np.mean((estimate_funcs_1000-true_trajectory_1000.numpy())**2, axis=0))
    

    # save
    print(f"Simulation {s} finished")
    np.save(f"{output_dir}/results/trajectory_RMSE100_{s}_after.npy", trajectory_RMSE_100)
    np.save(f"{output_dir}/results/trajectory_RMSE1000_{s}_after.npy", trajectory_RMSE_1000)
    print(f"trajectory_RMSE_100_after: {trajectory_RMSE_100}", flush=True)
    print(f"trajectory_RMSE_1000_after: {trajectory_RMSE_1000}", flush=True)
    
    S, Sd, R, SR, Rpp = best_model.diff_eqs.compute_func_val(best_model.nets, [estimate_t])
    param_results = torch.cat([S, Sd, R, SR, Rpp], dim=1)  # (N,5)
    
    dt = estimate_t[1] - estimate_t[0]

    val_term = np.sum((estimate_funcs - true_trajectory_100) ** 2) * dt
    print("val_term_part:", val_term)
    dX_hat = fOde(theta = param_results, x = estimate_funcs, tvec = estimate_t)
    dtrue = fOde(theta= theta_true, x=true_trajectory_100, tvec=estimate_t)

    der_term = np.sum((dX_hat  - dtrue) ** 2)* dt
    print("der_term_part:", der_term)
    h1_error = np.sqrt(val_term + der_term)
    print("h1_error: ", h1_error)
    np.save(f"{output_dir}/results/h1_errors_{s}.npy", h1_error)


def get_args():
    parser = argparse.ArgumentParser(description="Run simulation with customizable parameters.")
    parser.add_argument("--seed", type = int, default = 1,
                        help = "See number (default: 1)")
    parser.add_argument("--true_sigma", type = float, default = 0.01,
                        help = "observation errors (default: 0.01)")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)
