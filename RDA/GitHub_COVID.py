import numpy as np
import torch
import torch.nn as nn
from abc import ABC
from scipy.io import loadmat
from solvers_utils import PretrainedSolver
from networks import FCNN
from generators import SamplerGenerator, Generator1D
from neurodiffeq import safe_diff as diff


class ODESystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = 3.4e-05
        self.Lambda = 11400
        self.deltadh = 1.0e-4
        self.deltaoh = 5.0e-5
        self.deltad1 = 2.5e-5
        self.deltao1 = 1.3e-5
        self.deltad2 = 5.0e-5
        self.deltao2 = 2.5e-5
        self.rhod = 1.2e-4
        self.rhoo = 1.2e-4
        self.alphad1 = 1 / 5
        self.alphao1 = 1 / 5
        self.omegavf = 1 / 274
        self.omegavb = 1 / 365
        self.omegadr = 1 / 274
        self.omegaor = 1 / 274
        self.sigmade = 1 / 5
        self.sigmadp = 1 / 2
        self.sigmaoe = 1 / 3
        self.sigmaop = 1 / 2
        self.rd = 0.20
        self.ro = 0.20
        self.epsilondf = 0.95
        self.epsilonof = 0.88
        self.epsilondb = 0.95
        self.epsilonob = 0.755
        self.phid1 = 1 / 5
        self.phio1 = 1 / 5
        self.phid2 = 1 / 3
        self.phio2 = 1 / 3
        self.gammada = 1 / 5
        self.gammaoa = 2 / 5
        self.gammadh = 1 / 5
        self.gammaoh = 1 / 3.4  # 2/5
        self.gammad2 = 1 / 4
        self.gammao2 = 2 / 4
        self.gammadq = 1 / 5
        self.gammaoq = 2 / 5
        self.psid = 1 / 3
        self.psio = 1 / 3
        self.taud1 = 0
        self.tauo1 = 0
        self.taud2 = 0
        self.tauo2 = 0
        self.taudh = 0
        self.tauoh = 0
        self.xivf = nn.Parameter(torch.logit(torch.tensor(0.006)))
        self.xivb = nn.Parameter(torch.logit(torch.tensor(0.001)))
        self.betadp = nn.Parameter(torch.logit(torch.tensor(0.3)))
        self.betada = nn.Parameter(torch.logit(torch.tensor(0.003)))
        self.betad1 = nn.Parameter(torch.logit(torch.tensor(0.025)))
        self.betad2 = nn.Parameter(torch.logit(torch.tensor(0.0006)))
        self.betadq = nn.Parameter(torch.logit(torch.tensor(8e-06)))
        self.betadh = nn.Parameter(torch.logit(torch.tensor(3.5e-06)))
        self.betaop = nn.Parameter(torch.logit(torch.tensor(1.0)))
        self.betaoa = nn.Parameter(torch.logit(torch.tensor(0.0096)))
        self.betao1 = nn.Parameter(torch.logit(torch.tensor(0.074)))
        self.betao2 = nn.Parameter(torch.logit(torch.tensor(0.0017)))
        self.betaoq = nn.Parameter(torch.logit(torch.tensor(2.5e-05)))
        self.betaoh = nn.Parameter(torch.logit(torch.tensor(1.1e-05)))
        self.y0 = nn.Parameter(torch.tensor([244961714.063182,  # S
                                             252794,  # Vf
                                             807346,  # Vb
                                             2e6,  # Ed
                                             2.01e6,  # Pd
                                             187740 * 0.1,  # Ad
                                             187740 * 0.1,  # Qd
                                             187740 * 0.89,  # Id1
                                             187740 * 0.5,  # Id2
                                             53000 * 3,  # Hd
                                             205371.346920750,  # Rd
                                             251967.362850553,  # Eo
                                             0e6,  # Po
                                             0 * 100000,  # Ao
                                             1,  # Qo
                                             0000000,  # Io1
                                             0000000,  # Io2
                                             0000000,  # Ho
                                             0000000,  # Ro
                                             48560513]))
        self.a_s = observed_y0[0]
        self.b_s = 1e6
        self.a_vf = observed_y0[1]
        self.b_vf = 1e6
        self.a_vb = observed_y0[2]
        self.b_vb = 1e6
        self.a_ed = observed_y0[3]
        self.b_ed = 1e6
        self.a_pd = observed_y0[4]
        self.b_pd = 1e6
        self.a_ad = observed_y0[5]
        self.b_ad = 1e6
        self.a_qd = observed_y0[6]
        self.b_qd = 1e6
        self.a_id1 = observed_y0[7]
        self.b_id1 = 1e6
        self.a_id2 = observed_y0[8]
        self.b_id2 = 1e6
        self.a_hd = observed_y0[9]
        self.b_hd = 1e6
        self.a_rd = observed_y0[10]
        self.b_rd = 1e6
        self.a_eo = observed_y0[11]
        self.b_eo = 1e6
        self.a_po = observed_y0[12]
        self.b_po = 1e6
        self.a_ao = observed_y0[13]
        self.b_ao = 1e6
        self.a_qo = observed_y0[14]
        self.b_qo = 1e6
        self.a_io1 = observed_y0[15]
        self.b_io1 = 1e6
        self.a_io2 = observed_y0[16]
        self.b_io2 = 1e6
        self.a_ho = observed_y0[17]
        self.b_ho = 1e6
        self.a_ro = observed_y0[18]
        self.b_ro = 1e6
        self.standardize_parameters = [self.a_s, self.b_s, self.a_vf, self.b_vf, self.a_vb, self.b_vb, self.a_ed,
                                       self.b_ed, self.a_pd, self.b_pd, self.a_ad, self.b_ad, self.a_qd, self.b_qd,
                                       self.a_id1, self.b_id1, self.a_id2, self.b_id2, self.a_hd, self.b_hd, self.a_rd,
                                       self.b_rd, self.a_eo, self.b_eo, self.a_po, self.b_po, self.a_ao, self.b_ao,
                                       self.a_qo, self.b_qo, self.a_io1, self.b_io1, self.a_io2, self.b_io2, self.a_ho,
                                       self.b_ho, self.a_ro, self.b_ro]

    def compute_derivative(self, new_S, new_Vf, new_Vb, new_Ed, new_Pd, new_Ad, new_Qd, new_Id1, new_Id2, new_Hd,
                           new_Rd, new_Eo, new_Po, new_Ao, new_Qo, new_Io1, new_Io2, new_Ho, new_Ro, t):
        """u.shape = [batch, 1]
        t.shape = [batch, 1]
        """
        xivf_limited = torch.sigmoid(self.xivf)
        xivb_limited = torch.sigmoid(self.xivb)
        betadp_limited = torch.sigmoid(self.betadp)
        betada_limited = torch.sigmoid(self.betada)
        betad1_limited = torch.sigmoid(self.betad1)
        betad2_limited = torch.sigmoid(self.betad2)
        betadq_limited = torch.sigmoid(self.betadq)
        betadh_limited = torch.sigmoid(self.betadh)
        betaop_limited = torch.sigmoid(self.betaop)
        betaoa_limited = torch.sigmoid(self.betaoa)
        betao1_limited = torch.sigmoid(self.betao1)
        betao2_limited = torch.sigmoid(self.betao2)
        betaoq_limited = torch.sigmoid(self.betaoq)
        betaoh_limited = torch.sigmoid(self.betaoh)
        S = self.b_s * new_S + self.a_s
        Vf = self.b_vf * new_Vf + self.a_vf
        Vb = self.b_vb * new_Vb + self.a_vb
        Ed = self.b_ed * new_Ed + self.a_ed
        Pd = self.b_pd * new_Pd + self.a_pd
        Ad = self.b_ad * new_Ad + self.a_ad
        Qd = self.b_qd * new_Qd + self.a_qd
        Id1 = self.b_id1 * new_Id1 + self.a_id1
        Id2 = self.b_id2 * new_Id2 + self.a_id2
        Hd = self.b_hd * new_Hd + self.a_hd
        Rd = self.b_rd * new_Rd + self.a_rd
        Eo = self.b_eo * new_Eo + self.a_eo
        Po = self.b_po * new_Po + self.a_po
        Ao = self.b_ao * new_Ao + self.a_ao
        Qo = self.b_qo * new_Qo + self.a_qo
        Io1 = self.b_io1 * new_Io1 + self.a_io1
        Io2 = self.b_io2 * new_Io2 + self.a_io2
        Ho = self.b_ho * new_Ho + self.a_ho
        Ro = self.b_ro * new_Ro + self.a_ro
        Ns = S + Vf + Vb
        Nd = Ed + Pd + Ad + Id1 + Id2 + Qd + Hd + Rd
        No = Eo + Po + Ao + Io1 + Io2 + Qo + Ho + Ro
        N = Ns + Nd + No
        lambdad = (betadp_limited * Pd + betada_limited * Ad + betad1_limited * Id1 + betad2_limited * Id2
                   + betadq_limited * Qd + betadh_limited * Hd) / N
        lambdao = (betaop_limited * Po + betaoa_limited * Ao + betao1_limited * Io1 + betao2_limited * Io2
                   + betaoq_limited * Qo + betaoh_limited * Ho) / N
        lambdadf = (1 - self.epsilondf) * (betadp_limited * Pd + betada_limited * Ad + betad1_limited * Id1
                                           + betad2_limited * Id2 + betadq_limited * Qd + betadh_limited * Hd) / N
        lambdaof = (1 - self.epsilonof) * (betaop_limited * Po + betaoa_limited * Ao + betao1_limited * Io1
                                           + betao2_limited * Io2 + betaoq_limited * Qo + betaoh_limited * Ho) / N
        lambdadb = (1 - self.epsilondb) * (betadp_limited * Pd + betada_limited * Ad + betad1_limited * Id1
                                           + betad2_limited * Id2 + betadq_limited * Qd + betadh_limited * Hd) / N
        lambdaob = (1 - self.epsilonob) * (betaop_limited * Po + betaoa_limited * Ao + betao1_limited * Io1
                                           + betao2_limited * Io2 + betaoq_limited * Qo + betaoh_limited * Ho) / N
        Sdot = self.Lambda + self.omegavf * Vf + self.omegavb * Vb + self.omegadr * Rd + self.omegaor * Ro \
               - (lambdad + lambdao + xivf_limited + self.mu) * S
        new_Sdot = Sdot / self.b_s
        Vfdot = xivf_limited * S - (lambdadf + lambdaof + xivb_limited + self.omegavf + self.mu) * Vf
        new_Vfdot = Vfdot / self.b_vf
        Vbdot = xivb_limited * Vf - (lambdadb + lambdaob + self.omegavb + self.mu) * Vb
        new_Vbdot = Vbdot / self.b_vb
        Eddot = lambdad * S + lambdadf * Vf + lambdadb * Vb - (self.sigmade + self.rhod + self.mu) * Ed
        new_Eddot = Eddot / self.b_ed
        Pddot = self.sigmade * Ed - (self.sigmadp + self.rhod + self.mu) * Pd
        new_Pddot = Pddot / self.b_pd
        Addot = (1 - self.rd) * self.sigmadp * Pd - (self.gammada + self.rhod + self.mu) * Ad
        new_Addot = Addot / self.b_ad
        Qddot = self.rhod * (Ed + Pd + Ad) - (self.gammadq + self.psid + self.mu) * Qd
        new_Qddot = Qddot / self.b_qd
        Id1dot = self.rd * self.sigmadp * Pd + self.psid * Qd - (self.taud1 + self.alphad1 + self.phid1
                                                                 + self.mu + self.deltad1) * Id1
        new_Id1dot = Id1dot / self.b_id1
        Id2dot = self.alphad1 * Id1 - (self.taud2 + self.gammad2 + self.phid2 + self.mu + self.deltad2) * Id2
        new_Id2dot = Id2dot / self.b_id2
        Hddot = self.phid1 * Id1 + self.phid2 * Id2 - (self.taudh + self.gammadh + self.mu + self.deltadh) * Hd
        new_Hddot = Hddot / self.b_hd
        Rddot = self.gammada * Ad + self.gammadq * Qd + self.taud1 * Id1 + (self.taud2 + self.gammad2) * Id2 \
                + (self.taudh + self.gammadh) * Hd - (self.omegadr + self.mu) * Rd
        new_Rddot = Rddot / self.b_rd
        Eodot = lambdao * S + lambdaof * Vf + lambdaob * Vb - (self.sigmaoe + self.rhoo + self.mu) * Eo
        new_Eodot = Eodot / self.b_eo
        Podot = self.sigmaoe * Eo - (self.sigmaop + self.rhoo + self.mu) * Po
        new_Podot = Podot / self.b_po
        Aodot = (1 - self.ro) * self.sigmaop * Po - (self.gammaoa + self.rhoo + self.mu) * Ao
        new_Aodot = Aodot / self.b_ao
        Qodot = self.rhoo * (Eo + Po + Ao) - (self.gammaoq + self.psio + self.mu) * Qo
        new_Qodot = Qodot / self.b_qo
        Io1dot = self.ro * self.sigmaop * Po + self.psio * Qo - (self.tauo1 + self.alphao1 + self.phio1
                                                                 + self.mu + self.deltao1) * Io1
        new_Io1dot = Io1dot / self.b_io1
        Io2dot = self.alphao1 * Io1 - (self.tauo2 + self.gammao2 + self.phio2 + self.mu + self.deltao2) * Io2
        new_Io2dot = Io2dot / self.b_io2
        Hodot = self.phio1 * Io1 + self.phio2 * Io2 - (self.tauoh + self.gammaoh + self.mu + self.deltaoh) * Ho
        new_Hodot = Hodot / self.b_ho
        Rodot = self.gammaoa * Ao + self.gammaoq * Qo + self.tauo1 * Io1 + (self.tauo2 + self.gammao2) * Io2 \
                + (self.tauoh + self.gammaoh) * Ho - (self.omegaor + self.mu) * Ro
        new_Rodot = Rodot / self.b_ro
        return [diff(new_S, t) - new_Sdot, diff(new_Vf, t) - new_Vfdot, diff(new_Vb, t) - new_Vbdot,
                diff(new_Ed, t) - new_Eddot, diff(new_Pd, t) - new_Pddot, diff(new_Ad, t) - new_Addot,
                diff(new_Qd, t) - new_Qddot, diff(new_Id1, t) - new_Id1dot, diff(new_Id2, t) - new_Id2dot,
                diff(new_Hd, t) - new_Hddot, diff(new_Rd, t) - new_Rddot, diff(new_Eo, t) - new_Eodot,
                diff(new_Po, t) - new_Podot, diff(new_Ao, t) - new_Aodot, diff(new_Qo, t) - new_Qodot,
                diff(new_Io1, t) - new_Io1dot, diff(new_Io2, t) - new_Io2dot, diff(new_Ho, t) - new_Hodot,
                diff(new_Ro, t) - new_Rodot]

    def compute_func_val(self, nets, derivative_batch_t):
        t_0 = 0.0
        rslt = []
        for idx, net in enumerate(nets):
            u_0 = (self.y0[idx] - self.standardize_parameters[2 * idx]) / self.standardize_parameters[2 * idx + 1]
            network_output = net(torch.cat(derivative_batch_t, dim=1))
            new_network_output = u_0 + (1 - torch.exp(-torch.cat(derivative_batch_t, dim=1) + t_0)) * network_output
            rslt.append(new_network_output)
        return rslt


class BaseSolver(ABC, PretrainedSolver, nn.Module):
    def __init__(self, diff_eqs, net1, net2, net3, net4, net5, net6, net7, net8, net9, net10, net11, net12,
                 net13, net14, net15, net16, net17, net18, net19):
        super().__init__()
        self.diff_eqs = diff_eqs
        self.net1 = net1
        self.net2 = net2
        self.net3 = net3
        self.net4 = net4
        self.net5 = net5
        self.net6 = net6
        self.net7 = net7
        self.net8 = net8
        self.net9 = net9
        self.net10 = net10
        self.net11 = net11
        self.net12 = net12
        self.net13 = net13
        self.net14 = net14
        self.net15 = net15
        self.net16 = net16
        self.net17 = net17
        self.net18 = net18
        self.net19 = net19
        self.nets = [net1, net2, net3, net4, net5, net6, net7, net8, net9, net10, net11, net12, net13, net14, net15,
                     net16, net17, net18, net19]

    def compute_loss(self, derivative_batch_t, variable_batch_t, batch_y, derivative_weight=0.5):
        """derivative_batch_t can be sampled in any distribution and sample size.
        derivative_batch_t = list(torch.Size([64, 1]))
        variable_batch_t = list(torch.Size([65, 1]))
        batch_y = torch.Size([65, 1])
        """
        derivative_loss = 0.0
        derivative_funcs = self.diff_eqs.compute_func_val(self.nets, derivative_batch_t)
        # len(derivative_residuals) = number of variables
        derivative_residuals = self.diff_eqs.compute_derivative(*derivative_funcs,
                                                                *derivative_batch_t)  # list([32, 1], [32, 1])
        derivative_residuals = torch.cat(derivative_residuals, dim=1)  # [32, 2]
        derivative_loss += (derivative_residuals ** 2).mean()
        """(variable_batch_t, batch_y) is sampled from data
         variable_batch_t[0].shape = [10, 1]
        batch_y.shape = [10, 1]
        """
        variable_loss = 0.0
        variable_funcs = self.diff_eqs.compute_func_val(self.nets, variable_batch_t)
        variable_funcs = torch.cat(variable_funcs, dim=1)  # shape = [10, 19]
        DC_funcs = 1.2e-4 * (variable_funcs[:, 3] + variable_funcs[:, 4] + variable_funcs[:, 5]) + \
                   0.2 * 0.5 * variable_funcs[:, 4] + \
                   1.2e-4 * (variable_funcs[:, 11] + variable_funcs[:, 12] + variable_funcs[:, 13]) + \
                   0.2 * 0.5 * variable_funcs[:, 12]
        variable_loss += ((DC_funcs.view(-1, 1) - batch_y) ** 2).mean()
        return derivative_weight * derivative_loss + variable_loss


n = 65
train_t = torch.linspace(678, 742, n)
new_train_t = (train_t - min(train_t)) / (max(train_t) - min(train_t))
covid = loadmat('COVID19Data.mat')
covid_data = covid['ObtainCOVID19Data']
DailyCaseData = torch.tensor(data=covid_data[677:742 + 46, 4], dtype=torch.float)
CumulativeCaseData = torch.tensor(data=covid_data[677:742 + 46, 1], dtype=torch.float)
observed_y = DailyCaseData.view(-1, 1)
train_y = observed_y[0:n]
observed_y0 = torch.tensor([244961714.063182,  # S
                            252794,  # Vf
                            807346,  # Vb
                            2e6,  # Ed
                            2.01e6,  # Pd
                            187740 * 0.1,  # Ad
                            187740 * 0.1,  # Qd
                            187740 * 0.89,  # Id1
                            187740 * 0.5,  # Id2
                            53000 * 3,  # Hd
                            205371.346920750,  # Rd
                            251967.362850553,  # Eo
                            0e6,  # Po
                            0 * 100000,  # Ao
                            1,  # Qo
                            0000000,  # Io1
                            0000000,  # Io2
                            0000000,  # Ho
                            0000000,  # Ro
                            48560513])  # DC
observed_DC0 = 1.2e-4 * (observed_y0[3] + observed_y0[4] + observed_y0[5]) + 0.2 * 0.5 * observed_y0[4] \
               + 1.2e-4 * (observed_y0[11] + observed_y0[12] + observed_y0[13]) + 0.2 * 0.5 * observed_y0[12]
new_train_y = (train_y - observed_DC0) / 1e6

seed = 578271369
torch.manual_seed(seed)
np.random.seed(seed)
t_min = 0
t_max = 1
variable_batch_size = 65
derivative_batch_size = 64
train_generator = SamplerGenerator(
    Generator1D(size=derivative_batch_size, t_min=t_min, t_max=t_max, method='equally-spaced-noisy'))
model = BaseSolver(diff_eqs=ODESystem(),
                   net1=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                   net2=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                   net3=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                   net4=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                   net5=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                   net6=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                   net7=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                   net8=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                   net9=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                   net10=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                   net11=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                   net12=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                   net13=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                   net14=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                   net15=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                   net16=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                   net17=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                   net18=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                   net19=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh))
best_model = BaseSolver(diff_eqs=ODESystem(),
                        net1=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                        net2=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                        net3=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                        net4=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                        net5=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                        net6=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                        net7=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                        net8=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                        net9=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                        net10=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                        net11=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                        net12=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                        net13=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                        net14=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                        net15=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                        net16=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                        net17=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                        net18=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh),
                        net19=FCNN(n_input_units=1, n_output_units=1, hidden_units=[64, 64], actv=nn.Tanh))
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
y_ind = np.arange(n)
train_epochs = 10000
loss_history = []

for epoch in range(train_epochs):
    np.random.shuffle(y_ind)
    epoch_loss = 0.0
    batch_loss = 0.0
    optimizer.zero_grad()
    for i in range(0, n, variable_batch_size):
        variable_batch_id = y_ind[i:(i + variable_batch_size)]
        batch_loss = model.compute_loss(derivative_batch_t=[s.reshape(-1, 1) for s in train_generator.get_examples()],
                                        variable_batch_t=[new_train_t[variable_batch_id].view(-1, 1)],  # list([10, 1])
                                        batch_y=new_train_y[variable_batch_id],  # [10, 2]
                                        derivative_weight=1e-4)
        batch_loss.backward()
        epoch_loss += batch_loss.item()
    optimizer.step()
    loss_history.append(epoch_loss)
    if loss_history[-1] == min(loss_history):
        best_model.load_state_dict(model.state_dict())
    if epoch % 500 == 0:
        print(f'Train Epoch: {epoch} '
              f'\tLoss: {epoch_loss:.6f}')

# predict on test data
test_t = torch.linspace(678, 742 + 46, 65 + 46)
new_test_t = (test_t - min(train_t)) / (max(train_t) - min(train_t))
with torch.no_grad():
    test_estimate_funcs = best_model.diff_eqs.compute_func_val(best_model.nets, [new_test_t.view(-1, 1)])
    test_estimate_funcs = torch.cat(test_estimate_funcs, dim=1)
test_DC = 1.2e-4 * (test_estimate_funcs[:, 3] + test_estimate_funcs[:, 4] + test_estimate_funcs[:, 5]) \
          + 0.2 * 0.5 * test_estimate_funcs[:, 4] \
          + 1.2e-4 * (test_estimate_funcs[:, 11] + test_estimate_funcs[:, 12] + test_estimate_funcs[:, 13]) \
          + 0.2 * 0.5 * test_estimate_funcs[:, 12]
predict_DC = test_DC * 1e6 + observed_DC0
predict_cumulative = torch.zeros((111, ))
predict_cumulative[0] = CumulativeCaseData[0]
for j in range(1, 111):
    predict_cumulative[j] = predict_cumulative[j-1] + predict_DC[j-1]
