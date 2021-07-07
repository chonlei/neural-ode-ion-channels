#!/usr/bin/env python3
import sys
import os
import argparse
import time
import numpy as np
from scipy.interpolate import interp1d

import torch
import torch.nn as nn

parser = argparse.ArgumentParser('IKr NN ODE real data error table')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--cached', action='store_true')
args = parser.parse_args()

from torchdiffeq import odeint

#device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# Set random seed
np.random.seed(0)
torch.manual_seed(0)

noise_sigma = 0.1

true_y0s = [torch.tensor([[1., 0.]]).to(device),  # what you get after holding at +40mV
            torch.tensor([[0., 1.]]).to(device)]  # (roughly) what you get after holding at -80mV

# B1.2 in https://physoc.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1113%2FJP275733&file=tjp12905-sup-0001-textS1.pdf#page=4
e = torch.tensor([-88.4]).to(device)  # assume we know
# https://github.com/CardiacModelling/FourWaysOfFitting/blob/master/method-3/cell-5-fit-3-run-001.txt
g = g_nnf = torch.tensor([0.133898199260611944]).to(device)  # assume we know

g_nn = g * 1.2  # just because we see a-gate gets to ~1.2 at some point (in prt V=50), so can absorb that into the g.
e_nnf = e - 5    # just because in pr4, at -90 mV, a-gates became negative, meaning e < -90mV; and only if adding an extra -5mV, a ~ [0, 1].

#
#
#
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

makedirs('table-2')


#
# Load data
#
raw_data1 = np.loadtxt('data/pr3-steady-activation-cell-5.csv', delimiter=',', skiprows=1)
time1 = raw_data1[:, 0]
time1_torch = torch.from_numpy(raw_data1[:, 0]).to(device)
current1 = raw_data1[:, 1]
voltage1 = raw_data1[:, 2]

raw_data2 = np.loadtxt('data/pr4-inactivation-cell-5.csv', delimiter=',', skiprows=1)
time2 = raw_data2[:, 0]
time2_torch = torch.from_numpy(raw_data2[:, 0]).to(device)
current2 = raw_data2[:, 1]
voltage2 = raw_data2[:, 2]

#
# Make filters
#
n_ms = 3
dt = 0.1  # ms
n_points = int(n_ms / dt)
change_pt1 = np.append([True], ~(voltage1[1:] != voltage1[:-1]))
cap_mask1 = np.copy(change_pt1)
for i in range(n_points):
    cap_mask1 = cap_mask1 & np.roll(change_pt1, i + 1)
# A bigger/final filter mask
extra_points = 20  # for numerical derivative or smoothing issue
mask1 = np.copy(cap_mask1)
for i in range(extra_points):
    mask1 = mask1 & np.roll(change_pt1, i + n_points + 1)
    mask1 = mask1 & np.roll(change_pt1, -i - 1)
change_pt2 = np.append([True], ~(voltage2[1:] != voltage2[:-1]))
cap_mask2 = np.copy(change_pt2)
for i in range(n_points):
    cap_mask2 = cap_mask2 & np.roll(change_pt2, i + 1)
# A bigger/final filter mask
extra_points = 20  # for numerical derivative or smoothing issue
mask2 = np.copy(cap_mask2)
for i in range(extra_points):
    mask2 = mask2 & np.roll(change_pt2, i + n_points + 1)
    mask2 = mask2 & np.roll(change_pt2, -i - 1)

prediction1 = np.loadtxt('data/ap-cell-5.csv', delimiter=',', skiprows=1)
timep1 = prediction1[:, 0]
timep1_torch = torch.from_numpy(prediction1[:, 0]).to(device)
currentp1 = prediction1[:, 1]
voltagep1 = prediction1[:, 2]

#
#
#
class Lambda(nn.Module):
    def __init__(self):
        super(Lambda, self).__init__()

        # https://github.com/CardiacModelling/FourWaysOfFitting/blob/master/method-3/cell-5-fit-3-run-001.txt
        self.p1 = 2.10551451120238317e-04
        self.p2 = 6.57994674459572992e-02
        self.p3 = 3.31717454417642909e-06
        self.p4 = 7.43102564328181336e-02
        self.p5 = 8.73243709432939552e-02
        self.p6 = 7.33380025549188515e-03
        self.p7 = 6.16551007196145754e-03
        self.p8 = 3.15741310933875322e-02

        self.unity = torch.tensor([1]).to(device)

    def set_fixed_form_voltage_protocol(self, t, v):
        # Regular time point voltage protocol time series
        self._t_regular = t
        self._v_regular = v
        self.__v = interp1d(t, v)

    def _v(self, t):
        return torch.from_numpy(self.__v([t.cpu().numpy()])).to(device)

    def voltage(self, t):
        # Return voltage
        return self._v(t).numpy()

    def forward(self, t, y):
        a, r = torch.unbind(y[0])

        try:
            v = self._v(t).to(device)
        except ValueError:
            v = torch.tensor([-80]).to(device)

        k1 = self.p1 * torch.exp(self.p2 * v)
        k2 = self.p3 * torch.exp(-self.p4 * v)
        k3 = self.p5 * torch.exp(self.p6 * v)
        k4 = self.p7 * torch.exp(-self.p8 * v)

        dadt = k1 * (self.unity - a) - k2 * a
        drdt = -k3 * r + k4 * (self.unity - r)

        return torch.stack([dadt[0], drdt[0]])


class ODEFunc1_6(nn.Module):

    def __init__(self):
        super(ODEFunc1_6, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 1),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        self.vrange = torch.tensor([100.]).to(device)
        self.netscale = torch.tensor([1000.]).to(device)

        # https://github.com/CardiacModelling/FourWaysOfFitting/blob/master/method-3/cell-5-fit-3-run-001.txt
        self.p5 = 8.73243709432939552e-02
        self.p6 = 7.33380025549188515e-03
        self.p7 = 6.16551007196145754e-03
        self.p8 = 3.15741310933875322e-02

        self.unity = torch.tensor([1]).to(device)

    def set_fixed_form_voltage_protocol(self, t, v):
        # Regular time point voltage protocol time series
        self._t_regular = t
        self._v_regular = v
        self.__v = interp1d(t, v)

    def _v(self, t):
        #return torch.from_numpy(np.interp([t.cpu().detach().numpy()], self._t_regular,
        #                                  self._v_regular))
        return torch.from_numpy(self.__v([t.cpu().detach().numpy()]))

    def voltage(self, t):
        # Return voltage
        return self._v(t).numpy()

    def forward(self, t, y):
        a, r = torch.unbind(y, dim=1)

        try:
            v = self._v(t).to(device)
        except ValueError:
            v = torch.tensor([-80]).to(device)
        nv = v / self.vrange

        k3 = self.p5 * torch.exp(self.p6 * v)
        k4 = self.p7 * torch.exp(-self.p8 * v)

        drdt = -k3 * r + k4 * (self.unity - r)

        dadt = self.net(torch.stack([nv[0], a[0]]).float()) / self.netscale

        return torch.stack([dadt[0], drdt[0]]).reshape(1, -1)


class ODEFunc1_6_2(nn.Module):

    def __init__(self):
        super(ODEFunc1_6_2, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 1),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-3)
                nn.init.constant_(m.bias, val=0)

        self.vrange = torch.tensor([100.]).to(device)
        self.netscale = torch.tensor([1000.]).to(device)

        # https://github.com/CardiacModelling/FourWaysOfFitting/blob/master/method-3/cell-5-fit-3-run-001.txt
        self.p1 = 2.10551451120238317e-04
        self.p2 = 6.57994674459572992e-02
        self.p3 = 3.31717454417642909e-06
        self.p4 = 7.43102564328181336e-02
        self.p5 = 8.73243709432939552e-02
        self.p6 = 7.33380025549188515e-03
        self.p7 = 6.16551007196145754e-03
        self.p8 = 3.15741310933875322e-02

        self.unity = torch.tensor([1]).to(device)

    def set_fixed_form_voltage_protocol(self, t, v):
        # Regular time point voltage protocol time series
        self._t_regular = t
        self._v_regular = v
        self.__v = interp1d(t, v)

    def _v(self, t):
        #return torch.from_numpy(np.interp([t.cpu().detach().numpy()], self._t_regular,
        #                                  self._v_regular))
        return torch.from_numpy(self.__v([t.cpu().detach().numpy()]))

    def voltage(self, t):
        # Return voltage
        return self._v(t).numpy()

    def _dadt(self, a, v):
        k1 = self.p1 * torch.exp(self.p2 * v)
        k2 = self.p3 * torch.exp(-self.p4 * v)
        return k1 * (self.unity - a) - k2 * a

    def _drdt(self, r, v):
        k3 = self.p5 * torch.exp(self.p6 * v)
        k4 = self.p7 * torch.exp(-self.p8 * v)
        return -k3 * r + k4 * (self.unity - r)

    def forward(self, t, y):
        a, r = torch.unbind(y, dim=1)

        try:
            v = self._v(t).to(device)
        except ValueError:
            v = torch.tensor([-80]).to(device)
        nv = v / self.vrange

        drdt = self._drdt(r, v)

        dadt = self._dadt(a, v).reshape(-1)
        ddadt = self.net(torch.stack([nv[0], a[0]]).float()) / self.netscale
        dadt += ddadt.reshape(-1)

        return torch.stack([dadt[0], drdt[0]]).reshape(1, -1)
#
#
#


#
#
#
func_o = Lambda().to(device)
func_o.eval()

func_1 = ODEFunc1_6().to(device)
#func_1.load_state_dict(torch.load('r1/model-state-dict.pt'))
best_checkpoint = torch.load('r1/best-model-checkpoint-2.pt')
func_1.load_state_dict(best_checkpoint['state_dict'])
func_1.eval()

func_2 = ODEFunc1_6_2().to(device)
func_2.load_state_dict(torch.load('r2/model-state-dict-2.pt'))
func_2.eval()

# Load more prediction data
prediction2 = np.loadtxt('data/cell-5.csv', delimiter=',', skiprows=1)
timep2 = prediction2[:, 0]
timep2_torch = torch.from_numpy(prediction2[:, 0]).to(device)
currentp2 = prediction2[:, 1]
voltagep2 = prediction2[:, 2]

prediction3 = np.loadtxt('data/pr5-deactivation-cell-5.csv', delimiter=',', skiprows=1)
timep3 = prediction3[:, 0]
timep3_torch = torch.from_numpy(prediction3[:, 0]).to(device)
currentp3 = prediction3[:, 1]
voltagep3 = prediction3[:, 2]

true_y0 = true_y0s[1]  # (roughly holding at -80mV)

def predict(func, time, voltage, time_torch, data, gg, y0, e, name):
    func.set_fixed_form_voltage_protocol(time, voltage)
    with torch.no_grad():
        pred_y = odeint(func, y0, time_torch).to(device)
    pred_yo = gg * pred_y[:, 0, 0] * pred_y[:, 0, 1] * (func._v(time_torch).to(device) - e)
    loss = torch.mean(torch.abs(pred_yo - torch.from_numpy(data).to(device)))
    print('{:s} prediction | Total Loss {:.6f}'.format(name, loss.item()))
    return pred_yo


if args.cached:
    pred_y_o_pr3 = torch.load('table-2/yo-pr3.pt')
    pred_y_1_pr3 = torch.load('table-2/y1-pr3.pt')
    pred_y_2_pr3 = torch.load('table-2/y2-pr3.pt')
    pred_y_o_pr5 = torch.load('table-2/yo-pr5.pt')
    pred_y_1_pr5 = torch.load('table-2/y1-pr5.pt')
    pred_y_2_pr5 = torch.load('table-2/y2-pr5.pt')
    pred_y_o_aps = torch.load('table-2/yo-aps.pt')
    pred_y_1_aps = torch.load('table-2/y1-aps.pt')
    pred_y_2_aps = torch.load('table-2/y2-aps.pt')
    pred_y_o_sin = torch.load('table-2/yo-sinewave.pt')
    pred_y_1_sin = torch.load('table-2/y1-sinewave.pt')
    pred_y_2_sin = torch.load('table-2/y2-sinewave.pt')
    pred_y_o_pr4 = torch.load('table-2/yo-pr4.pt')
    pred_y_1_pr4 = torch.load('table-2/y1-pr4.pt')
    pred_y_2_pr4 = torch.load('table-2/y2-pr4.pt')
else:
    with torch.no_grad():
        ###
        ### Training protocols
        ###

        #
        # Pr4
        #
        # Trained Neural ODE
        pred_y_o = predict(func_o, time2, voltage2, time2_torch, current2, g, true_y0, e, 'Pr4 (Mo)')
        pred_y_1 = predict(func_1, time2, voltage2, time2_torch, current2, g_nn, true_y0, e_nnf, 'Pr4 (M1)')
        pred_y_2 = predict(func_2, time2, voltage2, time2_torch, current2, g_nn, true_y0, e, 'Pr4 (M2)')

        # Cache it
        torch.save(pred_y_o, 'table-2/yo-pr4.pt')
        torch.save(pred_y_1, 'table-2/y1-pr4.pt')
        torch.save(pred_y_2, 'table-2/y2-pr4.pt')
        pred_y_o_pr4 = pred_y_o
        pred_y_1_pr4 = pred_y_1
        pred_y_2_pr4 = pred_y_2
        del(pred_y_o, pred_y_1, pred_y_2)

        #
        # Pr3
        #
        # Trained Neural ODE
        pred_y_o = predict(func_o, time1, voltage1, time1_torch, current1, g, true_y0, e, 'Pr3 (Mo)')
        pred_y_1 = predict(func_1, time1, voltage1, time1_torch, current1, g_nn, true_y0, e_nnf, 'Pr3 (M1)')
        pred_y_2 = predict(func_2, time1, voltage1, time1_torch, current1, g_nn, true_y0, e, 'Pr3 (M2)')

        # Cache it
        torch.save(pred_y_o, 'table-2/yo-pr3.pt')
        torch.save(pred_y_1, 'table-2/y1-pr3.pt')
        torch.save(pred_y_2, 'table-2/y2-pr3.pt')
        pred_y_o_pr3 = pred_y_o
        pred_y_1_pr3 = pred_y_1
        pred_y_2_pr3 = pred_y_2
        del(pred_y_o, pred_y_1, pred_y_2)

        #
        # Pr5
        #
        # Trained Neural ODE
        pred_y_o = predict(func_o, timep3, voltagep3, timep3_torch, currentp3, g, true_y0, e, 'Pr5 (Mo)')
        pred_y_1 = predict(func_1, timep3, voltagep3, timep3_torch, currentp3, g_nn, true_y0, e_nnf, 'Pr5 (M1)')
        pred_y_2 = predict(func_2, timep3, voltagep3, timep3_torch, currentp3, g_nn, true_y0, e, 'Pr5 (M2)')

        # Cache it
        torch.save(pred_y_o, 'table-2/yo-pr5.pt')
        torch.save(pred_y_1, 'table-2/y1-pr5.pt')
        torch.save(pred_y_2, 'table-2/y2-pr5.pt')
        pred_y_o_pr5 = pred_y_o
        pred_y_1_pr5 = pred_y_1
        pred_y_2_pr5 = pred_y_2
        del(pred_y_o, pred_y_1, pred_y_2)

        #
        # Sinewave
        #
        # Trained Neural ODE
        pred_y_o = predict(func_o, timep2, voltagep2, timep2_torch, currentp2, g, true_y0, e, 'Sinewave (Mo)')
        pred_y_1 = predict(func_1, timep2, voltagep2, timep2_torch, currentp2, g_nn, true_y0, e_nnf, 'Sinewave (M1)')
        pred_y_2 = predict(func_2, timep2, voltagep2, timep2_torch, currentp2, g_nn, true_y0, e, 'Sinewave (M2)')

        # Cache it
        torch.save(pred_y_o, 'table-2/yo-sinewave.pt')
        torch.save(pred_y_1, 'table-2/y1-sinewave.pt')
        torch.save(pred_y_2, 'table-2/y2-sinewave.pt')
        pred_y_o_sin = pred_y_o
        pred_y_1_sin = pred_y_1
        pred_y_2_sin = pred_y_2
        del(pred_y_o, pred_y_1, pred_y_2)

        #
        # APs
        #
        # Trained Neural ODE
        pred_y_o = predict(func_o, timep1, voltagep1, timep1_torch, currentp1, g, true_y0, e, 'APs (Mo)')
        pred_y_1 = predict(func_1, timep1, voltagep1, timep1_torch, currentp1, g_nn, true_y0, e_nnf, 'APs (M1)')
        pred_y_2 = predict(func_2, timep1, voltagep1, timep1_torch, currentp1, g_nn, true_y0, e, 'APs (M2)')

        # Cache it
        torch.save(pred_y_o, 'table-2/yo-aps.pt')
        torch.save(pred_y_1, 'table-2/y1-aps.pt')
        torch.save(pred_y_2, 'table-2/y2-aps.pt')
        pred_y_o_aps = pred_y_o
        pred_y_1_aps = pred_y_1
        pred_y_2_aps = pred_y_2
        del(pred_y_o, pred_y_1, pred_y_2)


def loss(x, y):
    # return torch.sqrt(torch.mean(torch.square(x - y))).item()
    return torch.mean(torch.abs(x - y)).item()

training_pr3_y_o = loss(pred_y_o_pr3, current1)
training_pr3_y_1 = loss(pred_y_1_pr3, current1)
training_pr3_y_2 = loss(pred_y_2_pr3, current1)
training_pr5_y_o = loss(pred_y_o_pr5, currentp3)
training_pr5_y_1 = loss(pred_y_1_pr5, currentp3)
training_pr5_y_2 = loss(pred_y_2_pr5, currentp3)
l = int(len(time2) / 16)  # 16 steps
pred_pr4_y_o = loss(pred_y_o_pr4.reshape(-1)[l*1:l*(3+1)], current2.reshape(-1)[l*1:l*(3+1)])
pred_pr4_y_1 = loss(pred_y_1_pr4.reshape(-1)[l*1:l*(3+1)], current2.reshape(-1)[l*1:l*(3+1)])
pred_pr4_y_2 = loss(pred_y_2_pr4.reshape(-1)[l*1:l*(3+1)], current2.reshape(-1)[l*1:l*(3+1)])
pred_sin_y_o = loss(pred_y_o_sin, currentp2)
pred_sin_y_1 = loss(pred_y_1_sin, currentp2)
pred_sin_y_2 = loss(pred_y_2_sin, currentp2)
pred_aps_y_o = loss(pred_y_o_aps, currentp1)
pred_aps_y_1 = loss(pred_y_1_aps, currentp1)
pred_aps_y_2 = loss(pred_y_2_aps, currentp1)

output = "{@{}XXXcXXX@{}}\n"
output += "\\toprule\n"
output += "         & \\multicolumn{2}{c}{Training} & \phantom{a} & \\multicolumn{3}{c}{Prediction} \\\\\n"
output += "           \\cmidrule{2-3}                               \\cmidrule{5-7}\n"
output += "         & Pr3 & Pr5 & & Pr4 & Sinusoidal & APs \\\\\n"
output += "\\midrule\n"
output += "Original & %s & %s & & %s & %s & %s \\\\\n" % (
    round(training_pr3_y_o, 3),
    round(training_pr5_y_o, 3),
    round(pred_pr4_y_o, 3),
    round(pred_sin_y_o, 3),
    round(pred_aps_y_o, 3),
)
output += "NN-f     & %s & %s & & %s & %s & %s \\\\\n" % (
    round(training_pr3_y_1, 3),
    round(training_pr5_y_1, 3),
    round(pred_pr4_y_1, 3),
    round(pred_sin_y_1, 3),
    round(pred_aps_y_1, 3),
)
output += "NN-d     & %s & %s & & %s & %s & %s \\\\\n" % (
    round(training_pr3_y_2, 3),
    round(training_pr5_y_2, 3),
    round(pred_pr4_y_2, 3),
    round(pred_sin_y_2, 3),
    round(pred_aps_y_2, 3),
)
output += "\\bottomrule"

with open('table-2/table-2.txt', 'w') as f:
    f.write(output)
