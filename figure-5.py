#!/usr/bin/env python3
import sys
import os
import argparse
import time
import numpy as np
from scipy.interpolate import interp1d

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
#print(sns.axes_style())
sns.set(rc={'axes.facecolor':'#E4EDE4'})
from matplotlib.path import Path
from matplotlib.patches import PathPatch

import torch
import torch.nn as nn

parser = argparse.ArgumentParser('IKr NN ODE real data plot 2')
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

makedirs('figure-5')


#
# Load data
#
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
    pred_y_o_aps = torch.load('figure-5/yo-aps.pt')
    pred_y_1_aps = torch.load('figure-5/y1-aps.pt')
    pred_y_2_aps = torch.load('figure-5/y2-aps.pt')
    pred_y_o_sin = torch.load('figure-5/yo-sinewave.pt')
    pred_y_1_sin = torch.load('figure-5/y1-sinewave.pt')
    pred_y_2_sin = torch.load('figure-5/y2-sinewave.pt')
    pred_y_o_pr4 = torch.load('figure-5/yo-pr4.pt')
    pred_y_1_pr4 = torch.load('figure-5/y1-pr4.pt')
    pred_y_2_pr4 = torch.load('figure-5/y2-pr4.pt')
else:
    with torch.no_grad():
        ###
        ### Prediction protocols
        ###

        #
        # Pr4
        #
        # Trained Neural ODE
        makedirs('figure-5/pr4')
        pred_y_o = predict(func_o, time2, voltage2, time2_torch, current2, g, true_y0, e, 'Pr4 (Mo)')
        pred_y_1 = predict(func_1, time2, voltage2, time2_torch, current2, g_nn, true_y0, e_nnf, 'Pr4 (M1)')
        pred_y_2 = predict(func_2, time2, voltage2, time2_torch, current2, g_nn, true_y0, e, 'Pr4 (M2)')

        l = int(len(time2) / 16)  # 16 steps

        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current (nA)')
        for i in range(16):
            ax1.plot(time2[:l], current2[l*i:l*(i+1)], c='#7f7f7f', label='__nolegend__' if i else 'Data')
            ax1.plot(time2[:l], pred_y_o.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C0', label='__nolegend__' if i else 'Original')
            ax1.plot(time2[:l], pred_y_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C1', label='__nolegend__' if i else 'Full NN')
            ax1.plot(time2[:l], pred_y_2.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '-.', c='C2', label='__nolegend__' if i else 'NN discrepancy')

            fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Current (nA)')
            ax2.plot(time2[:l], current2[l*i:l*(i+1)], c='#7f7f7f', label='__nolegend__' if i else 'Data')
            ax2.plot(time2[:l], pred_y_o.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C0', label='Original')
            ax2.plot(time2[:l], pred_y_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C1', label='Full NN')
            ax2.plot(time2[:l], pred_y_2.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '-.', c='C2', label='NN discrepancy')
            ax2.set_xlim(time2[:l].min(), time2[:l].max())
            #ax2.set_ylim(-3, 7.5)
            ax2.legend()
            fig2.tight_layout()
            fig2.savefig('figure-5/pr4/s%s' % i, dpi=200)
            plt.close(fig2)
        ax1.set_xlim(time2[:l].min(), time2[:l].max())
        #ax1.set_ylim(-3, 7.5)
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig('figure-5/pr4', dpi=200)
        # do another one with zooms
        ax1.set_xlim(1175, 1475)
        #ax1.set_ylim(-2.5, 7)
        fig1.tight_layout()
        fig1.savefig('figure-5/pr4-z', dpi=200)
        #plt.show()
        plt.close(fig1)

        # Cache it
        torch.save(pred_y_o, 'figure-5/yo-pr4.pt')
        torch.save(pred_y_1, 'figure-5/y1-pr4.pt')
        torch.save(pred_y_2, 'figure-5/y2-pr4.pt')
        pred_y_o_pr4 = pred_y_o
        pred_y_1_pr4 = pred_y_1
        pred_y_2_pr4 = pred_y_2
        del(pred_y_o, pred_y_1, pred_y_2)


        #
        # Sinewave
        #
        # Trained Neural ODE
        pred_y_o = predict(func_o, timep2, voltagep2, timep2_torch, currentp2, g, true_y0, e, 'Sinewave (Mo)')
        pred_y_1 = predict(func_1, timep2, voltagep2, timep2_torch, currentp2, g_nn, true_y0, e_nnf, 'Sinewave (M1)')
        pred_y_2 = predict(func_2, timep2, voltagep2, timep2_torch, currentp2, g_nn, true_y0, e, 'Sinewave (M2)')

        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current (nA)')
        ax1.plot(timep2, currentp2, c='#7f7f7f', label='Data')
        ax1.plot(timep2, pred_y_o.reshape(-1).cpu().numpy(), '--', label='Original')
        ax1.plot(timep2, pred_y_1.reshape(-1).cpu().numpy(), '--', label='Full NN')
        ax1.plot(timep2, pred_y_2.reshape(-1).cpu().numpy(), '-.', label='NN discrepancy')
        ax1.set_xlim(timep2.min(), timep2.max())
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig('figure-5/sinewave', dpi=200)
        #plt.show()
        plt.close(fig1)

        # Cache it
        torch.save(pred_y_o, 'figure-5/yo-sinewave.pt')
        torch.save(pred_y_1, 'figure-5/y1-sinewave.pt')
        torch.save(pred_y_2, 'figure-5/y2-sinewave.pt')
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

        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current (nA)')
        ax1.plot(timep1, currentp1, c='#7f7f7f', label='Data')
        ax1.plot(timep1, pred_y_o.reshape(-1).cpu().numpy(), '--', label='Original')
        ax1.plot(timep1, pred_y_1.reshape(-1).cpu().numpy(), '--', label='Full NN')
        ax1.plot(timep1, pred_y_2.reshape(-1).cpu().numpy(), '-.', label='NN discrepancy')
        ax1.set_xlim(timep1.min(), timep1.max())
        ax1.legend()
        fig1.tight_layout()

        fig1.savefig('figure-5/aps', dpi=200)
        plt.close(fig1)

        # Cache it
        torch.save(pred_y_o, 'figure-5/yo-aps.pt')
        torch.save(pred_y_1, 'figure-5/y1-aps.pt')
        torch.save(pred_y_2, 'figure-5/y2-aps.pt')
        pred_y_o_aps = pred_y_o
        pred_y_1_aps = pred_y_1
        pred_y_2_aps = pred_y_2
        del(pred_y_o, pred_y_1, pred_y_2)


#
# Settings
#
zoom_in_win = {
    0: [],  # pr4
    1: [(3700, 4500), (5500, 6250), (6400, 7000)],  # sinewave
    2: [(2400, 2800), (4300, 4900), (6100, 6500)],  # aps
}
zoom_in_y = {
    0: [],  # pr4
    1: [(-0., 1), (-0.6, 1.4), (-1.8, 0.2)],  # sinewave
    2: [(-0.2, 3.3), (0., 2.), (-0.1, 2.4)],  # aps
}
facecolors = [
    [],  # pr4
    [sns.color_palette("Set2")[0], sns.color_palette("Set2")[1], sns.color_palette("Set2")[2]],  # sinewave
    [sns.color_palette("Set2")[0], sns.color_palette("Set2")[1], sns.color_palette("Set2")[2]],  # aps
]


#
# Plot
#
fig = plt.figure(figsize=(11, 13.5))
n_maxzoom = 3
a1 = 8    # voltage
a11 = 1   # space
a2 = 19   # current
a22 = 13  # space with xticks (and xlabel)
aall = a1 + a11 + a2 + a22
b1 = 8    # voltage
b11 = 1   # space
b2 = 16   # current
b22 = 9   # space with xticks (and xlabel)
b3 = 18   # zoom
b33 = 10  # space
ball = b1 + b11 + b2 + b22 + b3 + b33
x1 = 5    # zoom
x11 = 1   # space
xall = x1 + x11
grid = plt.GridSpec(aall + (ball) * 2 - b33, (xall) * 3 - x11,
                    hspace=0.0, wspace=0.0)
axes = np.empty([3 * 3 - 1], dtype=object)
# pr4
axes[0] = np.empty(n_maxzoom, dtype=object)
axes[1] = np.empty(n_maxzoom, dtype=object)
for ii in range(n_maxzoom):
    i_grid = ii * xall
    f_grid = i_grid + x1
    axes[0][ii] = fig.add_subplot(grid[:a1, i_grid:f_grid])
    axes[0][ii].set_xticklabels([])
    axes[1][ii] = fig.add_subplot(grid[a1+a11:a1+a11+a2, i_grid:f_grid])
# sinewave
axes[2] = fig.add_subplot(grid[aall:aall+b1, :])
axes[2].set_xticklabels([])
axes[3] = fig.add_subplot(grid[aall+b1+b11:aall+b1+b11+b2, :])
axes[4] = np.empty(n_maxzoom, dtype=object)
n_zoom = len(zoom_in_win[1])
for ii in range(n_zoom):
    i_grid = ii * xall
    f_grid = i_grid + x1
    axes[4][ii] = fig.add_subplot(
            grid[aall+b1+b11+b2+b22:aall+b1+b11+b2+b22+b3, i_grid:f_grid])
    axes[4][ii].set_xticklabels([])
    axes[4][ii].set_xticks([])
    axes[4][ii].set_yticklabels([])
    axes[4][ii].set_yticks([])
# aps
axes[5] = fig.add_subplot(grid[aall+ball:aall+ball+b1, :])
axes[5].set_xticklabels([])
axes[6] = fig.add_subplot(grid[aall+ball+b1+b11:aall+ball+b1+b11+b2, :])
axes[7] = np.empty(n_maxzoom, dtype=object)
n_zoom = len(zoom_in_win[2])
for ii in range(n_zoom):
    i_grid = ii * xall
    f_grid = i_grid + x1
    axes[7][ii] = fig.add_subplot(
            grid[aall+ball+b1+b11+b2+b22:aall+ball+b1+b11+b2+b22+b3, i_grid:f_grid])
    axes[7][ii].set_xticklabels([])
    axes[7][ii].set_xticks([])
    axes[7][ii].set_yticklabels([])
    axes[7][ii].set_yticks([])

# Set labels
axes[0][0].set_ylabel('Voltage\n(mV)', fontsize=12)
axes[1][0].set_ylabel('Current\n(nA)', fontsize=12)
axes[2].set_ylabel('Voltage\n(mV)', fontsize=12)
axes[3].set_ylabel('Current\n(nA)', fontsize=12)
axes[4][0].set_ylabel('Zoom in', fontsize=12)
axes[5].set_ylabel('Voltage\n(mV)', fontsize=12)
axes[6].set_ylabel('Current\n(nA)', fontsize=12)
axes[7][0].set_ylabel('Zoom in', fontsize=12)
for ii in range(n_maxzoom):
    axes[1][ii].set_xlabel('Time (ms)', fontsize=12)
axes[3].set_xlabel('Time (ms)', fontsize=12)
axes[6].set_xlabel('Time (ms)', fontsize=12)


# Plot!

# pr4
l = int(len(time2) / 16)  # 16 steps
for ii in range(n_maxzoom):
    i = [1, 2, 3][ii]
    axes[0][ii].plot(time2[:l][:20000-4500], voltage2[l*i:l*(i+1)][4500:20000], c='#7f7f7f', ds='steps')

    axes[1][ii].plot(time2[:l][:20000-4500], current2.reshape(-1)[l*i:l*(i+1)][4500:20000], c='#7f7f7f', label='Data')
    axes[1][ii].plot(time2[:l][:20000-4500], pred_y_o_pr4.reshape(-1).cpu().numpy()[l*i:l*(i+1)][4500:20000], '--', c='C0', label='Original')
    axes[1][ii].plot(time2[:l][:20000-4500], pred_y_1_pr4.reshape(-1).cpu().numpy()[l*i:l*(i+1)][4500:20000], '--', c='C1', label=r'$a$-gate as NN (NN-f)')
    axes[1][ii].plot(time2[:l][:20000-4500], pred_y_2_pr4.reshape(-1).cpu().numpy()[l*i:l*(i+1)][4500:20000], '-.', c='C2', label=r'NN as discrepancy term (NN-d)')

    axes[0][ii].set_xlim(0, 1550)
    axes[1][ii].set_xlim(0, 1550)


# sinewave
axes[2].plot(timep2[:75000], voltagep2[:75000], c='#7f7f7f')

axes[3].plot(timep2[:75000], currentp2.reshape(-1)[:75000], c='#7f7f7f')
axes[3].plot(timep2[:75000], pred_y_o_sin.reshape(-1).cpu().numpy()[:75000], '--')
axes[3].plot(timep2[:75000], pred_y_1_sin.reshape(-1).cpu().numpy()[:75000], '--')
axes[3].plot(timep2[:75000], pred_y_2_sin.reshape(-1).cpu().numpy()[:75000], '-.')

axes[2].set_xlim(0, 7500)
axes[3].set_xlim(0, 7500)


# Zooms
for i_z, (t_i, t_f) in enumerate(zoom_in_win[1]):
    # Find closest time
    idx_i = np.argmin(np.abs(timep2 - t_i))
    idx_f = np.argmin(np.abs(timep2 - t_f))
    # Data
    t = timep2[idx_i:idx_f]
    c = currentp2.reshape(-1)[idx_i:idx_f]
    y0 = pred_y_o_sin.reshape(-1).cpu().numpy()[idx_i:idx_f]
    y1 = pred_y_1_sin.reshape(-1).cpu().numpy()[idx_i:idx_f]
    y2 = pred_y_2_sin.reshape(-1).cpu().numpy()[idx_i:idx_f]
    # Plot third panel
    axes[4][i_z].plot(t, c, c='#7f7f7f')
    axes[4][i_z].plot(t, y0, '--', c='C0')
    axes[4][i_z].plot(t, y1, '--', c='C1')
    axes[4][i_z].plot(t, y2, '-.', c='C2')

    axes[4][i_z].set_xlim([t[0], t[-1]])

    # Re-adjust the max and min
    y_min, y_max = zoom_in_y[1][i_z]
    axes[4][i_z].set_ylim([y_min, y_max])
    # And plot shading over second panels
    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    vertices = np.array([(t[0], y_min),
                         (t[0], y_max),
                         (t[-1], y_max),
                         (t[-1], y_min),
                         (0, 0)], float)
    pathpatch = PathPatch(Path(vertices, codes),
            facecolor=facecolors[1][i_z],
            edgecolor=facecolors[1][i_z],
            #edgecolor=None,
            alpha=0.25)
    plt.sca(axes[3])
    pyplot_axes = plt.gca()
    pyplot_axes.add_patch(pathpatch)
    # Set background color to match shading color
    vertices = np.array([(t[0], y_min),
                         (t[0], y_max),
                         (t[-1], y_max),
                         (t[-1], y_min),
                         (t[0], y_min)], float)
    pathpatch = PathPatch(Path(vertices, codes),
            facecolor=facecolors[1][i_z],
            #edgecolor=facecolors[1][i_z],
            edgecolor=None,
            alpha=0.25)
    plt.sca(axes[4][i_z])
    pyplot_axes = plt.gca()
    pyplot_axes.add_patch(pathpatch)
    # Set arrow and time duration
    axes[4][i_z].arrow(1, -0.05, -1, 0,
                       length_includes_head=True,
                       head_width=0.03, head_length=0.05,
                       clip_on=False, fc='k', ec='k',
                       transform=axes[4][i_z].transAxes)
    axes[4][i_z].arrow(0, -0.05, 1, 0,
                       length_includes_head=True,
                       head_width=0.03, head_length=0.05,
                       clip_on=False, fc='k', ec='k',
                       transform=axes[4][i_z].transAxes)
    axes[4][i_z].text(0.5, -0.2,
            '%s ms' % np.around(t_f - t_i, decimals=0),
            transform=axes[4][i_z].transAxes,
            horizontalalignment='center',
            verticalalignment='center')
    # Set arrow and current range
    axes[4][i_z].arrow(-0.05, 1, 0, -1,
                       length_includes_head=True,
                       head_width=0.03, head_length=0.05,
                       clip_on=False, fc='k', ec='k',
                       transform=axes[4][i_z].transAxes)
    axes[4][i_z].arrow(-0.05, 0, 0, 1,
                       length_includes_head=True,
                       head_width=0.03, head_length=0.05,
                       clip_on=False, fc='k', ec='k',
                       transform=axes[4][i_z].transAxes)
    axes[4][i_z].text(-0.1, 0.5,
            '%s nA' % np.around(y_max - y_min, decimals=1),
            rotation=90,
            transform=axes[4][i_z].transAxes,
            horizontalalignment='center',
            verticalalignment='center')


# aps
axes[5].plot(timep1[:84000], voltagep1[:84000], c='#7f7f7f')

axes[6].plot(timep1[:84000], currentp1.reshape(-1)[:84000], c='#7f7f7f')
axes[6].plot(timep1[:84000], pred_y_o_aps.reshape(-1).cpu().numpy()[:84000], '--')
axes[6].plot(timep1[:84000], pred_y_1_aps.reshape(-1).cpu().numpy()[:84000], '--')
axes[6].plot(timep1[:84000], pred_y_2_aps.reshape(-1).cpu().numpy()[:84000], '-.')

axes[5].set_xlim(0, 8400)
axes[6].set_xlim(0, 8400)


# Zooms
for i_z, (t_i, t_f) in enumerate(zoom_in_win[2]):
    # Find closest time
    idx_i = np.argmin(np.abs(timep1 - t_i))
    idx_f = np.argmin(np.abs(timep1 - t_f))
    # Data
    t = timep1[idx_i:idx_f]
    c = currentp1.reshape(-1)[idx_i:idx_f]
    y0 = pred_y_o_aps.reshape(-1).cpu().numpy()[idx_i:idx_f]
    y1 = pred_y_1_aps.reshape(-1).cpu().numpy()[idx_i:idx_f]
    y2 = pred_y_2_aps.reshape(-1).cpu().numpy()[idx_i:idx_f]
    # Plot third panel
    axes[7][i_z].plot(t, c, c='#7f7f7f')
    axes[7][i_z].plot(t, y0, '--', c='C0')
    axes[7][i_z].plot(t, y1, '--', c='C1')
    axes[7][i_z].plot(t, y2, '-.', c='C2')

    axes[7][i_z].set_xlim([t[0], t[-1]])

    # Re-adjust the max and min
    y_min, y_max = zoom_in_y[2][i_z]
    axes[7][i_z].set_ylim([y_min, y_max])
    # And plot shading over second panels
    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    vertices = np.array([(t[0], y_min),
                         (t[0], y_max),
                         (t[-1], y_max),
                         (t[-1], y_min),
                         (0, 0)], float)
    pathpatch = PathPatch(Path(vertices, codes),
            facecolor=facecolors[2][i_z],
            edgecolor=facecolors[2][i_z],
            #edgecolor=None,
            alpha=0.25)
    plt.sca(axes[6])
    pyplot_axes = plt.gca()
    pyplot_axes.add_patch(pathpatch)
    # Set background color to match shading color
    vertices = np.array([(t[0], y_min),
                         (t[0], y_max),
                         (t[-1], y_max),
                         (t[-1], y_min),
                         (t[0], y_min)], float)
    pathpatch = PathPatch(Path(vertices, codes),
            facecolor=facecolors[2][i_z],
            #edgecolor=facecolors[2][i_z],
            edgecolor=None,
            alpha=0.25)
    plt.sca(axes[7][i_z])
    pyplot_axes = plt.gca()
    pyplot_axes.add_patch(pathpatch)
    # Set arrow and time duration
    axes[7][i_z].arrow(1, -0.05, -1, 0,
                       length_includes_head=True,
                       head_width=0.03, head_length=0.05,
                       clip_on=False, fc='k', ec='k',
                       transform=axes[7][i_z].transAxes)
    axes[7][i_z].arrow(0, -0.05, 1, 0,
                       length_includes_head=True,
                       head_width=0.03, head_length=0.05,
                       clip_on=False, fc='k', ec='k',
                       transform=axes[7][i_z].transAxes)
    axes[7][i_z].text(0.5, -0.2,
            '%s ms' % np.around(t_f - t_i, decimals=0),
            transform=axes[7][i_z].transAxes,
            horizontalalignment='center',
            verticalalignment='center')
    # Set arrow and current range
    axes[7][i_z].arrow(-0.05, 1, 0, -1,
                       length_includes_head=True,
                       head_width=0.03, head_length=0.05,
                       clip_on=False, fc='k', ec='k',
                       transform=axes[7][i_z].transAxes)
    axes[7][i_z].arrow(-0.05, 0, 0, 1,
                       length_includes_head=True,
                       head_width=0.03, head_length=0.05,
                       clip_on=False, fc='k', ec='k',
                       transform=axes[7][i_z].transAxes)
    axes[7][i_z].text(-0.1, 0.5,
            '%s nA' % np.around(y_max - y_min, decimals=1),
            rotation=90,
            transform=axes[7][i_z].transAxes,
            horizontalalignment='center',
            verticalalignment='center')

axes[1][0].set_ylim(-1.8, 0.4)
axes[1][1].set_ylim(-2.1, 0.6)
axes[1][2].set_ylim(-2.4, 1.2)
axes[3].set_ylim(-3, 1.6)
axes[6].set_ylim(-2.2, 6.2)

axes[1][0].legend(loc='lower left', bbox_to_anchor=(-0.02, 1.65), ncol=4,
                  columnspacing=4, #handletextpad=1,
                  bbox_transform=axes[1][0].transAxes)

axes[0][0].text(-0.08, 1.05, '(A)', size=14, weight='bold',
                va='bottom', ha='right', transform=axes[0][0].transAxes)
axes[2].text(-0.025, 1.05, '(B)', size=14, weight='bold',
             va='bottom', ha='right', transform=axes[2].transAxes)
axes[5].text(-0.025, 1.05, '(C)', size=14, weight='bold',
             va='bottom', ha='right', transform=axes[5].transAxes)

fig.align_ylabels([axes[0][0], axes[1][0],
                   axes[2], axes[3], axes[4][0],
                   axes[5], axes[6], axes[7][0]])
#grid.tight_layout(fig, pad=0.1, rect=(0, 0, -0.8, 0.95))
grid.update(wspace=0.1, hspace=0)

plt.savefig('figure-5/fig5.pdf', format='pdf', pad_inches=0.02,
            bbox_inches='tight')
fig.canvas.start_event_loop(sys.float_info.min)  # Silence Tkinter callback
plt.savefig('figure-5/fig5', pad_inches=0.02, dpi=300, bbox_inches='tight')
#plt.show()
plt.close()
