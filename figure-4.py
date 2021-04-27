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
sns.set(rc={'axes.facecolor':'#E4EDE4'})
from matplotlib.path import Path
from matplotlib.patches import PathPatch

import torch
import torch.nn as nn

parser = argparse.ArgumentParser('IKr NN ODE real data plot 1')
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

makedirs('figure-4')


#
# Load data
#
raw_data1 = np.loadtxt('data/pr3-steady-activation-cell-5.csv', delimiter=',', skiprows=1)
time1 = raw_data1[:, 0]
time1_torch = torch.from_numpy(raw_data1[:, 0]).to(device)
current1 = raw_data1[:, 1]
voltage1 = raw_data1[:, 2]

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
    pred_y_o_pr3 = torch.load('figure-4/yo-pr3.pt')
    pred_y_1_pr3 = torch.load('figure-4/y1-pr3.pt')
    pred_y_2_pr3 = torch.load('figure-4/y2-pr3.pt')
    pred_y_o_pr5 = torch.load('figure-4/yo-pr5.pt')
    pred_y_1_pr5 = torch.load('figure-4/y1-pr5.pt')
    pred_y_2_pr5 = torch.load('figure-4/y2-pr5.pt')
else:
    with torch.no_grad():
        ###
        ### Training protocols
        ###

        #
        # Pr3
        #
        # Trained Neural ODE
        makedirs('figure-4/pr3')
        pred_y_o = predict(func_o, time1, voltage1, time1_torch, current1, g, true_y0, e, 'Pr3 (Mo)')
        pred_y_1 = predict(func_1, time1, voltage1, time1_torch, current1, g_nn, true_y0, e_nnf, 'Pr3 (M1)')
        pred_y_2 = predict(func_2, time1, voltage1, time1_torch, current1, g_nn, true_y0, e, 'Pr3 (M2)')

        l = int(len(time1) / 7)  # 7 steps

        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current (nA)')
        for i in range(7):
            ax1.plot(time1[:l], current1[l*i:l*(i+1)], c='#7f7f7f', label='__nolegend__' if i else 'Data')
            ax1.plot(time1[:l], pred_y_o.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C0', label='__nolegend__' if i else 'Original')
            ax1.plot(time1[:l], pred_y_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C1', label='__nolegend__' if i else 'Full NN')
            ax1.plot(time1[:l], pred_y_2.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '-.', c='C2', label='__nolegend__' if i else 'NN discrepancy')

            fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Current (nA)')
            ax2.plot(time1[:l], current1[l*i:l*(i+1)], c='#7f7f7f', label='__nolegend__' if i else 'Data')
            ax2.plot(time1[:l], pred_y_o.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C0', label='Original')
            ax2.plot(time1[:l], pred_y_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C1', label='Full NN')
            ax2.plot(time1[:l], pred_y_2.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '-.', c='C2', label='NN discrepancy')
            ax2.set_xlim(time1[:l].min(), time1[:l].max())
            #ax2.set_ylim(-4, 1.9)
            ax2.legend()
            fig2.tight_layout()
            fig2.savefig('figure-4/pr3/s%s' % i, dpi=200)
            plt.close(fig2)
        ax1.set_xlim(time1[:l].min(), time1[:l].max())
        #ax1.set_ylim(-4, 1.9)
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig('figure-4/pr3', dpi=200)
        # do another one with zooms
        ax1.set_xlim(5000, 7000)
        #ax1.set_ylim(-2, 1.7)
        fig1.tight_layout()
        fig1.savefig('figure-4/pr3-z', dpi=200)
        #plt.show()
        plt.close(fig1)

        # Cache it
        torch.save(pred_y_o, 'figure-4/yo-pr3.pt')
        torch.save(pred_y_1, 'figure-4/y1-pr3.pt')
        torch.save(pred_y_2, 'figure-4/y2-pr3.pt')
        pred_y_o_pr3 = pred_y_o
        pred_y_1_pr3 = pred_y_1
        pred_y_2_pr3 = pred_y_2
        del(pred_y_o, pred_y_1, pred_y_2)

        #
        # Pr5
        #
        makedirs('figure-4/pr5')
        # Trained Neural ODE
        pred_y_o = predict(func_o, timep3, voltagep3, timep3_torch, currentp3, g, true_y0, e, 'Pr5 (Mo)')
        pred_y_1 = predict(func_1, timep3, voltagep3, timep3_torch, currentp3, g_nn, true_y0, e_nnf, 'Pr5 (M1)')
        pred_y_2 = predict(func_2, timep3, voltagep3, timep3_torch, currentp3, g_nn, true_y0, e, 'Pr5 (M2)')

        l = int(len(timep3) / 9)  # 9 steps

        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current (nA)')
        for i in range(9):
            ax1.plot(timep3[:l], currentp3[l*i:l*(i+1)], c='#7f7f7f', label='__nolegend__' if i else 'Data')
            ax1.plot(timep3[:l], pred_y_o.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C0', label='__nolegend__' if i else 'Original')
            ax1.plot(timep3[:l], pred_y_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C1', label='__nolegend__' if i else 'Full NN')
            ax1.plot(timep3[:l], pred_y_2.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '-.', c='C2', label='__nolegend__' if i else 'NN discrepancy')

            fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Current (nA)')
            ax2.plot(timep3[:l], currentp3[l*i:l*(i+1)], c='#7f7f7f', label='__nolegend__' if i else 'Data')
            ax2.plot(timep3[:l], pred_y_o.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C0', label='Original')
            ax2.plot(timep3[:l], pred_y_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C1', label='Full NN')
            ax2.plot(timep3[:l], pred_y_2.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '-.', c='C2', label='NN discrepancy')
            ax2.set_xlim(timep3[:l].min(), timep3[:l].max())
            #ax2.set_ylim(-3, 7.5)
            ax2.legend()
            fig2.tight_layout()
            fig2.savefig('figure-4/pr5/s%s' % i, dpi=200)
            plt.close(fig2)
        ax1.set_xlim(timep3[:l].min(), timep3[:l].max())
        #ax1.set_ylim(-3, 7.5)
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig('figure-4/pr5', dpi=300)
        # do another one with zooms
        #ax1.set_xlim(1175, 1475)
        #ax1.set_ylim(-2.5, 7)
        #fig1.tight_layout()
        #fig1.savefig('figure-4/pr5-z', dpi=300)
        #plt.show()
        plt.close(fig1)

        # Cache it
        torch.save(pred_y_o, 'figure-4/yo-pr5.pt')
        torch.save(pred_y_1, 'figure-4/y1-pr5.pt')
        torch.save(pred_y_2, 'figure-4/y2-pr5.pt')
        pred_y_o_pr5 = pred_y_o
        pred_y_1_pr5 = pred_y_1
        pred_y_2_pr5 = pred_y_2
        del(pred_y_o, pred_y_1, pred_y_2)


#
# Settings
#
zoom_in_win = {
    0: [(1000, 5000), (6600, 7100)],  # pr3
    1: [(2600, 3000), (8650, 9100)],  # pr5
}
zoom_in_y = {
    0: [(-0.1, 0.7), (-4., 0.5)],  # pr3
    1: [(-4., 2.), (-3., 0.5)],  # pr5
}
facecolors = [
    [sns.color_palette("Set2")[0], sns.color_palette("Set2")[1]],  # pr3
    [sns.color_palette("Set2")[2], sns.color_palette("Set2")[3]],  # pr5
]


#
# Plot
#
ds = 20
fig = plt.figure(figsize=(11, 5))
n_maxzoom = 2
grid = plt.GridSpec(4 + 1 + 12 + 5 + 9, 4,
                    hspace=0.0, wspace=0.0)
axes = np.empty([3, 2], dtype=object)
for i in range(2):
    i_grid = i * (2 + 0)
    f_grid = (i + 1) * 2 + i * 0

    axes[0, i] = fig.add_subplot(grid[:4, i_grid:f_grid])
    axes[0, i].set_xticklabels([])
    axes[1, i] = fig.add_subplot(grid[5:17, i_grid:f_grid])
    axes[2, i] = np.empty(n_maxzoom, dtype=object)

    n_zoom = len(zoom_in_win[i])
    for ii in range(n_zoom):
        axes[2, i][ii] = fig.add_subplot(
                grid[-9:, i_grid+ii*1:i_grid+(ii+1)*1])
        axes[2, i][ii].set_xticklabels([])
        axes[2, i][ii].set_xticks([])
        axes[2, i][ii].set_yticklabels([])
        axes[2, i][ii].set_yticks([])

# Set labels
axes[0, 0].set_ylabel('Voltage\n(mV)', fontsize=12)
axes[1, 0].set_ylabel('Current\n(nA)', fontsize=12)
axes[2, 0][0].set_ylabel('Zoom in', fontsize=12)
axes[1, 0].set_xlabel('Time (ms)', fontsize=12)
axes[1, 1].set_xlabel('Time (ms)', fontsize=12)

# Plot!
l = int(len(time1) / 7)  # 7 steps
y_mins = [np.inf] * len(zoom_in_win[0])
y_maxs = [-np.inf] * len(zoom_in_win[0])
for i in range(7):
    axes[0, 0].plot(time1[:l], voltage1[l*i:l*(i+1)], c='#7f7f7f', ds='steps')

    axes[1, 0].plot(time1[:l:ds], current1.reshape(-1)[l*i:l*(i+1):ds], c='#7f7f7f', label='__nolegend__' if i else 'Data')
    axes[1, 0].plot(time1[:l:ds], pred_y_o_pr3.reshape(-1).cpu().numpy()[l*i:l*(i+1):ds], '--', c='C0', label='__nolegend__' if i else 'Original')
    axes[1, 0].plot(time1[:l:ds], pred_y_1_pr3.reshape(-1).cpu().numpy()[l*i:l*(i+1):ds], '--', c='C1', label='__nolegend__' if i else r'$a$-gate as NN (NN-f)')
    axes[1, 0].plot(time1[:l:ds], pred_y_2_pr3.reshape(-1).cpu().numpy()[l*i:l*(i+1):ds], '-.', c='C2', label='__nolegend__' if i else r'NN as discrepancy term (NN-d)')

    axes[0, 0].set_xlim([time1[:l:ds][0], time1[:l:ds][-1]])
    axes[1, 0].set_xlim([time1[:l:ds][0], time1[:l:ds][-1]])

    # Zooms
    for i_z, (t_i, t_f) in enumerate(zoom_in_win[0]):
        # Find closest time
        idx_i = np.argmin(np.abs(time1[:l:ds] - t_i))
        idx_f = np.argmin(np.abs(time1[:l:ds] - t_f))
        # Data
        t = time1[:l:ds][idx_i:idx_f]
        c = current1.reshape(-1)[l*i:l*(i+1):ds][idx_i:idx_f]
        y0 = pred_y_o_pr3.reshape(-1).cpu().numpy()[l*i:l*(i+1):ds][idx_i:idx_f]
        y1 = pred_y_1_pr3.reshape(-1).cpu().numpy()[l*i:l*(i+1):ds][idx_i:idx_f]
        y2 = pred_y_2_pr3.reshape(-1).cpu().numpy()[l*i:l*(i+1):ds][idx_i:idx_f]
        # Work out the max and min
        y_mins[i_z] = min(np.min(c), y_mins[i_z])
        y_maxs[i_z] = max(np.max(c), y_maxs[i_z])
        # Work out third panel plot
        axes[2, 0][i_z].plot(t, c, c='#7f7f7f')
        axes[2, 0][i_z].plot(t, y0, '--', c='C0')
        axes[2, 0][i_z].plot(t, y1, '--', c='C1')
        axes[2, 0][i_z].plot(t, y2, '-.', c='C2')

        if i == 6:
            axes[2, 0][i_z].set_xlim([t[0], t[-1]])
            # Re-adjust the max and min
            if False:
                y_min, y_max = y_mins[i_z], y_maxs[i_z]
                y_amp = y_max - y_min
                y_min -=  0.2 * y_amp
                y_max +=  0.2 * y_amp
                y_amp = y_max - y_min
            else:
                y_min, y_max = zoom_in_y[0][i_z]
            axes[2, 0][i_z].set_ylim([y_min, y_max])
            # And plot shading over second panels
            codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
            vertices = np.array([(t[0], y_min),
                                 (t[0], y_max),
                                 (t[-1], y_max),
                                 (t[-1], y_min),
                                 (0, 0)], float)
            pathpatch = PathPatch(Path(vertices, codes),
                    facecolor=facecolors[0][i_z],
                    edgecolor=facecolors[0][i_z],
                    #edgecolor=None,
                    alpha=0.25)
            plt.sca(axes[1, 0])
            pyplot_axes = plt.gca()
            pyplot_axes.add_patch(pathpatch)
            # Set background color to match shading color
            vertices = np.array([(t[0], y_min),
                                 (t[0], y_max),
                                 (t[-1], y_max),
                                 (t[-1], y_min),
                                 (t[0], y_min)], float)
            pathpatch = PathPatch(Path(vertices, codes),
                    facecolor=facecolors[0][i_z],
                    #edgecolor=facecolors[0][i_z],
                    edgecolor=None,
                    alpha=0.25)
            plt.sca(axes[2, 0][i_z])
            pyplot_axes = plt.gca()
            pyplot_axes.add_patch(pathpatch)
            # Set arrow and time duration
            axes[2, 0][i_z].arrow(1, -0.05, -1, 0,
                                  length_includes_head=True,
                                  head_width=0.03, head_length=0.05,
                                  clip_on=False, fc='k', ec='k',
                                  transform=axes[2, 0][i_z].transAxes)
            axes[2, 0][i_z].arrow(0, -0.05, 1, 0,
                                  length_includes_head=True,
                                  head_width=0.03, head_length=0.05,
                                  clip_on=False, fc='k', ec='k',
                                  transform=axes[2, 0][i_z].transAxes)
            axes[2, 0][i_z].text(0.5, -0.15,
                    '%s ms' % np.around(t_f - t_i, decimals=0),
                    transform=axes[2, 0][i_z].transAxes,
                    horizontalalignment='center',
                    verticalalignment='center')
            # Set arrow and current range
            axes[2, 0][i_z].arrow(-0.05, 1, 0, -1,
                                  length_includes_head=True,
                                  head_width=0.03, head_length=0.05,
                                  clip_on=False, fc='k', ec='k',
                                  transform=axes[2, 0][i_z].transAxes)
            axes[2, 0][i_z].arrow(-0.05, 0, 0, 1,
                                  length_includes_head=True,
                                  head_width=0.03, head_length=0.05,
                                  clip_on=False, fc='k', ec='k',
                                  transform=axes[2, 0][i_z].transAxes)
            axes[2, 0][i_z].text(-0.15, 0.5,
                    '%s nA' % np.around(y_max - y_min, decimals=1),
                    rotation=90,
                    transform=axes[2, 0][i_z].transAxes,
                    horizontalalignment='center',
                    verticalalignment='center')

l = int(len(timep3) / 9)  # 9 steps
y_mins = [np.inf] * len(zoom_in_win[1])
y_maxs = [-np.inf] * len(zoom_in_win[1])
for i in range(9):
    axes[0, 1].plot(timep3[:l], voltagep3[l*i:l*(i+1)], c='#7f7f7f', ds='steps')

    axes[1, 1].plot(timep3[:l:ds], currentp3.reshape(-1)[l*i:l*(i+1):ds], c='#7f7f7f', label='__nolegend__' if i else 'Data')
    axes[1, 1].plot(timep3[:l:ds], pred_y_o_pr5.reshape(-1).cpu().numpy()[l*i:l*(i+1):ds], '--', c='C0', label='__nolegend__' if i else 'Original')
    axes[1, 1].plot(timep3[:l:ds], pred_y_1_pr5.reshape(-1).cpu().numpy()[l*i:l*(i+1):ds], '--', c='C1', label='__nolegend__' if i else r'$a$-gate as NN (NN-f)')
    axes[1, 1].plot(timep3[:l:ds], pred_y_2_pr5.reshape(-1).cpu().numpy()[l*i:l*(i+1):ds], '-.', c='C2', label='__nolegend__' if i else 'NN as discrepancy term (NN-d)')

    axes[0, 1].set_xlim([timep3[:l:ds][0], timep3[:l:ds][-1]])
    axes[1, 1].set_xlim([timep3[:l:ds][0], timep3[:l:ds][-1]])

    # Zooms
    for i_z, (t_i, t_f) in enumerate(zoom_in_win[1]):
        # Find closest time
        idx_i = np.argmin(np.abs(timep3[:l:ds] - t_i))
        idx_f = np.argmin(np.abs(timep3[:l:ds] - t_f))
        # Data
        t = timep3[:l:ds][idx_i:idx_f]
        c = currentp3.reshape(-1)[l*i:l*(i+1):ds][idx_i:idx_f]
        y0 = pred_y_o_pr5.reshape(-1).cpu().numpy()[l*i:l*(i+1):ds][idx_i:idx_f]
        y1 = pred_y_1_pr5.reshape(-1).cpu().numpy()[l*i:l*(i+1):ds][idx_i:idx_f]
        y2 = pred_y_2_pr5.reshape(-1).cpu().numpy()[l*i:l*(i+1):ds][idx_i:idx_f]
        # Work out the max and min
        y_mins[i_z] = min(np.min(c), y_mins[i_z])
        y_maxs[i_z] = max(np.max(c), y_maxs[i_z])
        # Work out third panel plot
        axes[2, 1][i_z].plot(t, c, c='#7f7f7f')
        axes[2, 1][i_z].plot(t, y0, '--', c='C0')
        axes[2, 1][i_z].plot(t, y1, '--', c='C1')
        axes[2, 1][i_z].plot(t, y2, '-.', c='C2')

        if i == 6:
            axes[2, 1][i_z].set_xlim([t[0], t[-1]])
            # Re-adjust the max and min
            if False:
                y_min, y_max = y_mins[i_z], y_maxs[i_z]
                y_amp = y_max - y_min
                y_min -=  0.2 * y_amp
                y_max +=  0.2 * y_amp
                y_amp = y_max - y_min
            else:
                y_min, y_max = zoom_in_y[1][i_z]
            axes[2, 1][i_z].set_ylim([y_min, y_max])
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
            plt.sca(axes[1, 1])
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
            plt.sca(axes[2, 1][i_z])
            pyplot_axes = plt.gca()
            pyplot_axes.add_patch(pathpatch)
            # Set arrow and time duration
            axes[2, 1][i_z].arrow(1, -0.05, -1, 0,
                                  length_includes_head=True,
                                  head_width=0.03, head_length=0.05,
                                  clip_on=False, fc='k', ec='k',
                                  transform=axes[2, 1][i_z].transAxes)
            axes[2, 1][i_z].arrow(0, -0.05, 1, 0,
                                  length_includes_head=True,
                                  head_width=0.03, head_length=0.05,
                                  clip_on=False, fc='k', ec='k',
                                  transform=axes[2, 1][i_z].transAxes)
            axes[2, 1][i_z].text(0.5, -0.15,
                    '%s ms' % np.around(t_f - t_i, decimals=1),
                    transform=axes[2, 1][i_z].transAxes,
                    horizontalalignment='center',
                    verticalalignment='center')
            # Set arrow and current range
            axes[2, 1][i_z].arrow(-0.05, 1, 0, -1,
                                  length_includes_head=True,
                                  head_width=0.03, head_length=0.05,
                                  clip_on=False, fc='k', ec='k',
                                  transform=axes[2, 1][i_z].transAxes)
            axes[2, 1][i_z].arrow(-0.05, 0, 0, 1,
                                  length_includes_head=True,
                                  head_width=0.03, head_length=0.05,
                                  clip_on=False, fc='k', ec='k',
                                  transform=axes[2, 1][i_z].transAxes)
            axes[2, 1][i_z].text(-0.15, 0.5,
                    '%s nA' % np.around(y_max - y_min, decimals=0),
                    rotation=90,
                    transform=axes[2, 1][i_z].transAxes,
                    horizontalalignment='center',
                    verticalalignment='center')

axes[1, 0].set_ylim([-4, 2])
axes[1, 1].set_ylim([-4, 2])
axes[1, 0].legend(loc='lower left', bbox_to_anchor=(-0.02, 1.55), ncol=4,
                  columnspacing=4, #handletextpad=1,
                  bbox_transform=axes[1, 0].transAxes)

axes[0, 0].text(-0.05, 1.05, '(A)', size=12, weight='bold',
                va='bottom', ha='right', transform=axes[0, 0].transAxes)
axes[0, 1].text(-0.05, 1.05, '(B)', size=12, weight='bold',
                va='bottom', ha='right', transform=axes[0, 1].transAxes)

fig.align_ylabels([axes[0, 0], axes[1, 0], axes[2, 0][0]])
#grid.tight_layout(fig, pad=0.1, rect=(0, 0, -0.8, 0.95))
grid.update(wspace=0.4, hspace=0)

plt.savefig('figure-4/fig4.pdf', format='pdf', pad_inches=0.02,
            bbox_inches='tight')
fig.canvas.start_event_loop(sys.float_info.min)  # Silence Tkinter callback
plt.savefig('figure-4/fig4', pad_inches=0.02, dpi=300, bbox_inches='tight')
#plt.show()
plt.close()
