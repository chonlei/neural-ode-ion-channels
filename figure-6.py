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
sns.set(rc={'axes.facecolor':'#F2EAEA'})
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from smoothing import smooth

import torch
import torch.nn as nn

parser = argparse.ArgumentParser('IKr NN ODE real data plot 3.')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--cached', action='store_true')
args = parser.parse_args()

from torchdiffeq import odeint

device = 'cpu'

# Set random seed
np.random.seed(0)
torch.manual_seed(0)

noise_sigma = 0.1

true_y0s = [torch.tensor([[1., 0.]]).to(device),  # what you get after holding at +40mV
            torch.tensor([[0., 1.]]).to(device)]  # (roughly) what you get after holding at -80mV

# B1.2 in https://physoc.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1113%2FJP275733&file=tjp12905-sup-0001-textS1.pdf#page=4
e = e_nnf = torch.tensor([-88.4]).to(device)  # assume we know
# https://github.com/CardiacModelling/FourWaysOfFitting/blob/master/method-3/cell-5-fit-3-run-001.txt
g = torch.tensor([0.133898199260611944]).to(device)  # assume we know

g_nn = g * 1.2  # just because we see a-gate gets to ~1.2 at some point (in prt V=50), so can absorb that into the g.

#
#
#
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

makedirs('figure-6')


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

prediction3 = np.loadtxt('data/pr5-deactivation-cell-5.csv', delimiter=',', skiprows=1)
timep3 = prediction3[:, 0]
timep3_torch = torch.from_numpy(prediction3[:, 0]).to(device)
currentp3 = prediction3[:, 1]
voltagep3 = prediction3[:, 2]


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

#
#
#


#
#
#
func_1 = ODEFunc1_6().to(device)
func_1.load_state_dict(torch.load('r1-bad/model-state-dict.pt'))
func_1.eval()

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
    pred_y_1_pr3 = torch.load('figure-6/y1-pr3.pt')
    pred_y_1_pr4 = torch.load('figure-6/y1-pr4.pt')
    pred_y_1_pr5 = torch.load('figure-6/y1-pr5.pt')
else:
    with torch.no_grad():
        ###
        ### Training protocols
        ###

        #
        # Pr3
        #
        # Trained Neural ODE
        makedirs('figure-6/pr3')
        pred_y_1 = predict(func_1, time1, voltage1, time1_torch, current1, g_nn, true_y0, e_nnf, 'Pr3 (M1)')

        l = int(len(time1) / 7)  # 7 steps

        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current (nA)')
        for i in range(7):
            ax1.plot(time1[:l], current1[l*i:l*(i+1)], c='#7f7f7f', label='__nolegend__' if i else 'Data')
            ax1.plot(time1[:l], pred_y_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C1',
                     label='__nolegend__' if i else 'Full NN')

            fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Current (nA)')
            ax2.plot(time1[:l], current1[l*i:l*(i+1)], c='#7f7f7f', label='__nolegend__' if i else 'Data')
            ax2.plot(time1[:l], pred_y_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C1', label='Full NN')
            ax2.set_xlim(time1[:l].min(), time1[:l].max())
            #ax2.set_ylim(-4, 1.9)
            ax2.legend()
            fig2.tight_layout()
            fig2.savefig('figure-6/pr3/s%s' % i, dpi=200)
            plt.close(fig2)
        ax1.set_xlim(time1[:l].min(), time1[:l].max())
        #ax1.set_ylim(-4, 1.9)
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig('figure-6/pr3', dpi=200)
        # do another one with zooms
        ax1.set_xlim(5000, 7000)
        #ax1.set_ylim(-2, 1.7)
        fig1.tight_layout()
        fig1.savefig('figure-6/pr3-z', dpi=200)
        #plt.show()
        plt.close(fig1)

        # Cache it
        torch.save(pred_y_1, 'figure-6/y1-pr3.pt')
        pred_y_1_pr3 = pred_y_1
        del(pred_y_1)


        #
        # Pr4
        #
        # Trained Neural ODE
        makedirs('figure-6/pr4')
        pred_y_1 = predict(func_1, time2, voltage2, time2_torch, current2, g_nn, true_y0, e_nnf, 'Pr4 (M1)')

        l = int(len(time2) / 16)  # 16 steps

        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current (nA)')
        for i in range(16):
            ax1.plot(time2[:l], current2[l*i:l*(i+1)], c='#7f7f7f', label='__nolegend__' if i else 'Data')
            ax1.plot(time2[:l], pred_y_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C1',
                     label='__nolegend__' if i else 'Full NN')

            fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Current (nA)')
            ax2.plot(time2[:l], current2[l*i:l*(i+1)], c='#7f7f7f', label='__nolegend__' if i else 'Data')
            ax2.plot(time2[:l], pred_y_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C1', label='Full NN')
            ax2.set_xlim(time2[:l].min(), time2[:l].max())
            #ax2.set_ylim(-3, 7.5)
            ax2.legend()
            fig2.tight_layout()
            fig2.savefig('figure-6/pr4/s%s' % i, dpi=200)
            plt.close(fig2)
        ax1.set_xlim(time2[:l].min(), time2[:l].max())
        #ax1.set_ylim(-3, 7.5)
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig('figure-6/pr4', dpi=200)
        # do another one with zooms
        ax1.set_xlim(1175, 1475)
        #ax1.set_ylim(-2.5, 7)
        fig1.tight_layout()
        fig1.savefig('figure-6/pr4-z', dpi=200)
        #plt.show()
        plt.close(fig1)

        # Cache it
        torch.save(pred_y_1, 'figure-6/y1-pr4.pt')
        pred_y_1_pr4 = pred_y_1
        del(pred_y_1)


        #
        # Pr5 (prediction)
        #
        makedirs('figure-6/pr5')
        # Trained Neural ODE
        pred_y_1 = predict(func_1, timep3, voltagep3, timep3_torch, currentp3, g_nn, true_y0, e_nnf, 'Pr5 (M1)')

        l = int(len(timep3) / 9)  # 9 steps

        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current (nA)')
        for i in range(9):
            ax1.plot(timep3[:l], currentp3[l*i:l*(i+1)], c='#7f7f7f', label='__nolegend__' if i else 'Data')
            ax1.plot(timep3[:l], pred_y_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C1',
                     label='__nolegend__' if i else 'Full NN')

            fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Current (nA)')
            ax2.plot(timep3[:l], currentp3[l*i:l*(i+1)], c='#7f7f7f', label='__nolegend__' if i else 'Data')
            ax2.plot(timep3[:l], pred_y_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C1', label='Full NN')
            ax2.set_xlim(timep3[:l].min(), timep3[:l].max())
            #ax2.set_ylim(-3, 7.5)
            ax2.legend()
            fig2.tight_layout()
            fig2.savefig('figure-6/pr5/s%s' % i, dpi=200)
            plt.close(fig2)
        ax1.set_xlim(timep3[:l].min(), timep3[:l].max())
        #ax1.set_ylim(-3, 7.5)
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig('figure-6/pr5', dpi=300)
        plt.close(fig1)

        # Cache it
        torch.save(pred_y_1, 'figure-6/y1-pr5.pt')
        pred_y_1_pr5 = pred_y_1
        del(pred_y_1)
    #sys.exit()


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
fig = plt.figure(figsize=(11, 6.5))
n_maxzoom = 2
grid = plt.GridSpec((4 + 1 + 12 + 5) * 2, 4,
                    hspace=0.0, wspace=0.0)
axes = np.empty([4, 2], dtype=object)
for i in range(2):
    i_grid = i * (2 + 0)
    f_grid = (i + 1) * 2 + i * 0

    axes[0, i] = fig.add_subplot(grid[:4, i_grid:f_grid])
    axes[0, i].set_xticklabels([])
    axes[1, i] = fig.add_subplot(grid[5:17, i_grid:f_grid])

axes[2, 0] = fig.add_subplot(grid[22:22+4, :])
axes[2, 0].set_xticklabels([])
axes[3, 0] = fig.add_subplot(grid[22+5:22+17, :])

# Set labels
axes[0, 0].set_ylabel('Voltage\n(mV)', fontsize=12)
axes[1, 0].set_ylabel('Current\n(nA)', fontsize=12)
axes[1, 0].set_xlabel('Time (ms)', fontsize=12)
axes[1, 1].set_xlabel('Time (ms)', fontsize=12)
axes[2, 0].set_ylabel('Voltage\n(mV)', fontsize=12)
axes[3, 0].set_ylabel('Current\n(nA)', fontsize=12)
axes[3, 0].set_xlabel('Time (ms)', fontsize=12)

# Plot!
l = int(len(time1) / 7)  # 7 steps
for i in range(7)[::2]:
    axes[0, 0].plot(time1[:l], voltage1[l*i:l*(i+1)], c='#7f7f7f', ds='steps')

    axes[1, 0].plot(time1[:l:ds], current1.reshape(-1)[l*i:l*(i+1):ds], c='#7f7f7f', label='__nolegend__' if i else 'Data')
    axes[1, 0].plot(time1[:l:ds], pred_y_1_pr3.reshape(-1).cpu().numpy()[l*i:l*(i+1):ds], '--', c='C3',
                    label='__nolegend__' if i else r'Improperly trained $a$-gate as NN (NN-f)')

axes[0, 0].set_xlim([0, 7500])
axes[1, 0].set_xlim([0, 7500])


l = int(len(time2) / 16)  # 16 steps
for i in range(16)[::2]:
    axes[0, 1].plot(time2[:l], voltage2[l*i:l*(i+1)], c='#7f7f7f', ds='steps')

    axes[1, 1].plot(time2[:l:ds], current2.reshape(-1)[l*i:l*(i+1):ds], c='#7f7f7f')
    axes[1, 1].plot(time2[:l:ds], pred_y_1_pr4.reshape(-1).cpu().numpy()[l*i:l*(i+1):ds], '--', c='C3')

axes[0, 1].set_xlim([0, 2000])
axes[1, 1].set_xlim([0, 2000])


l = int(len(timep3) / 9)  # 9 steps
for i in range(9)[::2]:
    axes[2, 0].plot(timep3[:l], voltagep3[l*i:l*(i+1)], c='#7f7f7f', ds='steps')

    axes[3, 0].plot(timep3[:l:ds], currentp3.reshape(-1)[l*i:l*(i+1):ds], c='#7f7f7f')
    axes[3, 0].plot(timep3[:l:ds], pred_y_1_pr5.reshape(-1).cpu().numpy()[l*i:l*(i+1):ds], '--', c='C3')

axes[2, 0].set_xlim([timep3[:l:ds][0], timep3[:l:ds][-1]])
axes[3, 0].set_xlim([timep3[:l:ds][0], timep3[:l:ds][-1]])


axes[1, 0].set_ylim([-4, 2])
axes[1, 1].set_ylim([-2.75, 11])
axes[3, 0].set_ylim([-4.5, 5.5])

axes[1, 0].legend(loc='lower left', bbox_to_anchor=(-0.02, 1.55), ncol=4,
                  columnspacing=4, #handletextpad=1,
                  bbox_transform=axes[1, 0].transAxes)

axes[0, 0].text(-0.05, 1.05, '(A)', size=12, weight='bold',
                va='bottom', ha='right', transform=axes[0, 0].transAxes)
axes[0, 1].text(-0.05, 1.05, '(B)', size=12, weight='bold',
                va='bottom', ha='right', transform=axes[0, 1].transAxes)
axes[2, 0].text(-0.02, 1.05, '(C)', size=12, weight='bold',
                va='bottom', ha='right', transform=axes[2, 0].transAxes)

fig.align_ylabels([axes[0, 0], axes[1, 0], axes[2, 0], axes[3, 0]])
grid.update(wspace=0.4, hspace=0)

plt.savefig('figure-6/fig6.pdf', format='pdf', pad_inches=0.02,
            bbox_inches='tight')
fig.canvas.start_event_loop(sys.float_info.min)  # Silence Tkinter callback
plt.savefig('figure-6/fig6', pad_inches=0.02, dpi=300, bbox_inches='tight')
#plt.show()
plt.close()
