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

parser = argparse.ArgumentParser('IKr NN ODE real data plot 4.')
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

g_nn = g #* 1.2  # just because we see a-gate gets to ~1.2 at some point (in prt V=50), so can absorb that into the g.

#
#
#
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

makedirs('figure-7')


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
'''  # NN-f wrong
func_1 = ODEFunc1_6().to(device)
func_1.load_state_dict(torch.load('r1-bad/model-state-dict.pt'))
func_1.eval()
'''  # Candidate model
func_1 = Lambda().to(device)
func_1.eval()
#'''

true_y0 = true_y0s[1]  # (roughly holding at -80mV)

def predict(func, time, voltage, time_torch, data, gg, y0, e, name):
    func.set_fixed_form_voltage_protocol(time, voltage)
    with torch.no_grad():
        pred_y = odeint(func, y0, time_torch).to(device)
    return pred_y[:, 0, 0], gg * pred_y[:, 0, 1] * (func._v(time_torch).to(device) - e)


if args.cached:
    pred_a_1_pr3 = torch.load('figure-7/a1-pr3.pt')
    pred_a_1_pr4 = torch.load('figure-7/a1-pr4.pt')
    pred_a_1_pr5 = torch.load('figure-7/a1-pr5.pt')
    pred_d_1_pr3 = torch.load('figure-7/d1-pr3.pt')
    pred_d_1_pr4 = torch.load('figure-7/d1-pr4.pt')
    pred_d_1_pr5 = torch.load('figure-7/d1-pr5.pt')
else:
    with torch.no_grad():
        ###
        ### Training protocols
        ###

        #
        # Pr3
        #
        # Trained Neural ODE
        makedirs('figure-7/pr3')
        pred_a_1, pred_d_1 = predict(func_1, time1, voltage1, time1_torch, current1, g_nn, true_y0, e_nnf, 'Pr3 (M1)')

        l = int(len(time1) / 7)  # 7 steps

        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current (nA)')
        for i in range(7):
            ax1.plot(time1[:l], current1[l*i:l*(i+1)] / pred_d_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], c='#7f7f7f', label='__nolegend__' if i else 'Data')
            ax1.plot(time1[:l], pred_a_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C1',
                     label='__nolegend__' if i else 'Full NN')

            fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Current (nA)')
            ax2.plot(time1[:l], current1[l*i:l*(i+1)] / pred_d_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], c='#7f7f7f', label='__nolegend__' if i else 'Data')
            ax2.plot(time1[:l], pred_a_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C1', label='Full NN')
            ax2.set_xlim(time1[:l].min(), time1[:l].max())
            #ax2.set_ylim(-4, 1.9)
            ax2.legend()
            fig2.tight_layout()
            fig2.savefig('figure-7/pr3/s%s' % i, dpi=200)
            plt.close(fig2)
        ax1.set_xlim(time1[:l].min(), time1[:l].max())
        #ax1.set_ylim(-4, 1.9)
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig('figure-7/pr3', dpi=200)
        # do another one with zooms
        ax1.set_xlim(5000, 7000)
        #ax1.set_ylim(-2, 1.7)
        fig1.tight_layout()
        fig1.savefig('figure-7/pr3-z', dpi=200)
        #plt.show()
        plt.close(fig1)

        # Cache it
        torch.save(pred_a_1, 'figure-7/a1-pr3.pt')
        torch.save(pred_d_1, 'figure-7/d1-pr3.pt')
        pred_a_1_pr3 = pred_a_1
        pred_d_1_pr3 = pred_d_1
        del(pred_a_1, pred_d_1)


        #
        # Pr4
        #
        # Trained Neural ODE
        makedirs('figure-7/pr4')
        pred_a_1, pred_d_1 = predict(func_1, time2, voltage2, time2_torch, current2, g_nn, true_y0, e_nnf, 'Pr4 (M1)')

        l = int(len(time2) / 16)  # 16 steps

        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current (nA)')
        for i in range(16):
            ax1.plot(time2[:l], current2[l*i:l*(i+1)] / pred_d_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], c='#7f7f7f', label='__nolegend__' if i else 'Data')
            ax1.plot(time2[:l], pred_a_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C1',
                     label='__nolegend__' if i else 'Full NN')

            fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Current (nA)')
            ax2.plot(time2[:l], current2[l*i:l*(i+1)] / pred_d_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], c='#7f7f7f', label='__nolegend__' if i else 'Data')
            ax2.plot(time2[:l], pred_a_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C1', label='Full NN')
            ax2.set_xlim(time2[:l].min(), time2[:l].max())
            #ax2.set_ylim(-3, 7.5)
            ax2.legend()
            fig2.tight_layout()
            fig2.savefig('figure-7/pr4/s%s' % i, dpi=200)
            plt.close(fig2)
        ax1.set_xlim(time2[:l].min(), time2[:l].max())
        #ax1.set_ylim(-3, 7.5)
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig('figure-7/pr4', dpi=200)
        # do another one with zooms
        ax1.set_xlim(1175, 1475)
        #ax1.set_ylim(-2.5, 7)
        fig1.tight_layout()
        fig1.savefig('figure-7/pr4-z', dpi=200)
        #plt.show()
        plt.close(fig1)

        # Cache it
        torch.save(pred_a_1, 'figure-7/a1-pr4.pt')
        torch.save(pred_d_1, 'figure-7/d1-pr4.pt')
        pred_a_1_pr4 = pred_a_1
        pred_d_1_pr4 = pred_d_1
        del(pred_a_1, pred_d_1)


        #
        # Pr5 (prediction)
        #
        makedirs('figure-7/pr5')
        # Trained Neural ODE
        pred_a_1, pred_d_1 = predict(func_1, timep3, voltagep3, timep3_torch, currentp3, g_nn, true_y0, e_nnf, 'Pr5 (M1)')

        l = int(len(timep3) / 9)  # 9 steps

        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current (nA)')
        for i in range(9):
            ax1.plot(timep3[:l], currentp3[l*i:l*(i+1)] / pred_d_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], c='#7f7f7f', label='__nolegend__' if i else 'Data')
            ax1.plot(timep3[:l], pred_a_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C1',
                     label='__nolegend__' if i else 'Full NN')

            fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Current (nA)')
            ax2.plot(timep3[:l], currentp3[l*i:l*(i+1)] / pred_d_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], c='#7f7f7f', label='__nolegend__' if i else 'Data')
            ax2.plot(timep3[:l], pred_a_1.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', c='C1', label='Full NN')
            ax2.set_xlim(timep3[:l].min(), timep3[:l].max())
            #ax2.set_ylim(-3, 7.5)
            ax2.legend()
            fig2.tight_layout()
            fig2.savefig('figure-7/pr5/s%s' % i, dpi=200)
            plt.close(fig2)
        ax1.set_xlim(timep3[:l].min(), timep3[:l].max())
        #ax1.set_ylim(-3, 7.5)
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig('figure-7/pr5', dpi=300)
        plt.close(fig1)

        # Cache it
        torch.save(pred_a_1, 'figure-7/a1-pr5.pt')
        torch.save(pred_d_1, 'figure-7/d1-pr5.pt')
        pred_a_1_pr5 = pred_a_1
        pred_d_1_pr5 = pred_d_1
        del(pred_a_1, pred_d_1)
    #sys.exit()


#
# Plot
#
ds = 20
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(9, 3.5))

# Set labels
axes[0].set_ylabel(r'$a$')
#axes[1].set_ylabel(r'$a$')
axes[0].set_xlabel(r'$V$ (mV)')
axes[1].set_xlabel(r'$V$ (mV)')
axes[0].set_ylim(0, 1)
axes[1].set_ylim(0, 1)

# Plot!
#axes[0].plot(voltage1[::ds], (current1/pred_d_1_pr3.reshape(-1).cpu().numpy())[::ds])
#axes[0].plot(voltage2[::ds], (current2/pred_d_1_pr4.reshape(-1).cpu().numpy())[::ds])
#axes[0].plot(voltagep3[::ds], (currentp3/pred_d_1_pr5.reshape(-1).cpu().numpy())[::ds])



ds = 1
discont1 = np.append([False], (voltage1[1::ds] != voltage1[:-1:ds]))
discont1[-1] = True  # include the last step
idx1 = np.arange(len(voltage1[::ds]))[discont1]
i = 0
for f in idx1:
    axes[0].plot(voltage1[::ds][i:f]-0.6, pred_a_1_pr3.reshape(-1).cpu().numpy()[::ds][i:f], alpha=0.95, color='#bcbddc', label='__nolegend__' if i!=0 else 'Pr3')
    axes[0].scatter(voltage1[::ds][i]-0.6, pred_a_1_pr3.reshape(-1).cpu().numpy()[::ds][i], alpha=0.95, color='#bcbddc', marker='.')
    axes[0].scatter(voltage1[::ds][f-1]-0.6, pred_a_1_pr3.reshape(-1).cpu().numpy()[::ds][f-1], alpha=0.95, color='#bcbddc', marker='.')
    axes[1].plot(voltage1[::ds][i:f]-0.6, pred_a_1_pr3.reshape(-1).cpu().numpy()[::ds][i:f], alpha=0.4, color='#bcbddc', label='__nolegend__' if i!=0 else 'Pr3')
    axes[1].scatter(voltage1[::ds][i]-0.6, pred_a_1_pr3.reshape(-1).cpu().numpy()[::ds][i], alpha=0.4, color='#bcbddc', marker='.')
    axes[1].scatter(voltage1[::ds][f-1]-0.6, pred_a_1_pr3.reshape(-1).cpu().numpy()[::ds][f-1], alpha=0.4, color='#bcbddc', marker='.')
    i = f

ds2 = 1
discont2 = np.append([False], (voltage2[1::ds2] != voltage2[:-1:ds2]))
discont2[-1] = True  # include the last step
idx2 = np.arange(len(voltage2[::ds2]))[discont2]
i = 0
for f in idx2:
    axes[0].plot(voltage2[::ds2][i:f]-0.6, pred_a_1_pr4.reshape(-1).cpu().numpy()[::ds2][i:f], alpha=0.95, color='#bcbddc', label='__nolegend__' if i!=0 else 'Pr4')
    axes[0].scatter(voltage2[::ds2][i]-0.6, pred_a_1_pr4.reshape(-1).cpu().numpy()[::ds2][i], alpha=0.95, color='#bcbddc', marker='.')
    axes[0].scatter(voltage2[::ds2][f-1]-0.6, pred_a_1_pr4.reshape(-1).cpu().numpy()[::ds2][f-1], alpha=0.95, color='#bcbddc', marker='.')
    axes[1].plot(voltage2[::ds2][i:f]-0.6, pred_a_1_pr4.reshape(-1).cpu().numpy()[::ds2][i:f], alpha=0.4, color='#bcbddc', label='__nolegend__' if i!=0 else 'Pr4')
    axes[1].scatter(voltage2[::ds2][i]-0.6, pred_a_1_pr4.reshape(-1).cpu().numpy()[::ds2][i], alpha=0.4, color='#bcbddc', marker='.')
    axes[1].scatter(voltage2[::ds2][f-1]-0.6, pred_a_1_pr4.reshape(-1).cpu().numpy()[::ds2][f-1], alpha=0.4, color='#bcbddc', marker='.')
    i = f

ds3 = 1
discontp3 = np.append([False], (voltagep3[1::ds3] != voltagep3[:-1:ds3]))
discontp3[-1] = True  # include the last step
idxp3 = np.arange(len(voltagep3[::ds3]))[discontp3]
i = 0
for f in idxp3:
    #print(f, voltagep3[::ds3][i])
    if f in [602733,705957,809181,912405]:
        color = 'C3'
    else:
        color = 'C1'
    axes[1].plot(voltagep3[::ds3][i:f]+0.6, pred_a_1_pr5.reshape(-1).cpu().numpy()[::ds3][i:f], alpha=0.95, color=color, label='__nolegend__' if i!=0 else 'Pr5')
    axes[1].scatter(voltagep3[::ds3][i]+0.6, pred_a_1_pr5.reshape(-1).cpu().numpy()[::ds3][i], alpha=0.95, color=color, marker='.')
    axes[1].scatter(voltagep3[::ds3][f-1]+0.6, pred_a_1_pr5.reshape(-1).cpu().numpy()[::ds3][f-1], alpha=0.95, color=color, marker='.')
    i = f

'''
y1 = [0, 0.354, 0.549, 0.677, 0.75, 0.789, 0.807, 0.816, 0.82, 0.823, 0.8235, 0.8235, 0.8235, 0.8235]
x1 = [-120, -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20]

y2 = [0, 0, 0, 0.0025, 0.01, 0.075, 0.243, 0.651, 0.8235]
x2 = [-120, -100, -90, -80, -60, -40, -20, 0, 20]

y3 = [0.967, 0.822, 0.822, 0.826, 0.833, 0.976]
x3 = [-90, -80, -10, 0, 10, 20]
'''
y1 = [0.005, 0.354, 0.549, 0.822, 0.995, 0.995]
x1 = [-120, -100, -90, -80, -70, 20]

y2 = [0.005, 0.005, 0.005, 0.005, 0.01, 0.075, 0.243, 0.651, 0.995]
x2 = [-120, -100, -90, -80, -60, -40, -20, 0, 20]

axes[0].plot(x1, y1, '--', c='#7f7f7f', alpha=0.5)
axes[0].plot(x2, y2, '--', c='#7f7f7f', alpha=0.5)
#axes[0].plot(x3, y3, '--', c='#7f7f7f', alpha=0.5)
axes[1].plot(x1, y1, '--', c='#7f7f7f', alpha=0.25)
axes[1].plot(x2, y2, '--', c='#7f7f7f', alpha=0.25)
#axes[1].plot(x3, y3, '--', c='#7f7f7f', alpha=0.25)

axes[0].text(-0.1, 1.05, '(A)', size=12, weight='bold',
                va='bottom', ha='right', transform=axes[0].transAxes)
axes[1].text(-0.025, 1.05, '(B)', size=12, weight='bold',
                va='bottom', ha='right', transform=axes[1].transAxes)

'''
axes[1, 0].legend(loc='lower left', bbox_to_anchor=(-0.02, 1.55), ncol=4,
                  columnspacing=4, #handletextpad=1,
                  bbox_transform=axes[1, 0].transAxes)
'''

plt.tight_layout()
plt.savefig('figure-7/fig7.pdf', format='pdf', pad_inches=0.02,
            bbox_inches='tight')
fig.canvas.start_event_loop(sys.float_info.min)  # Silence Tkinter callback
plt.savefig('figure-7/fig7', pad_inches=0.02, dpi=300, bbox_inches='tight')
#plt.show()
plt.close()
