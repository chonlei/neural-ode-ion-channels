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
#sns.set_theme()
#print(sns.axes_style())
#sns.set(rc={'figure.facecolor':'#ffffff', 'axes.facecolor':'#ffffff'})
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from smoothing import smooth

import torch
import torch.nn as nn

parser = argparse.ArgumentParser('IKr NN ODE syn. data plot 1')
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
gt_true_y0s = [torch.tensor([[0., 0.]]).to(device),  # what you get after holding at +40mV
               torch.tensor([[0., 0.]]).to(device)]  # (roughly) what you get after holding at -80mV

#
#
#
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

makedirs('figure-1')


#
# Load data
#
raw_data1 = np.loadtxt('data/pr3-steady-activation-cell-5.csv', delimiter=',', skiprows=1)
time1 = raw_data1[:, 0]
time1_torch = torch.from_numpy(raw_data1[:, 0]).to(device)
voltage1 = raw_data1[:, 2]

prediction3 = np.loadtxt('data/pr5-deactivation-cell-5.csv', delimiter=',', skiprows=1)
timep3 = prediction3[:, 0]
timep3_torch = torch.from_numpy(prediction3[:, 0]).to(device)
voltagep3 = prediction3[:, 2]


#
#
#
class GroundTruth_a(nn.Module):
    def __init__(self):
        super(GroundTruth_a, self).__init__()

        # Best of 10 fits for data herg25oc1 cell B06 (seed 542811797)
        self.p1 = 5.94625498751561316e-02 * 1e-3
        self.p2 = 1.21417701632850410e+02 * 1e-3
        self.p3 = 4.76436985414236425e+00 * 1e-3
        self.p4 = 3.49383233960778904e-03 * 1e-3
        self.p9 = 2.06822607368134157e+02 * 1e-3
        self.p10 = 3.30791433507312362e+01 * 1e-3
        self.p11 = 1.26069071928587784e+00 * 1e-3
        self.p12 = 2.24844970727316245e+01 * 1e-3

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
        a, u = torch.unbind(y, dim=1)

        try:
            v = self._v(t).to(device)
        except ValueError:
            v = torch.tensor([-80]).to(device)

        a1 = self.p1 * torch.exp(self.p2 * v)
        b1 = self.p3 * torch.exp(-self.p4 * v)
        a2 = self.p9 * torch.exp(self.p10 * v)
        b2 = self.p11 * torch.exp(-self.p12 * v)

        K1 = a1*a2 + b1*b2 + a1*b2
        K2 = a1 + a2 + b1 + b2
        K3 = a1*a2

        dudt = - K1 * a - K2 * u + K3

        return torch.stack([u[0], dudt[0]]).reshape(1, -1)


#
#
#
class Lambda(nn.Module):
    def __init__(self):
        super(Lambda, self).__init__()

        # Fit to GroundTruth model using train-d0.py, in `./d0/model-parameters.txt`.
        self.p1 = 5.694588454735844622e-05
        self.p2 = 1.172955815858964107e-01
        self.p3 = 3.522672347205991382e-05
        self.p4 = 4.972513487995382231e-02
        # Best of 10 fits (M10) for data herg25oc1 cell B06 (seed 542811797) - assume correct.
        self.p5 = 9.62243079990877703e+01 * 1e-3
        self.p6 = 2.26404683824047979e+01 * 1e-3
        self.p7 = 8.00924780462999131e+00 * 1e-3
        self.p8 = 2.43749808069009823e+01 * 1e-3

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

#
#
#
true_model = GroundTruth_a().to(device)

func_o = Lambda().to(device)
func_o.eval()

true_gy0 = gt_true_y0s[1]  # (roughly holding at -80mV)
true_y0 = true_y0s[1]  # (roughly holding at -80mV)

def sim_gt(func, time, voltage, time_torch, y0):
    func.set_fixed_form_voltage_protocol(time, voltage)
    with torch.no_grad():
        pred_y = odeint(func, y0, time_torch).to(device)
    return pred_y[:, 0, 0], pred_y[:, 0, 1]

if args.cached:
    current1 = torch.load('figure-1/yc-pr3.pt')
    currentp3 = torch.load('figure-1/yc-pr5.pt')
    current1o = torch.load('figure-1/yco-pr3.pt')
    currentp3o = torch.load('figure-1/yco-pr5.pt')
else:
    with torch.no_grad():
        ###
        ### Training protocols
        ###
        current1 = sim_gt(true_model, time1, voltage1, time1_torch, true_gy0)
        currentp3 = sim_gt(true_model, timep3, voltagep3, timep3_torch, true_gy0)
        torch.save(current1, 'figure-1/yc-pr3.pt')
        torch.save(currentp3, 'figure-1/yc-pr5.pt')

        current1o = sim_gt(func_o, time1, voltage1, time1_torch, true_y0)
        currentp3o = sim_gt(func_o, timep3, voltagep3, timep3_torch, true_y0)
        torch.save(current1, 'figure-1/yco-pr3.pt')
        torch.save(currentp3, 'figure-1/yco-pr5.pt')

x1 = torch.reshape(torch.linspace(-120, 60, 50).to(device), (-1, 1))
x2 = torch.reshape(torch.linspace(0, 1, 50).to(device), (-1, 1))
X1, X2 = torch.meshgrid(x1.reshape(-1), x2.reshape(-1))

# Figure
fig = plt.figure(figsize=(8.5, 3.75))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

#
# ax1
#

# Data
ds = 1#30
discont1 = np.append([False], (voltage1[1::ds] != voltage1[:-1:ds]))
discont1[-1] = True  # include the last step
idx1 = np.arange(len(voltage1[::ds]))[discont1]
i = 0
for f in idx1:
    k1 = func_o.p1 * np.exp(func_o.p2 * voltage1[::ds][i:f])
    k2 = func_o.p3 * np.exp(-func_o.p4 * voltage1[::ds][i:f])
    dadt = k1 * (1. - current1o[0][::ds].numpy()[i:f]) - k2 * current1o[0][::ds].numpy()[i:f]

    ax1.plot(voltage1[::ds][i:f], current1o[0][::ds].numpy()[i:f], dadt, alpha=0.8, color='#fdbb84', label='__nolegend__' if i!=0 else 'Pr3')
    ax1.scatter(voltage1[::ds][i], current1o[0][::ds].numpy()[i], dadt[0], alpha=0.8, color='#fdbb84', marker='.')
    ax1.scatter(voltage1[::ds][f-1], current1o[0][::ds].numpy()[f-1], dadt[-1], alpha=0.8, color='#fdbb84', marker='.')
    i = f
ds2 = 1#80
discontp3 = np.append([False], (voltagep3[1::ds2] != voltagep3[:-1:ds2]))
discontp3[-1] = True  # include the last step
idxp3 = np.arange(len(voltagep3[::ds2]))[discontp3]
i = 0
for f in idxp3:
    k1 = func_o.p1 * np.exp(func_o.p2 * voltagep3[::ds][i:f])
    k2 = func_o.p3 * np.exp(-func_o.p4 * voltagep3[::ds][i:f])
    dadt = k1 * (1. - currentp3o[0][::ds2].numpy()[i:f]) - k2 * currentp3o[0][::ds2].numpy()[i:f]

    ax1.plot(voltagep3[::ds2][i:f], currentp3o[0][::ds2].numpy()[i:f], dadt, alpha=0.8, color='#bcbddc', label='__nolegend__' if i!=0 else 'Pr5')
    ax1.scatter(voltagep3[::ds2][i], currentp3o[0][::ds2].numpy()[i], dadt[0], alpha=0.8, color='#bcbddc', marker='.')
    ax1.scatter(voltagep3[::ds2][f-1], currentp3o[0][::ds2].numpy()[f-1], dadt[-1], alpha=0.8, color='#bcbddc', marker='.')
    i = f

# Original
k1 = func_o.p1 * np.exp(func_o.p2 * X1)
k2 = func_o.p3 * np.exp(-func_o.p4 * X1)
dadt_o = k1 * (1 - X2) - k2 * X2
ax1.plot_surface(X1.cpu().numpy(), X2.cpu().numpy(), dadt_o.reshape(50, 50).detach().cpu().numpy(), color='C0', alpha=0.5)#, label='Original')

ax1.view_init(30, -125)

ax1.set_xlabel(r'$V$ (mV)')
ax1.set_ylabel(r'$a$')
ax1.zaxis.set_rotate_label(False)
ax1.set_zlabel(r'$\frac{da}{dt}$', rotation=0, fontsize=13)


#
# ax2
#

# Data
ds = 1#30
discont1 = np.append([False], (voltage1[1::ds] != voltage1[:-1:ds]))
discont1[-1] = True  # include the last step
idx1 = np.arange(len(voltage1[::ds]))[discont1]
i = 0
for f in idx1:
    ax2.plot(voltage1[::ds][i:f], current1[0][::ds].numpy()[i:f], current1[1][::ds].numpy()[i:f], alpha=0.8, color='#fdbb84', label='__nolegend__' if i!=0 else 'Pr3')
    ax2.scatter(voltage1[::ds][i], current1[0][::ds].numpy()[i], current1[1][::ds].numpy()[i], alpha=0.8, color='#fdbb84', marker='.')
    ax2.scatter(voltage1[::ds][f-1], current1[0][::ds].numpy()[f-1], current1[1][::ds].numpy()[f-1], alpha=0.8, color='#fdbb84', marker='.')
    i = f
ds2 = 1#80
discontp3 = np.append([False], (voltagep3[1::ds2] != voltagep3[:-1:ds2]))
discontp3[-1] = True  # include the last step
idxp3 = np.arange(len(voltagep3[::ds2]))[discontp3]
i = 0
for f in idxp3:
    ax2.plot(voltagep3[::ds2][i:f], currentp3[0][::ds2].numpy()[i:f], currentp3[1][::ds2].numpy()[i:f], alpha=0.8, color='#bcbddc', label='__nolegend__' if i!=0 else 'Pr5')
    ax2.scatter(voltagep3[::ds2][i], currentp3[0][::ds2].numpy()[i], currentp3[1][::ds2].numpy()[i], alpha=0.8, color='#bcbddc', marker='.')
    ax2.scatter(voltagep3[::ds2][f-1], currentp3[0][::ds2].numpy()[f-1], currentp3[1][::ds2].numpy()[f-1], alpha=0.8, color='#bcbddc', marker='.')
    i = f

# Original
k1 = func_o.p1 * np.exp(func_o.p2 * X1)
k2 = func_o.p3 * np.exp(-func_o.p4 * X1)
dadt_o = k1 * (1 - X2) - k2 * X2
ax2.plot_surface(X1.cpu().numpy(), X2.cpu().numpy(), dadt_o.reshape(50, 50).detach().cpu().numpy(), color='C0', alpha=0.5)#, label='Original')

ax2.view_init(30, -125)

ax2.set_xlabel(r'$V$ (mV)')
ax2.set_ylabel(r'$a$')
ax2.zaxis.set_rotate_label(False)
ax2.set_zlabel(r'$\frac{da}{dt}$', rotation=0, fontsize=13)

ax2.legend(ncol=2)

ax1.text2D(-0.05, 0.925, '(A)', size=12, weight='bold',
                va='bottom', ha='right', transform=ax1.transAxes)
ax2.text2D(-0.05, 0.925, '(B)', size=12, weight='bold',
                va='bottom', ha='right', transform=ax2.transAxes)

#
# Done
#
plt.savefig('figure-1/fig1.pdf', format='pdf', pad_inches=0.02,
            bbox_inches='tight')
fig.canvas.start_event_loop(sys.float_info.min)  # Silence Tkinter callback
plt.savefig('figure-1/fig1', pad_inches=0.02, dpi=300, bbox_inches='tight')
#plt.show()
plt.close()
