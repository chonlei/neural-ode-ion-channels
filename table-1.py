#!/usr/bin/env python3
import sys
import os
import argparse
import time
import numpy as np
from scipy.interpolate import interp1d

import torch
import torch.nn as nn

parser = argparse.ArgumentParser('IKr NN ODE syn. data RMSE table')
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
gt_true_y0s = [torch.tensor([[0., 0., 1., 0., 0., 0.]]).to(device),  # what you get after holding at +40mV
               torch.tensor([[0., 1., 0., 0., 0., 0.]]).to(device)]  # (roughly) what you get after holding at -80mV

# B1.2 in https://physoc.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1113%2FJP275733&file=tjp12905-sup-0001-textS1.pdf#page=4
e = torch.tensor([-88.4]).to(device)  # assume we know
g = torch.tensor([1]).to(device)  # assume we know

g_nn = g

#
#
#
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

makedirs('table-1')


#
# Load data
#
raw_data1 = np.loadtxt('data/pr3-steady-activation-cell-5.csv', delimiter=',', skiprows=1)
time1 = raw_data1[:, 0]
time1_torch = torch.from_numpy(raw_data1[:, 0]).to(device)
#current1 = raw_data1[:, 1]
voltage1 = raw_data1[:, 2]

raw_data2 = np.loadtxt('data/pr4-inactivation-cell-5.csv', delimiter=',', skiprows=1)
time2 = raw_data2[:, 0]
time2_torch = torch.from_numpy(raw_data2[:, 0]).to(device)
#current2 = raw_data2[:, 1]
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
#currentp1 = prediction1[:, 1]
voltagep1 = prediction1[:, 2]


#
#
#
class GroundTruth(nn.Module):
    def __init__(self):
        super(GroundTruth, self).__init__()

        # Best of 10 fits for data herg25oc1 cell B06 (seed 542811797)
        self.p1 = 5.94625498751561316e-02 * 1e-3
        self.p2 = 1.21417701632850410e+02 * 1e-3
        self.p3 = 4.76436985414236425e+00 * 1e-3
        self.p4 = 3.49383233960778904e-03 * 1e-3
        self.p5 = 9.62243079990877703e+01 * 1e-3
        self.p6 = 2.26404683824047979e+01 * 1e-3
        self.p7 = 8.00924780462999131e+00 * 1e-3
        self.p8 = 2.43749808069009823e+01 * 1e-3
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
        c1, c2, i, ic1, ic2, o = torch.unbind(y[0])

        try:
            v = self._v(t).to(device)
        except ValueError:
            v = torch.tensor([-80]).to(device)

        a1 = self.p1 * torch.exp(self.p2 * v)
        b1 = self.p3 * torch.exp(-self.p4 * v)
        bh = self.p5 * torch.exp(self.p6 * v)
        ah = self.p7 * torch.exp(-self.p8 * v)
        a2 = self.p9 * torch.exp(self.p10 * v)
        b2 = self.p11 * torch.exp(-self.p12 * v)

        dc1dt = a1 * c2 + ah * ic1 + b2 * o - (b1 + bh + a2) * c1
        dc2dt = b1 * c1 + ah * ic2 - (a1 + bh) * c2
        didt = a2 * ic1 + bh * o - (b2 + ah) * i
        dic1dt = a1 * ic2 + bh * c1 + b2 * i - (b1 + ah + a2) * ic1
        dic2dt = b1 * ic1 + bh * c2 - (ah + a1) * ic2
        dodt = a2 * c1 + ah * i - (b2 + bh) * o

        return torch.stack([dc1dt[0], dc2dt[0], didt[0], dic1dt[0], dic2dt[0], dodt[0]])



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

        # Best of 10 fits (M10) for data herg25oc1 cell B06 (seed 542811797)
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

        # https://github.com/CardiacModelling/hERGRapidCharacterisation/blob/master/room-temperature-only/out/herg25oc1/herg25oc1-staircaseramp-B06-solution-542811797.txt
        self.p1 = 1.12592345582957387e-01 * 1e-3
        self.p2 = 8.26751134920666146e+01 * 1e-3
        self.p3 = 3.38768033864048357e-02 * 1e-3
        self.p4 = 4.67106147665183542e+01 * 1e-3
        # 
        # Best of 10 fits (M10) for data herg25oc1 cell B06 (seed 542811797)
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
true_model = GroundTruth().to(device)

func_o = Lambda().to(device)
func_o.eval()

func_1 = ODEFunc1_6().to(device)
func_1.load_state_dict(torch.load('d1/model-state-dict.pt'))
func_1.eval()

func_2 = ODEFunc1_6_2().to(device)
func_2.load_state_dict(torch.load('d2/model-state-dict.pt', map_location=torch.device('cpu')))
func_2.eval()

# Load more prediction data
prediction2 = np.loadtxt('data/cell-5.csv', delimiter=',', skiprows=1)
timep2 = prediction2[:, 0]
timep2_torch = torch.from_numpy(prediction2[:, 0]).to(device)
#currentp2 = prediction2[:, 1]
voltagep2 = prediction2[:, 2]

prediction3 = np.loadtxt('data/pr5-deactivation-cell-5.csv', delimiter=',', skiprows=1)
timep3 = prediction3[:, 0]
timep3_torch = torch.from_numpy(prediction3[:, 0]).to(device)
#currentp3 = prediction3[:, 1]
voltagep3 = prediction3[:, 2]

true_gy0 = gt_true_y0s[1]  # (roughly holding at -80mV)
true_y0 = true_y0s[1]  # (roughly holding at -80mV)

def sim_data(func, time, voltage, time_torch, gg, y0, noise=None):
    func.set_fixed_form_voltage_protocol(time, voltage)
    with torch.no_grad():
        pred_y = odeint(func, y0, time_torch).to(device)
    sim = gg * pred_y[:, 0, -1] * (func._v(time_torch).to(device) - e)
    if noise:
        sim += torch.from_numpy(np.random.normal(0, noise, sim.cpu().numpy().shape)).to(device)
    return sim.reshape(-1).cpu().numpy()

def predict(func, time, voltage, time_torch, data, gg, y0, name):
    func.set_fixed_form_voltage_protocol(time, voltage)
    with torch.no_grad():
        pred_y = odeint(func, y0, time_torch).to(device)
    pred_yo = gg * pred_y[:, 0, 0] * pred_y[:, 0, 1] * (func._v(time_torch).to(device) - e)
    loss = torch.mean(torch.abs(pred_yo - torch.from_numpy(data).to(device)))
    print('{:s} prediction | Total Loss {:.6f}'.format(name, loss.item()))
    return pred_yo


if args.cached:
    current1 = torch.from_numpy(torch.load('table-1/yc-pr3.pt')).to(device)
    pred_y_o_pr3 = torch.load('table-1/yo-pr3.pt')
    pred_y_1_pr3 = torch.load('table-1/y1-pr3.pt')
    pred_y_2_pr3 = torch.load('table-1/y2-pr3.pt')
    current2 = torch.from_numpy(torch.load('table-1/yc-pr4.pt')).to(device)
    pred_y_o_pr4 = torch.load('table-1/yo-pr4.pt')
    pred_y_1_pr4 = torch.load('table-1/y1-pr4.pt')
    pred_y_2_pr4 = torch.load('table-1/y2-pr4.pt')
    currentp1 = torch.from_numpy(torch.load('table-1/yc-aps.pt')).to(device)
    pred_y_o_aps = torch.load('table-1/yo-aps.pt')
    pred_y_1_aps = torch.load('table-1/y1-aps.pt')
    pred_y_2_aps = torch.load('table-1/y2-aps.pt')
    currentp2 = torch.from_numpy(torch.load('table-1/yc-sinewave.pt')).to(device)
    pred_y_o_sin = torch.load('table-1/yo-sinewave.pt')
    pred_y_1_sin = torch.load('table-1/y1-sinewave.pt')
    pred_y_2_sin = torch.load('table-1/y2-sinewave.pt')
    currentp3 = torch.from_numpy(torch.load('table-1/yc-pr5.pt')).to(device)
    pred_y_o_pr5 = torch.load('table-1/yo-pr5.pt')
    pred_y_1_pr5 = torch.load('table-1/y1-pr5.pt')
    pred_y_2_pr5 = torch.load('table-1/y2-pr5.pt')
else:
    with torch.no_grad():
        ###
        ### Training protocols
        ###

        #
        # Pr4
        #
        # True model
        current2 = sim_data(true_model, time2, voltage2, time2_torch, g, true_gy0, noise_sigma)
        # Trained Neural ODE
        pred_y_o = predict(func_o, time2, voltage2, time2_torch, current2, g, true_y0, 'Pr4 (Mo)')
        pred_y_1 = predict(func_1, time2, voltage2, time2_torch, current2, g_nn, true_y0, 'Pr4 (M1)')
        pred_y_2 = predict(func_2, time2, voltage2, time2_torch, current2, g_nn, true_y0, 'Pr4 (M2)')

        #
        # Pr3
        #
        # True model
        current1 = sim_data(true_model, time1, voltage1, time1_torch, g, true_gy0, noise_sigma)
        # Trained Neural ODE
        pred_y_o = predict(func_o, time1, voltage1, time1_torch, current1, g, true_y0, 'Pr3 (Mo)')
        pred_y_1 = predict(func_1, time1, voltage1, time1_torch, current1, g_nn, true_y0, 'Pr3 (M1)')
        pred_y_2 = predict(func_2, time1, voltage1, time1_torch, current1, g_nn, true_y0, 'Pr3 (M2)')

        # Cache it
        torch.save(current2, 'table-1/yc-pr4.pt')
        torch.save(pred_y_o, 'table-1/yo-pr4.pt')
        torch.save(pred_y_1, 'table-1/y1-pr4.pt')
        torch.save(pred_y_2, 'table-1/y2-pr4.pt')
        pred_y_o_pr4 = pred_y_o
        pred_y_1_pr4 = pred_y_1
        pred_y_2_pr4 = pred_y_2
        del(pred_y_o, pred_y_1, pred_y_2)

        # Cache it
        torch.save(current1, 'table-1/yc-pr3.pt')
        torch.save(pred_y_o, 'table-1/yo-pr3.pt')
        torch.save(pred_y_1, 'table-1/y1-pr3.pt')
        torch.save(pred_y_2, 'table-1/y2-pr3.pt')
        pred_y_o_pr3 = pred_y_o
        pred_y_1_pr3 = pred_y_1
        pred_y_2_pr3 = pred_y_2
        del(pred_y_o, pred_y_1, pred_y_2)

        #
        # Pr5
        #
        # True model
        currentp3 = sim_data(true_model, timep3, voltagep3, timep3_torch, g, true_gy0, noise_sigma)
        # Trained Neural ODE
        pred_y_o = predict(func_o, timep3, voltagep3, timep3_torch, currentp3, g, true_y0, 'Pr5 (Mo)')
        pred_y_1 = predict(func_1, timep3, voltagep3, timep3_torch, currentp3, g_nn, true_y0, 'Pr5 (M1)')
        pred_y_2 = predict(func_2, timep3, voltagep3, timep3_torch, currentp3, g_nn, true_y0, 'Pr5 (M2)')

        # Cache it
        torch.save(currentp3, 'table-1/yc-pr5.pt')
        torch.save(pred_y_o, 'table-1/yo-pr5.pt')
        torch.save(pred_y_1, 'table-1/y1-pr5.pt')
        torch.save(pred_y_2, 'table-1/y2-pr5.pt')
        pred_y_o_pr5 = pred_y_o
        pred_y_1_pr5 = pred_y_1
        pred_y_2_pr5 = pred_y_2
        del(pred_y_o, pred_y_1, pred_y_2)

        #
        # Sinewave
        #
        # True model
        currentp2 = sim_data(true_model, timep2, voltagep2, timep2_torch, g, true_gy0, noise_sigma)
        # Trained Neural ODE
        pred_y_o = predict(func_o, timep2, voltagep2, timep2_torch, currentp2, g, true_y0, 'Sinewave (Mo)')
        pred_y_1 = predict(func_1, timep2, voltagep2, timep2_torch, currentp2, g_nn, true_y0, 'Sinewave (M1)')
        pred_y_2 = predict(func_2, timep2, voltagep2, timep2_torch, currentp2, g_nn, true_y0, 'Sinewave (M2)')

        # Cache it
        torch.save(currentp2, 'table-1/yc-sinewave.pt')
        torch.save(pred_y_o, 'table-1/yo-sinewave.pt')
        torch.save(pred_y_1, 'table-1/y1-sinewave.pt')
        torch.save(pred_y_2, 'table-1/y2-sinewave.pt')
        pred_y_o_sin = pred_y_o
        pred_y_1_sin = pred_y_1
        pred_y_2_sin = pred_y_2
        del(pred_y_o, pred_y_1, pred_y_2)

        #
        # APs
        #
        # True model
        currentp1 = sim_data(true_model, timep1, voltagep1, timep1_torch, g, true_gy0, noise_sigma)
        # Trained Neural ODE
        pred_y_o = predict(func_o, timep1, voltagep1, timep1_torch, currentp1, g, true_y0, 'APs (Mo)')
        pred_y_1 = predict(func_1, timep1, voltagep1, timep1_torch, currentp1, g_nn, true_y0, 'APs (M1)')
        pred_y_2 = predict(func_2, timep1, voltagep1, timep1_torch, currentp1, g_nn, true_y0, 'APs (M2)')

        # Cache it
        torch.save(currentp1, 'table-1/yc-aps.pt')
        torch.save(pred_y_o, 'table-1/yo-aps.pt')
        torch.save(pred_y_1, 'table-1/y1-aps.pt')
        torch.save(pred_y_2, 'table-1/y2-aps.pt')
        pred_y_o_aps = pred_y_o
        pred_y_1_aps = pred_y_1
        pred_y_2_aps = pred_y_2
        del(pred_y_o, pred_y_1, pred_y_2)


def loss(x, y):
    return torch.sqrt(torch.mean(torch.square(x - y))).item()

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

with open('table-1/table-1.txt', 'w') as f:
    f.write(output)
