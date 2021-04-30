import sys
import os
import argparse
import time
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from smoothing import smooth

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

parser = argparse.ArgumentParser('IKr simple syn. fit with NN-d.')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--cached', action='store_true')
parser.add_argument('--pred', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# Set random seed
np.random.seed(0)
torch.manual_seed(0)

noise_sigma = 0.1

use_pt3 = False

prediction_protocol = np.loadtxt('test-protocols/ap2hz.csv', skiprows=1, delimiter=',')
prediction_protocol[:, 0] *= 1e3  # s -> ms
#
t1 = torch.linspace(0., 8000, 80001).to(device)
t2 = torch.linspace(0., 10000, 100001).to(device)
if use_pt3:
    t3 = torch.linspace(0., 5000, 50001).to(device)
    mask3s = [np.ones(len(t3), dtype=bool) for i in range(6)]
mask1 = np.ones(len(t1), dtype=bool)
# Filter out +/- 5 time points for each step (numerical issue for estimating derivatives)
mask1[list(range(9995, 10050))
    + list(range(59995, 60050))
    + list(range(69995, 70050))
    + list(range(74995, 75050))] \
    = False
mask2 = np.ones(len(t2), dtype=bool)
mask2[list(range(9995, 10050))
    + list(range(29995, 30050))
    + list(range(89995, 90050))
    + list(range(94995, 95050))] \
    = False
prediction_t = torch.linspace(0., 3000, 1501).to(device)
#
# Activation
#
protocol_batches1 = []
pt1 = np.linspace(0., 8000., 80001)  # 8 seconds, 0.1 ms interval
template_v = np.zeros(pt1.shape)
template_v[:10000] = -80
# template_v[10000:60000] to be set
template_v[60000:70000] = -40
template_v[70000:75000] = -120
template_v[75000:] = -80
for v_i in [-60, -40, -20, 0, 20, 40, 60]:
    v = np.copy(template_v)
    v[10000:60000] = v_i
    protocol_batches1.append(np.array([pt1, v]).T)
#
# Deactivation
#
protocol_batches2 = []
pt2 = np.linspace(0., 10000., 100001)  # 10 seconds, 0.1 ms interval
template_v = np.zeros(pt2.shape)
template_v[:10000] = -80
template_v[10000:30000] = 50
# template_v[30000:90000] to be set
template_v[90000:95000] = -120
template_v[95000:] = -80
for v_i in [-120, -110, -100, -90, -80, -70, -60, -50, -40]:
    v = np.copy(template_v)
    v[30000:90000] = v_i
    protocol_batches2.append(np.array([pt2, v]).T)
if use_pt3:
    #
    # Activation time constant at 40mV
    #
    protocol_batches3 = []
    pt3 = np.linspace(0., 5000., 50001)  # 5 seconds, 0.1 ms interval
    for i, t_i in enumerate([30, 100, 300, 1000, 3000, 10000]):  # 0.1ms
        v = np.zeros(pt3.shape)
        v[:10000] = -80
        v[10000:10000+t_i] = 40
        v[10000+t_i:35000+t_i] = -120
        v[35000+t_i:] = -80
        protocol_batches3.append(np.array([pt3, v]).T)

        # NOTE data time has the same index as the protocol time
        mask3s[i][list(range(9995, 10005))
            + list(range(9995+t_i, 10005+t_i))
            + list(range(34995+t_i, 35005+t_i))] \
            = False
true_y0s = [torch.tensor([[1., 0.]]).to(device),  # what you get after holding at +40mV
            torch.tensor([[0., 1.]]).to(device)]  # (roughly) what you get after holding at -80mV
gt_true_y0s = [torch.tensor([[1., 0.]]).to(device),  # what you get after holding at +40mV
               torch.tensor([[0., 1.]]).to(device)]  # (roughly) what you get after holding at -80mV

#
#
#
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

makedirs('s2')


#
#
#
class Lambda(nn.Module):
    def __init__(self):
        super(Lambda, self).__init__()

        # https://github.com/CardiacModelling/hERGRapidCharacterisation/blob/master/room-temperature-only/out/herg25oc1/herg25oc1-staircaseramp-B06-solution-542811797.txt
        self.p1 = 1.12592345582957387e-01 * 1e-3
        self.p2 = 8.26751134920666146e+01 * 1e-3
        self.p3 = 3.38768033864048357e-02 * 1e-3
        self.p4 = 4.67106147665183542e+01 * 1e-3
        self.p5 = 8.47769667061995875e+01 * 1e-3
        self.p6 = 2.04001345352499328e+01 * 1e-3
        self.p7 = 1.02860743916105211e+01 * 1e-3
        self.p8 = 2.78201179336874098e+01 * 1e-3

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

        dadt = k1 * (1. - a) - k2 * a
        drdt = -k3 * r + k4 * (1. - r)

        return torch.stack([dadt[0], drdt[0]])



class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

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
        self.p5 = 8.47769667061995875e+01 * 1e-3
        self.p6 = 2.04001345352499328e+01 * 1e-3
        self.p7 = 1.02860743916105211e+01 * 1e-3
        self.p8 = 2.78201179336874098e+01 * 1e-3

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
if args.pred:
    true_model = Lambda()

    func = ODEFunc().to(device)
    func.load_state_dict(torch.load('s2/model-state-dict.pt'))
    func.eval()

    prediction_protocol2 = np.loadtxt('test-protocols/staircase.csv', skiprows=1, delimiter=',')
    prediction_protocol2[:, 0] *= 1e3  # s -> ms
    prediction_t2 = torch.linspace(0., 15000, 7501).to(device)

    prediction_protocol3 = np.loadtxt('test-protocols/sinewave.csv', skiprows=1, delimiter=',')
    prediction_protocol3[:, 0] *= 1e3  # s -> ms
    prediction_t3 = torch.linspace(0., 8000, 4001).to(device)

    prediction_protocol4 = np.loadtxt('test-protocols/aps.csv', skiprows=1, delimiter=',')
    prediction_protocol4[:, 0] *= 1e3  # s -> ms
    prediction_t4 = torch.linspace(0., 8000, 4001).to(device)

    ii = 999

    with torch.no_grad():
        ###
        ### Predict unseen protocols
        ###

        #
        # AP 2Hz
        #
        # Ground truth
        true_y0 = gt_true_y0s[1]  # (roughly holding at -80mV)
        true_model.set_fixed_form_voltage_protocol(prediction_protocol[:, 0], prediction_protocol[:, 1])
        prediction_y = odeint(true_model, true_y0, prediction_t, method='dopri5')
        prediction_yo = prediction_y[:, 0, 0] * prediction_y[:, 0, 1] * (true_model._v(prediction_t) + 86)
        # Trained Neural ODE
        true_y0 = true_y0s[1]  # (roughly holding at -80mV)
        func.set_fixed_form_voltage_protocol(prediction_protocol[:, 0], prediction_protocol[:, 1])
        pred_y = odeint(func, true_y0, prediction_t).to(device)
        pred_yo = pred_y[:, 0, 0] * pred_y[:, 0, 1] * (func._v(prediction_t).to(device) + 86)
        loss = torch.mean(torch.abs(pred_yo - prediction_yo))
        print('AP 2Hz prediction | Total Loss {:.6f}'.format(loss.item()))

        prediction_yo += torch.from_numpy(np.random.normal(0, noise_sigma, prediction_t.cpu().numpy().shape)).to(device)

        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('t')
        ax1.set_ylabel('i')
        ax1.plot(prediction_t.cpu().numpy(), prediction_yo.reshape(-1).cpu().numpy(), 'g-')
        ax1.plot(prediction_t.cpu().numpy(), pred_yo.reshape(-1).cpu().numpy(), 'b--')
        ax1.set_xlim(prediction_t.cpu().min(), prediction_t.cpu().max())
        fig1.tight_layout()
        fig1.savefig('s2/{:03d}-ap2hz'.format(ii), dpi=200)
        plt.close(fig1)

        #
        # Kylie's APs
        #
        # Ground truth
        true_y0 = gt_true_y0s[1]  # (roughly holding at -80mV)
        true_model.set_fixed_form_voltage_protocol(prediction_protocol4[:, 0], prediction_protocol4[:, 1])
        prediction_y4 = odeint(true_model, true_y0, prediction_t4, method='dopri5')
        prediction_yo4 = prediction_y4[:, 0, 0] * prediction_y4[:, 0, 1] * (true_model._v(prediction_t4) + 86)
        # Trained Neural ODE
        true_y0 = true_y0s[1]  # (roughly holding at -80mV)
        func.set_fixed_form_voltage_protocol(prediction_protocol4[:, 0], prediction_protocol4[:, 1])
        pred_y4 = odeint(func, true_y0, prediction_t4).to(device)
        pred_yo4 = pred_y4[:, 0, 0] * pred_y4[:, 0, 1] * (func._v(prediction_t4).to(device) + 86)
        loss = torch.mean(torch.abs(pred_yo4 - prediction_yo4))
        print('APs prediction | Total Loss {:.6f}'.format(loss.item()))

        prediction_yo4 += torch.from_numpy(np.random.normal(0, noise_sigma, prediction_t4.cpu().numpy().shape)).to(device)

        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('t')
        ax1.set_ylabel('i')
        ax1.plot(prediction_t4.cpu().numpy(), prediction_yo4.reshape(-1).cpu().numpy(), 'g-')
        ax1.plot(prediction_t4.cpu().numpy(), pred_yo4.reshape(-1).cpu().numpy(), 'b--')
        ax1.set_xlim(prediction_t4.cpu().min(), prediction_t4.cpu().max())
        fig1.tight_layout()
        fig1.savefig('s2/{:03d}-aps'.format(ii), dpi=200)
        plt.close(fig1)

        #
        # Sinewave
        #
        # Ground truth
        true_y0 = gt_true_y0s[1]  # (roughly holding at -80mV)
        true_model.set_fixed_form_voltage_protocol(prediction_protocol3[:, 0], prediction_protocol3[:, 1])
        prediction_y3 = odeint(true_model, true_y0, prediction_t3, method='dopri5')
        prediction_yo3 = prediction_y3[:, 0, 0] * prediction_y3[:, 0, 1] * (true_model._v(prediction_t3) + 86)
        # Trained Neural ODE
        true_y0 = true_y0s[1]  # (roughly holding at -80mV)
        func.set_fixed_form_voltage_protocol(prediction_protocol3[:, 0], prediction_protocol3[:, 1])
        pred_y3 = odeint(func, true_y0, prediction_t3).to(device)
        pred_yo3 = pred_y3[:, 0, 0] * pred_y3[:, 0, 1] * (func._v(prediction_t3).to(device) + 86)
        loss = torch.mean(torch.abs(pred_yo3 - prediction_yo3))
        print('Sinewave prediction | Total Loss {:.6f}'.format(loss.item()))

        prediction_yo4 += torch.from_numpy(np.random.normal(0, noise_sigma, prediction_t3.cpu().numpy().shape)).to(device)

        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('t')
        ax1.set_ylabel('i')
        ax1.plot(prediction_t3.cpu().numpy(), prediction_yo3.reshape(-1).cpu().numpy(), 'g-')
        ax1.plot(prediction_t3.cpu().numpy(), pred_yo3.reshape(-1).cpu().numpy(), 'b--')
        ax1.set_xlim(prediction_t3.cpu().min(), prediction_t3.cpu().max())
        fig1.tight_layout()
        fig1.savefig('s2/{:03d}-sinewave'.format(ii), dpi=200)
        plt.close(fig1)

        #
        # Staircase
        #
        # Ground truth
        true_y0 = gt_true_y0s[1]  # (roughly holding at -80mV)
        true_model.set_fixed_form_voltage_protocol(prediction_protocol2[:, 0], prediction_protocol2[:, 1])
        prediction_y2 = odeint(true_model, true_y0, prediction_t2, method='dopri5')
        prediction_yo2 = prediction_y2[:, 0, 0] * prediction_y2[:, 0, 1] * (true_model._v(prediction_t2) + 86)
        # Trained Neural ODE
        true_y0 = true_y0s[1]  # (roughly holding at -80mV)
        func.set_fixed_form_voltage_protocol(prediction_protocol2[:, 0], prediction_protocol2[:, 1])
        pred_y2 = odeint(func, true_y0, prediction_t2).to(device)
        pred_yo2 = pred_y2[:, 0, 0] * pred_y2[:, 0, 1] * (func._v(prediction_t2).to(device) + 86)
        loss = torch.mean(torch.abs(pred_yo2 - prediction_yo2))
        print('Staircase prediction | Total Loss {:.6f}'.format(loss.item()))

        prediction_yo2 += torch.from_numpy(np.random.normal(0, noise_sigma, prediction_t2.cpu().numpy().shape)).to(device)

        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('t')
        ax1.set_ylabel('i')
        ax1.plot(prediction_t2.cpu().numpy(), prediction_yo2.reshape(-1).cpu().numpy(), 'g-')
        ax1.plot(prediction_t2.cpu().numpy(), pred_yo2.reshape(-1).cpu().numpy(), 'b--')
        ax1.set_xlim(prediction_t2.cpu().min(), prediction_t2.cpu().max())
        fig1.tight_layout()
        fig1.savefig('s2/{:03d}-staircase'.format(ii), dpi=200)
        plt.close(fig1)

        #
        # Pr 3 Activation
        #
        t = torch.linspace(0., 8000., 8001).to(device)  # 8 seconds, 1 ms interval
        template_v = np.zeros(t.cpu().numpy().shape)
        template_v[:1000] = -80
        # template_v[1000:6000] to be set
        template_v[6000:7000] = -40
        template_v[7000:7500] = -120
        template_v[7500:] = -80
        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('t')
        ax1.set_ylabel('i')
        print('Activation prediction:')
        for v_i in [-60, -40, -20, 0, 20, 40, 60]:
            v = np.copy(template_v)
            v[1000:6000] = v_i

            # Ground truth
            true_y0 = gt_true_y0s[1]  # (roughly holding at -80mV)
            true_model.set_fixed_form_voltage_protocol(t.cpu().numpy(), v)
            yt = odeint(true_model, true_y0, t, method='dopri5')
            ot = yt[:, 0, 0] * yt[:, 0, 1] * (true_model._v(t) + 86)
            # Trained Neural ODE
            true_y0 = true_y0s[1]  # (roughly holding at -80mV)
            func.set_fixed_form_voltage_protocol(t.cpu().numpy(), v)
            yp = odeint(func, true_y0, t).to(device)
            op = yp[:, 0, 0] * yp[:, 0, 1] * (func._v(t).to(device) + 86)
            loss = torch.mean(torch.abs(op - ot))
            print('    {:.1f}mV | Total Loss {:.6f}'.format(v_i, loss.item()))

            ot += torch.from_numpy(np.random.normal(0, noise_sigma, t.cpu().numpy().shape)).to(device)

            ax1.plot(t.cpu().numpy(), ot.reshape(-1).cpu().numpy(), c='#7f7f7f')
            ax1.plot(t.cpu().numpy(), op.reshape(-1).cpu().numpy(), c='C0', ls='--')
        ax1.set_xlim(t.cpu().min(), t.cpu().max())
        fig1.tight_layout()
        fig1.savefig('s2/{:03d}-act'.format(ii), dpi=200)
        plt.close(fig1)

        #
        # Pr 5 Deactivation
        #
        t = torch.linspace(0., 10000., 10001).to(device)  # 8 seconds, 1 ms interval
        template_v = np.zeros(t.cpu().numpy().shape)
        template_v[:1000] = -80
        template_v[1000:3000] = 50
        # template_v[3000:9000] to be set
        template_v[9000:9500] = -120
        template_v[9500:] = -80
        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('t')
        ax1.set_ylabel('i')
        print('Deactivation prediction:')
        for v_i in [-120, -110, -100, -90, -80, -70, -60, -50, -40]:
            v = np.copy(template_v)
            v[3000:9000] = v_i

            # Ground truth
            true_y0 = gt_true_y0s[1]  # (roughly holding at -80mV)
            true_model.set_fixed_form_voltage_protocol(t.cpu().numpy(), v)
            yt = odeint(true_model, true_y0, t, method='dopri5')
            ot = yt[:, 0, 0] * yt[:, 0, 1] * (true_model._v(t) + 86)
            # Trained Neural ODE
            true_y0 = true_y0s[1]  # (roughly holding at -80mV)
            func.set_fixed_form_voltage_protocol(t.cpu().numpy(), v)
            yp = odeint(func, true_y0, t).to(device)
            op = yp[:, 0, 0] * yp[:, 0, 1] * (func._v(t).to(device) + 86)
            loss = torch.mean(torch.abs(op - ot))
            print('    {:.1f}mV | Total Loss {:.6f}'.format(v_i, loss.item()))

            ot += torch.from_numpy(np.random.normal(0, noise_sigma, t.cpu().numpy().shape)).to(device)

            ax1.plot(t.cpu().numpy(), ot.reshape(-1).cpu().numpy(), c='#7f7f7f')
            ax1.plot(t.cpu().numpy(), op.reshape(-1).cpu().numpy(), c='C0', ls='--')
        ax1.set_xlim(t.cpu().min(), t.cpu().max())
        fig1.tight_layout()
        fig1.savefig('s2/{:03d}-deact'.format(ii), dpi=200)
        plt.close(fig1)

        #
        # Pr 2 Activation time constant at 40mV
        #
        t = torch.linspace(0., 5000., 5001).to(device)  # 8 seconds, 1 ms interval
        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('t')
        ax1.set_ylabel('i')
        print('Activation time constant at 40mV prediction:')
        for t_i in [3, 10, 30, 100, 300, 1000]:
            v = np.zeros(t.cpu().numpy().shape)
            v[:1000] = -80
            v[1000:1000+t_i] = 40
            v[1000+t_i:3500+t_i] = -120
            v[3500+t_i:] = -80

            # Ground truth
            true_y0 = gt_true_y0s[1]  # (roughly holding at -80mV)
            true_model.set_fixed_form_voltage_protocol(t.cpu().numpy(), v)
            yt = odeint(true_model, true_y0, t, method='dopri5')
            ot = yt[:, 0, 0] * yt[:, 0, 1] * (true_model._v(t) + 86)
            # Trained Neural ODE
            true_y0 = true_y0s[1]  # (roughly holding at -80mV)
            func.set_fixed_form_voltage_protocol(t.cpu().numpy(), v)
            yp = odeint(func, true_y0, t).to(device)
            op = yp[:, 0, 0] * yp[:, 0, 1] * (func._v(t).to(device) + 86)
            loss = torch.mean(torch.abs(op - ot))
            print('    {:.1f}ms | Total Loss {:.6f}'.format(t_i, loss.item()))

            ot += torch.from_numpy(np.random.normal(0, noise_sigma, t.cpu().numpy().shape)).to(device)

            ax1.plot(t.cpu().numpy(), ot.reshape(-1).cpu().numpy(), c='#7f7f7f')
            ax1.plot(t.cpu().numpy(), op.reshape(-1).cpu().numpy(), c='C0', ls='--')
        ax1.set_xlim(t.cpu().min(), t.cpu().max())
        fig1.tight_layout()
        fig1.savefig('s2/{:03d}-atau'.format(ii), dpi=200)
        plt.close(fig1)


    sys.exit()
#
#
#


#
# Generate syn data from the ground truth model
#
true_model = Lambda()

# ap 2hz for prediction
true_y0 = gt_true_y0s[1]
true_model.set_fixed_form_voltage_protocol(prediction_protocol[:, 0], prediction_protocol[:, 1])
with torch.no_grad():
    prediction_y = odeint(true_model, true_y0, prediction_t, method='dopri5')
    prediction_yo = prediction_y[:, 0, 0] * prediction_y[:, 0, 1] * (true_model._v(prediction_t) + 86)

if not args.cached:
    true_y_batches1 = []
    true_yo_batches1 = []
    true_y_batches2 = []
    true_yo_batches2 = []
    true_y_batches3 = []
    true_yo_batches3 = []
    with torch.no_grad():
        for protocol in protocol_batches1:
            true_y0 = gt_true_y0s[1]
            true_model.set_fixed_form_voltage_protocol(protocol[:, 0], protocol[:, 1])
            true_y = odeint(true_model, true_y0, t1, method='dopri5')
            true_y_batches1.append(true_y)
            true_yo_batches1.append(true_y[:, 0, 0] * true_y[:, 0, 1] * (true_model._v(t1) + 86) +
                    torch.from_numpy(np.random.normal(0, noise_sigma, t1.cpu().numpy().shape)).to(device))

        for protocol in protocol_batches2:
            true_y0 = gt_true_y0s[1]
            true_model.set_fixed_form_voltage_protocol(protocol[:, 0], protocol[:, 1])
            true_y = odeint(true_model, true_y0, t2, method='dopri5')
            true_y_batches2.append(true_y)
            true_yo_batches2.append(true_y[:, 0, 0] * true_y[:, 0, 1] * (true_model._v(t2) + 86) +
                    torch.from_numpy(np.random.normal(0, noise_sigma, t2.cpu().numpy().shape)).to(device))

        if use_pt3:
            for protocol in protocol_batches3:
                true_y0 = gt_true_y0s[1]
                true_model.set_fixed_form_voltage_protocol(protocol[:, 0], protocol[:, 1])
                true_y = odeint(true_model, true_y0, t3, method='dopri5')
                true_y_batches3.append(true_y)
                true_yo_batches3.append(true_y[:, 0, 0] * true_y[:, 0, 1] * (true_model._v(t3) + 86) +
                        torch.from_numpy(np.random.normal(0, noise_sigma, t3.cpu().numpy().shape)).to(device))


###
### 'post-processing': estimating dadt and a
###
if args.cached:
    v_batches = torch.load('s2/v.pt')
    a_batches = torch.load('s2/a.pt')
    dadt_batches = torch.load('s2/dadt.pt')
else:
    skip = 5  # not accurate for the first few time points for estimating derivatives
    sparse = 11#21  # use less data points
    dvdt_constant = torch.tensor([0]).to(device)  # for now yes
    e = torch.tensor([-86.]).to(device)  # assume we know
    g = torch.tensor([1.]).to(device)  # assume we know
    r_batches1 = []  # assume we know to the extent of which we can ignore its discrepancy
    r_batches2 = []
    if use_pt3:
        r_batches3 = []
    with torch.no_grad():
        m = ODEFunc().to(device)
        for protocol in protocol_batches1:
            true_y0 = true_y0s[1]
            m.set_fixed_form_voltage_protocol(protocol[:, 0], protocol[:, 1])
            true_y = odeint(m, true_y0, t1, method='dopri5')
            r_batches1.append(true_y[:, 0, 1])
        for protocol in protocol_batches2:
            true_y0 = true_y0s[1]
            m.set_fixed_form_voltage_protocol(protocol[:, 0], protocol[:, 1])
            true_y = odeint(m, true_y0, t2, method='dopri5')
            r_batches2.append(true_y[:, 0, 1])
        if use_pt3:
            for protocol in protocol_batches3:
                true_y0 = true_y0s[1]
                m.set_fixed_form_voltage_protocol(protocol[:, 0], protocol[:, 1])
                true_y = odeint(m, true_y0, t3, method='dopri5')
                r_batches3.append(true_y[:, 0, 1])
    v_batches1 = []
    for protocol in protocol_batches1:
        true_model.set_fixed_form_voltage_protocol(protocol[:, 0], protocol[:, 1])
        v_batches1.append(true_model._v(t1)[0])
    v_batches2 = []
    for protocol in protocol_batches2:
        true_model.set_fixed_form_voltage_protocol(protocol[:, 0], protocol[:, 1])
        v_batches2.append(true_model._v(t2)[0])
    if use_pt3:
        v_batches3 = []
        for protocol in protocol_batches3:
            true_model.set_fixed_form_voltage_protocol(protocol[:, 0], protocol[:, 1])
            v_batches3.append(true_model._v(t3)[0])
    drdt_batches1 = []
    drdt_batches2 = []
    for r, v in zip(r_batches1, v_batches1):
        k3 = m.p5 * torch.exp(m.p6 * v)
        k4 = m.p7 * torch.exp(-m.p8 * v)
        drdt = -k3 * r + k4 * (1. - r)
        drdt_batches1.append(drdt)
    for r, v in zip(r_batches2, v_batches2):
        k3 = m.p5 * torch.exp(m.p6 * v)
        k4 = m.p7 * torch.exp(-m.p8 * v)
        drdt = -k3 * r + k4 * (1. - r)
        drdt_batches2.append(drdt)
    if use_pt3:
        drdt_batches3 = []
        for r, v in zip(r_batches3, v_batches3):
            k3 = m.p5 * torch.exp(m.p6 * v)
            k4 = m.p7 * torch.exp(-m.p8 * v)
            drdt = -k3 * r + k4 * (1. - r)
            drdt_batches3.append(drdt)
    i_batches1 = true_yo_batches1
    i_batches2 = true_yo_batches2
    didt_batches1 = []
    didt_batches2 = []
    for j, (i, protocol) in enumerate(zip(i_batches1, protocol_batches1)):
        ii = i.cpu().numpy().reshape(-1)
        pt = protocol[:, 0]
        pv = protocol[:, 1]
        t_split = pt[np.append([False], pv[:-1] != pv[1:])]
        t_split = np.append(t_split, pt[-1] + 1)
        t_i = 0
        io = []
        didto = []
        for t_f in t_split:
            idx = np.where((t1 >= t_i) & (t1 < t_f))[0]
            tfit = t1.cpu().numpy()[idx]
            ifit = smooth(ii[idx], 61)[30:-30]  # smoothing
            spl = UnivariateSpline(tfit, ifit, k=3)
            spl.set_smoothing_factor(0)
            io = np.append(io, spl(tfit))
            didto = np.append(didto, spl.derivative()(tfit))
            t_i = t_f
        i_batches1[j] = torch.from_numpy(io).to(device)
        didt_batches1.append(torch.from_numpy(didto).to(device))
    for j, (i, protocol) in enumerate(zip(i_batches2, protocol_batches2)):
        ii = i.cpu().numpy().reshape(-1)
        pt = protocol[:, 0]
        pv = protocol[:, 1]
        t_split = pt[np.append([False], pv[:-1] != pv[1:])]
        t_split = np.append(t_split, pt[-1] + 1)
        t_i = 0
        io = []
        didto = []
        for t_f in t_split:
            idx = np.where((t2 >= t_i) & (t2 < t_f))[0]
            tfit = t2.cpu().numpy()[idx]
            ifit = smooth(ii[idx], 61)[30:-30]  # smoothing
            spl = UnivariateSpline(tfit, ifit, k=3)
            spl.set_smoothing_factor(0)
            io = np.append(io, spl(tfit))
            didto = np.append(didto, spl.derivative()(tfit))
            t_i = t_f
        i_batches2[j] = torch.from_numpy(io).to(device)
        didt_batches2.append(torch.from_numpy(didto).to(device))
    if use_pt3:
        didt_batches3 = []
        i_batches3 = true_yo_batches3
        for j, (i, protocol) in enumerate(zip(i_batches3, protocol_batches3)):
            ii = i.cpu().numpy().reshape(-1)
            pt = protocol[:, 0]
            pv = protocol[:, 1]
            t_split = pt[np.append([False], pv[:-1] != pv[1:])]
            t_split = np.append(t_split, pt[-1] + 1)
            t_i = 0
            io = []
            didto = []
            for t_f in t_split:
                idx = np.where((t3 >= t_i) & (t3 < t_f))[0]
                tfit = t3.cpu().numpy()[idx]
                ifit = smooth(ii[idx], 61)[30:-30]  # smoothing
                spl = UnivariateSpline(tfit, ifit, k=3)
                spl.set_smoothing_factor(0)
                io = np.append(io, spl(tfit))
                didto = np.append(didto, spl.derivative()(tfit))
                t_i = t_f
            i_batches3[j] = torch.from_numpy(io).to(device)
            didt_batches3.append(torch.from_numpy(didto).to(device))
    # Calculate a and dadt
    a_batches1 = []
    dadt_batches1 = []
    for j, (i, r, v, drdt, didt) in enumerate(zip(i_batches1, r_batches1, v_batches1, drdt_batches1, didt_batches1)):
        ii = i.reshape(-1)
        a = ii / (g * r * (v - e))
        if np.all(v.cpu().numpy() == v.cpu().numpy()[0]) or True:  # all steps even different values
            dvdt = dvdt_constant
        else:
            spl = UnivariateSpline(range(len(v.cpu().numpy())), v.cpu().numpy(), k=3, s=0)
            dvdt = torch.from_numpy(spl.derivative()(range(len(v.cpu().numpy()))))
        dadt = r ** (-1) * (
                (didt / g - a * r * dvdt) / (v - e)
                - a * drdt
                )
        a_batches1.append(a)
        dadt_batches1.append(dadt)
    a_batches2 = []
    dadt_batches2 = []
    for j, (i, r, v, drdt, didt) in enumerate(zip(i_batches2, r_batches2, v_batches2, drdt_batches2, didt_batches2)):
        ii = i.reshape(-1)
        a = ii / (g * r * (v - e))
        if np.all(v.cpu().numpy() == v.cpu().numpy()[0]) or True:  # all steps even different values
            dvdt = dvdt_constant
        else:
            spl = UnivariateSpline(range(len(v.cpu().numpy())), v.cpu().numpy(), k=3, s=0)
            dvdt = torch.from_numpy(spl.derivative()(range(len(v.cpu().numpy()))))
        dadt = r ** (-1) * (
                (didt / g - a * r * dvdt) / (v - e)
                - a * drdt
                )
        a_batches2.append(a)
        dadt_batches2.append(dadt)
    if use_pt3:
        a_batches3 = []
        dadt_batches3 = []
        for j, (i, r, v, drdt, didt) in enumerate(zip(i_batches3, r_batches3, v_batches3, drdt_batches3, didt_batches3)):
            ii = i.reshape(-1)
            a = ii / (g * r * (v - e))
            if np.all(v.cpu().numpy() == v.cpu().numpy()[0]) or True:  # all steps even different values
                dvdt = dvdt_constant
            else:
                spl = UnivariateSpline(range(len(v.cpu().numpy())), v.cpu().numpy(), k=3, s=0)
                dvdt = torch.from_numpy(spl.derivative()(range(len(v.cpu().numpy()))))
            dadt = r ** (-1) * (
                    (didt / g - a * r * dvdt) / (v - e)
                    - a * drdt
                    )
            a_batches3.append(a)
            dadt_batches3.append(dadt)
    # To tensors
    for i, (v, a, dadt) in enumerate(zip(v_batches1, a_batches1, dadt_batches1)):
        v_batches1[i] = v[mask1,...][skip::sparse]
        a_batches1[i] = a[mask1,...][skip::sparse]
        dadt_batches1[i] = dadt[mask1,...][skip::sparse]
    for i, (v, a, dadt) in enumerate(zip(v_batches2, a_batches2, dadt_batches2)):
        v_batches2[i] = v[mask2,...][skip::sparse]
        a_batches2[i] = a[mask2,...][skip::sparse]
        dadt_batches2[i] = dadt[mask2,...][skip::sparse]
    if use_pt3:
        for i, (v, a, dadt) in enumerate(zip(v_batches3, a_batches3, dadt_batches3)):
            v_batches3[i] = v[mask3s[i],...][skip::sparse]
            a_batches3[i] = a[mask3s[i],...][skip::sparse]
            dadt_batches3[i] = dadt[mask3s[i],...][skip::sparse]
    if use_pt3:
        v_batches = torch.cat(v_batches1 + v_batches2 + v_batches3).to(device)
        a_batches = torch.cat(a_batches1 + a_batches2 + a_batches3).to(device)
        dadt_batches = torch.cat(dadt_batches1 + dadt_batches2 + dadt_batches3).to(device)
    else:
        v_batches = torch.cat(v_batches1 + v_batches2).to(device)
        a_batches = torch.cat(a_batches1 + a_batches2).to(device)
        dadt_batches = torch.cat(dadt_batches1 + dadt_batches2).to(device)

    # Cache it
    torch.save(v_batches, 's2/v.pt')
    torch.save(a_batches, 's2/a.pt')
    torch.save(dadt_batches, 's2/dadt.pt')

if args.debug:
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(v_batches.reshape(-1).detach().cpu().numpy(), a_batches.reshape(-1).detach().cpu().numpy(),
            dadt_batches.reshape(-1).detach().cpu().numpy())
    ax.set_xlabel('V')
    ax.set_ylabel('a')
    ax.set_zlabel('da/dt')
    plt.show()
###
###
###



if __name__ == '__main__':

    ii = 0

    func = ODEFunc().to(device)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    ###
    ### To predict
    ###
    x1 = torch.reshape(torch.linspace(-130, 70, 50).to(device), (-1, 1))
    xx1 = x1 / func.vrange
    x2 = torch.reshape(torch.linspace(0, 1, 50).to(device), (-1, 1))
    X1, X2 = torch.meshgrid(x1.reshape(-1), x2.reshape(-1))
    XX1, X2 = torch.meshgrid(xx1.reshape(-1), x2.reshape(-1))
    XX = torch.stack([XX1.reshape(-1), X2.reshape(-1)]).T

    ###

    ###
    ### Training
    ###
    #'''
    model_dadt = func._dadt(a_batches.reshape(-1), v_batches.reshape(-1))

    x_av = torch.stack([v_batches.reshape(-1) / func.vrange, a_batches.reshape(-1)]).T
    y_dadt = dadt_batches.reshape(-1)

    # Keep only 0 < a < 1
    to_keep = (x_av[:, 1] > 0) & (x_av[:, 1] < 1)
    x_av = x_av[to_keep, :]
    model_dadt = model_dadt[to_keep]
    y_dadt = y_dadt[to_keep]

    if True:
        func2 = ODEFunc().to(device)
        func2.load_state_dict(torch.load('s1/model-state-dict.pt'))
        func2.eval()
        with torch.no_grad():
            p = func2.net(x_av.float()).to(device) / func2.netscale
            loss = loss_fn(p.reshape(-1), y_dadt.float())
            print('Target Loss', loss.item())
        del(loss, p, func2)

    opt = optim.Adam(func.net.parameters(), lr=0.001)
    # gamma = decaying factor
    scheduler = StepLR(opt, step_size=100, gamma=0.9)  # 0.9**(4000steps/100) ~ 0.016
    for itr in range(4000):
        p = func.net(x_av.float()).to(device) / func.netscale
        p += model_dadt.reshape(-1, 1)
        loss = loss_fn(p.reshape(-1), y_dadt.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        # Decay Learning Rate
        scheduler.step()
        if (itr % 400) == 0:
            print('Iter', itr, 'LR', opt.param_groups[0]['lr'], 'Loss', loss.item())
        del(loss, p)
    #'''

    # Save model
    torch.save(func.state_dict(), 's2/model-state-dict.pt')
    torch.save(func, 's2/model-entire.pt')
    # To load model:
    # func = TheModelClass(*args, **kwargs)
    # func.set_fixed_form_voltage_protocol(protocol[:, 0], protocol[:, 1])
    # func.load_state_dict(torch.load('s2/model-state-dict.pt'))
    # func.eval()
    #
    # Or:
    # func = torch.load('s2/model-entire.pt')
    # func.eval()

    with torch.no_grad():
        true_y0 = true_y0s[1]
        func.set_fixed_form_voltage_protocol(prediction_protocol[:, 0], prediction_protocol[:, 1])
        pred_y = odeint(func, true_y0, prediction_t).to(device)
        pred_yo = pred_y[:, 0, 0] * pred_y[:, 0, 1] * (func._v(prediction_t).to(device) + 86)
        loss = torch.mean(torch.abs(pred_yo - prediction_yo))
        print('Pretraining | Total Loss {:.6f}'.format(loss.item()))
        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('t')
        ax1.set_ylabel('i')
        ax1.plot(prediction_t.cpu().numpy(), prediction_yo.reshape(-1).cpu().numpy(), 'g-')
        ax1.plot(prediction_t.cpu().numpy(), pred_yo.reshape(-1).cpu().numpy(), 'b--')
        ax1.set_xlim(prediction_t.cpu().min(), prediction_t.cpu().max())
        fig1.tight_layout()
        fig1.savefig('s2/{:03d}'.format(ii), dpi=200)
        plt.close(fig1)

        ax = plt.axes(projection='3d')
        ax.scatter(v_batches.reshape(-1).detach().cpu().numpy(), a_batches.reshape(-1).detach().cpu().numpy(),
                dadt_batches.reshape(-1).detach().cpu().numpy())
        pred = (func.net(XX) / func.netscale).reshape(-1) + func._dadt(X2.reshape(-1), X1.reshape(-1)).reshape(-1)
        ax.plot_surface(X1.cpu().numpy(), X2.cpu().numpy(), pred.reshape(50, 50).detach().cpu().numpy(), color='C1')
        ax.set_xlabel('V')
        ax.set_ylabel('a')
        ax.set_zlabel('da/dt')
        plt.savefig('s2/rates3d-{:03d}'.format(ii), dpi=200)
        #plt.show()
        plt.close()

        ii += 1
    ###
    ###
