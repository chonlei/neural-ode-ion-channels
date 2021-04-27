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

parser = argparse.ArgumentParser('IKr real data fit with NN-f.')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--niters', type=int, default=4000)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--pred', action='store_true')
parser.add_argument('--cached', action='store_true')
args = parser.parse_args()

from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# Set random seed
np.random.seed(0)
torch.manual_seed(0)

noise_sigma = 0.1

true_y0s = [torch.tensor([[1., 0.]]).to(device),  # what you get after holding at +40mV
            torch.tensor([[0., 1.]]).to(device)]  # (roughly) what you get after holding at -80mV

# B1.2 in https://physoc.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1113%2FJP275733&file=tjp12905-sup-0001-textS1.pdf#page=4
e = torch.tensor([-88.4]).to(device)  # assume we know
# https://github.com/CardiacModelling/FourWaysOfFitting/blob/master/method-3/cell-5-fit-3-run-001.txt
g = torch.tensor([0.133898199260611944]).to(device)  # assume we know
g *= 1.2  # just because we see a-gate gets to ~1.2 at some point (in prt V=50), so can absorb that into the g.
e -= 5    # just because in pr4, at -90 mV, a-gates became negative, meaning e < -90mV; and only if adding an extra -5mV, a ~ [0, 1].

#
#
#
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

makedirs('r1')


suffix = '-2'

def save_ckp(state, isbest=False, checkpoint_dir='r1'):
    f_path = checkpoint_dir + '/checkpoint%s.pt' % suffix
    torch.save(state, f_path)
    if isbest:
        b_path = checkpoint_dir + '/best-model-checkpoint%s.pt' % suffix
        torch.save(state, b_path)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

ckp_path = 'r1/checkpoint%s.pt' % suffix


#
# Load data
#
raw_data1 = np.loadtxt('data/pr3-steady-activation-cell-5.csv', delimiter=',', skiprows=1)
raw_data2 = np.loadtxt('data/cell-5.csv', delimiter=',', skiprows=1)
raw_data3 = np.loadtxt('data/pr5-deactivation-cell-5.csv', delimiter=',', skiprows=1)
time1 = raw_data1[:, 0]
time1_torch = torch.from_numpy(raw_data1[:, 0]).to(device)
current1 = raw_data1[:, 1]
voltage1 = raw_data1[:, 2]
time2 = raw_data2[:, 0]
time2_torch = torch.from_numpy(raw_data2[:, 0]).to(device)
current2 = raw_data2[:, 1]
voltage2 = raw_data2[:, 2]
time3 = raw_data3[:, 0]
time3_torch = torch.from_numpy(raw_data3[:, 0]).to(device)
current3 = raw_data3[:, 1]
voltage3 = raw_data3[:, 2]

#
# Make filters
#
n_ms = 7
dt = 0.1  # ms
n_points = int(n_ms / dt)
change_pt1 = np.append([True], ~(voltage1[1:] != voltage1[:-1]))
cap_mask1 = np.copy(change_pt1)
for i in range(n_points):
    cap_mask1 = cap_mask1 & np.roll(change_pt1, i + 1)
change_pt2 = np.append([True], ~(voltage2[1:] != voltage2[:-1]))
#sinewave_pts = (time2 > 3000.1) & (time2 < 6500.1)
sinewave_pts = (time2 > 3000.1 + 1e-8) & (time2 < 6500.1 - 1e-8)  # allow floating point error
change_pt2 |= sinewave_pts
cap_mask2 = np.copy(change_pt2)
for i in range(n_points):
    cap_mask2 = cap_mask2 & np.roll(change_pt2, i + 1)
change_pt3 = np.append([True], ~(voltage3[1:] != voltage3[:-1]))
cap_mask3 = np.copy(change_pt3)
for i in range(n_points):
    cap_mask3 = cap_mask3 & np.roll(change_pt3, i + 1)
# A bigger/final filter mask
extra_points = 20  # for numerical derivative or smoothing issue
mask1 = np.copy(cap_mask1)
for i in range(extra_points):
    mask1 = mask1 & np.roll(change_pt1, i + n_points + 1)
    mask1 = mask1 & np.roll(change_pt1, -i - 1)
mask2 = np.copy(cap_mask2)
for i in range(extra_points):
    mask2 = mask2 & np.roll(change_pt2, i + n_points + 1)
    mask2 = mask2 & np.roll(change_pt2, -i - 1)
mask3 = np.copy(cap_mask3)
for i in range(extra_points):
    mask3 = mask3 & np.roll(change_pt3, i + n_points + 1)
    mask3 = mask3 & np.roll(change_pt3, -i - 1)

prediction1 = np.loadtxt('data/ap-cell-5.csv', delimiter=',', skiprows=1)
timep1 = prediction1[:, 0]
timep1_torch = torch.from_numpy(prediction1[:, 0]).to(device)
currentp1 = prediction1[:, 1]
voltagep1 = prediction1[:, 2]

#
#
#
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

        dadt = self.net(torch.stack([nv[0], a[0]]).float()) / self.netscale

        return torch.stack([dadt[0], drdt[0]]).reshape(1, -1)

#
#
#


#
#
#
if args.pred:
    func = ODEFunc().to(device)
    #func.load_state_dict(torch.load('r1/model-state-dict.pt'))
    best_checkpoint = torch.load('r1/best-model-checkpoint.pt')
    func.load_state_dict(best_checkpoint['state_dict'])
    print('Best checkpoint loss:', best_checkpoint['loss'])
    #func = torch.load('r1/model-entire.pt').to(device)
    func.eval()

    ii = 999

    # Load more prediction data
    prediction2 = np.loadtxt('data/cell-5.csv', delimiter=',', skiprows=1)
    timep2 = prediction2[:, 0]
    timep2_torch = torch.from_numpy(prediction2[:, 0]).to(device)
    currentp2 = prediction2[:, 1]
    voltagep2 = prediction2[:, 2]

    # Look at the fitted surface
    v_batches = torch.load('r1/v.pt')
    a_batches = torch.load('r1/a.pt')
    dadt_batches = torch.load('r1/dadt.pt')

    x1 = torch.reshape(torch.linspace(-130, 70, 50).to(device), (-1, 1))
    xx1 = x1 / func.vrange
    x2 = torch.reshape(torch.linspace(0, 1, 50).to(device), (-1, 1))
    X1, X2 = torch.meshgrid(x1.reshape(-1), x2.reshape(-1))
    XX1, X2 = torch.meshgrid(xx1.reshape(-1), x2.reshape(-1))
    XX = torch.stack([XX1.reshape(-1), X2.reshape(-1)]).T

    ax = plt.axes(projection='3d')
    ax.scatter(v_batches.reshape(-1).detach().cpu().numpy(), a_batches.reshape(-1).detach().cpu().numpy(),
            dadt_batches.reshape(-1).detach().cpu().numpy())
    pred = func.net(XX) / func.netscale
    ax.plot_surface(X1.cpu().numpy(), X2.cpu().numpy(), pred.reshape(50, 50).detach().cpu().numpy(), color='C1')
    ax.set_ylim([0, 1])
    ax.set_xlabel('V')
    ax.set_ylabel('a')
    ax.set_zlabel('da/dt')
    plt.savefig('r1/rates3d-{:03d}'.format(ii), dpi=200)
    #plt.show()
    plt.close()

    true_y0 = true_y0s[1]  # (roughly holding at -80mV)

    with torch.no_grad():
        ###
        ### Predict unseen protocols
        ###

        #
        # APs
        #
        # Trained Neural ODE
        func.set_fixed_form_voltage_protocol(timep1, voltagep1)
        pred_y = odeint(func, true_y0, timep1_torch).to(device)
        pred_yo = g * pred_y[:, 0, 0] * pred_y[:, 0, 1] * (func._v(timep1_torch).to(device) - e)
        loss = torch.mean(torch.abs(pred_yo - torch.from_numpy(currentp1).to(device)))
        print('APs prediction | Total Loss {:.6f}'.format(loss.item()))

        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current (nA)')
        ax1.plot(timep1, currentp1, c='#7f7f7f', label='Data')
        ax1.plot(timep1, pred_yo.reshape(-1).cpu().numpy(), '--', label='Prediction')
        ax1.set_xlim(timep1.min(), timep1.max())
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig('r1/{:03d}-aps'.format(ii), dpi=200)
        #plt.show()
        plt.close(fig1)

        #
        # Sinewave
        #
        # Trained Neural ODE
        func.set_fixed_form_voltage_protocol(timep2, voltagep2)
        pred_y = odeint(func, true_y0, timep2_torch).to(device)
        pred_yo = g * pred_y[:, 0, 0] * pred_y[:, 0, 1] * (func._v(timep2_torch).to(device) - e)
        loss = torch.mean(torch.abs(pred_yo - torch.from_numpy(currentp2).to(device)))
        print('Sinewave prediction | Total Loss {:.6f}'.format(loss.item()))

        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current (nA)')
        ax1.plot(timep2, currentp2, c='#7f7f7f', label='Data')
        ax1.plot(timep2, pred_yo.reshape(-1).cpu().numpy(), '--', label='Prediction')
        ax1.set_xlim(timep2.min(), timep2.max())
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig('r1/{:03d}-sinewave'.format(ii), dpi=200)
        #plt.show()
        plt.close(fig1)

        #
        # Pr3
        #
        # Trained Neural ODE
        func.set_fixed_form_voltage_protocol(time1, voltage1)
        pred_y = odeint(func, true_y0, time1_torch).to(device)
        pred_yo = g * pred_y[:, 0, 0] * pred_y[:, 0, 1] * (func._v(time1_torch).to(device) - e)
        loss = torch.mean(torch.abs(pred_yo - torch.from_numpy(current1).to(device)))
        print('Pr3 prediction | Total Loss {:.6f}'.format(loss.item()))

        l = int(len(time1) / 7)  # 7 steps

        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current (nA)')
        for i in range(7):
            ax1.plot(time1[:l], current1[l*i:l*(i+1)], c='#7f7f7f', label='__nolegend__' if i else 'Data')
            ax1.plot(time1[:l], pred_yo.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', label='__nolegend__' if i else 'Prediction')
        ax1.set_xlim(time1[:l].min(), time1[:l].max())
        ax1.set_ylim(-4, 1.9)
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig('r1/{:03d}-pr3'.format(ii), dpi=200)
        # do another one with zooms
        ax1.set_xlim(5000, 7000)
        ax1.set_ylim(-2, 1.7)
        fig1.tight_layout()
        fig1.savefig('r1/{:03d}-pr3-z'.format(ii), dpi=200)
        #plt.show()
        plt.close(fig1)

        #
        # Pr4
        #
        # Trained Neural ODE
        func.set_fixed_form_voltage_protocol(time2, voltage2)
        pred_y = odeint(func, true_y0, time2_torch).to(device)
        pred_yo = g * pred_y[:, 0, 0] * pred_y[:, 0, 1] * (func._v(time2_torch).to(device) - e)
        loss = torch.mean(torch.abs(pred_yo - torch.from_numpy(current2).to(device)))
        print('Pr4 prediction | Total Loss {:.6f}'.format(loss.item()))

        l = int(len(time2) / 16)  # 16 steps

        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Current (nA)')
        for i in range(16):
            ax1.plot(time2[:l], current2[l*i:l*(i+1)], c='#7f7f7f', label='__nolegend__' if i else 'Data')
            ax1.plot(time2[:l], pred_yo.reshape(-1).cpu().numpy()[l*i:l*(i+1)], '--', label='__nolegend__' if i else 'Prediction')
        ax1.set_xlim(time2[:l].min(), time2[:l].max())
        ax1.set_ylim(-3, 7.5)
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig('r1/{:03d}-pr4'.format(ii), dpi=200)
        # do another one with zooms
        ax1.set_xlim(1175, 1475)
        ax1.set_ylim(-2.5, 7)
        fig1.tight_layout()
        fig1.savefig('r1/{:03d}-pr4-z'.format(ii), dpi=200)
        #plt.show()
        plt.close(fig1)

    sys.exit()
#
#
#


if args.cached:
    t_batches = torch.load('r1/t.pt')
    v_batches = torch.load('r1/v.pt')
    a_batches = torch.load('r1/a.pt')
    dadt_batches = torch.load('r1/dadt.pt')
    d2adt2_batches = torch.load('r1/d2adt2.pt')
else:
    ###
    ### 'post-processing': estimating dadt and a
    ###
    skip = 10  # not accurate for the first few time points for estimating derivatives
    sparse = 7  # use less data points
    r_batches1 = []  # assume we know to the extent of which we can ignore its discrepancy
    r_batches2 = []
    r_batches3 = []
    v_batches1 = []
    v_batches2 = []
    v_batches3 = []
    with torch.no_grad():
        m = ODEFunc().to(device)

        true_y0 = true_y0s[1]
        m.set_fixed_form_voltage_protocol(time1, voltage1)
        true_y = odeint(m, true_y0, time1_torch, method='dopri5')
        r_batches1.append(true_y[:, 0, 1].to(device))
        v_batches1.append(m._v(time1_torch)[0].to(device))

        true_y0 = true_y0s[1]
        m.set_fixed_form_voltage_protocol(time2, voltage2)
        true_y = odeint(m, true_y0, time2_torch, method='dopri5')
        r_batches2.append(true_y[:, 0, 1].to(device))
        v_batches2.append(m._v(time2_torch)[0].to(device))

        true_y0 = true_y0s[1]
        m.set_fixed_form_voltage_protocol(time3, voltage3)
        true_y = odeint(m, true_y0, time3_torch, method='dopri5')
        r_batches3.append(true_y[:, 0, 1].to(device))
        v_batches3.append(m._v(time3_torch)[0].to(device))
    i_batches1 = [torch.from_numpy(current1).to(device)]
    i_batches2 = [torch.from_numpy(current2).to(device)]
    i_batches3 = [torch.from_numpy(current3).to(device)]
    # Calculate a and dadt
    import pints
    from scipy import optimize
    x02 = [0.7, 1./50., 0.2, 1./100., 0.1, 1./200., 0.01]
    x0 = [1, 1./100., 0.5, 1./200., 0.25, 1./400., 0.1]
    def tri_exp(t, x):
        a, b, c, d, e, f, g = x
        return a * np.exp(-b * t) + c * np.exp(-d * t) + e * np.exp(-f * t) + g

    def dtri_exp(t, x):
        a, b, c, d, e, f, g = x
        return -a*b * np.exp(-b * t) - c*d * np.exp(-d * t) - e*f * np.exp(-f * t)

    def d2tri_exp(t, x):
        a, b, c, d, e, f, g = x
        return a*b*b * np.exp(-b * t) + c*d*d * np.exp(-d * t) + e*f*f * np.exp(-f * t)

    x02bi = [0.7, 1./50., 0.2, 1./100., 0.01]
    def bi_exp(t, x):
        a, b, c, d, g = x
        return a * np.exp(-b * t) + c * np.exp(-d * t) + g

    def dbi_exp(t, x):
        a, b, c, d, g = x
        return -a*b * np.exp(-b * t) - c*d * np.exp(-d * t)

    def d2bi_exp(t, x):
        a, b, c, d, g = x
        return a*b*b * np.exp(-b * t) + c*d*d * np.exp(-d * t)

    def is_within(r, x):
        return ((np.min(r) < x) and (np.max(r) > x))

    a_batches1 = []
    dadt_batches1 = []
    d2adt2_batches1 = []
    for j, (i, r, v) in enumerate(zip(i_batches1, r_batches1, v_batches1)):
        std_cutoff = 0.01
        pt = time1
        pv = voltage1
        cc = change_pt1
        dd = cap_mask1
        t_split = pt[~cc]
        t_split = np.append(t_split, pt[-1] + 1)
        t_i = 0

        a = (i / (g * r * (v - e))).cpu().numpy()

        tt = pt[dd]
        aa = a.reshape(-1)[dd]

        ao = np.zeros(pt.shape)
        dadto = np.zeros(pt.shape)
        d2adt2o = np.zeros(pt.shape)

        for t_f in t_split:
            idx = np.where((tt >= t_i) & (tt < t_f))[0]
            std = np.std(aa[idx])
            tfit = tt[idx]
            idx_full = np.where((pt >= tfit[0]) & (pt <= tfit[-1]))[0]
            if std > std_cutoff:
                #afit = smooth(aa[idx], 201)[100:-100]  # smoothing with 51/10ms
                afit = aa[idx]
                t = tfit - tfit[0]
                def f(x):
                    return np.sqrt(np.mean((tri_exp(t, x) - afit)**2))
                xopt = optimize.fmin(f, x0, disp=False)

                ao[idx_full] = tri_exp(t, xopt)
                dadto[idx_full] = dtri_exp(t, xopt)
                d2adt2o[idx_full] = d2tri_exp(t, xopt)
                print(v[idx_full][0].item(), std, 'exp fit')
            else:
                afit = smooth(aa[idx], 51)[25:-25]  # smoothing with 51/10ms
                spl = UnivariateSpline(tfit, afit, k=4)  # want smooth 2nd derivate, so k>3
                spl.set_smoothing_factor(0.2)

                ao[idx_full] = spl(tfit)
                dadto[idx_full] = spl(tfit, 1)
                d2adt2o[idx_full] = spl(tfit, 2)
                print(v[idx_full][0].item(), std, 'spline fit')


            t_i = t_f

        a_batches1.append(torch.from_numpy(ao).to(device))
        dadt_batches1.append(torch.from_numpy(dadto).to(device))
        d2adt2_batches1.append(torch.from_numpy(d2adt2o).to(device))
    if True:
        plt.plot(time1[dd], a[dd])
        plt.plot(time1, a_batches1[-1].reshape(-1).cpu().numpy())
        plt.plot(time1, dadt_batches1[-1].reshape(-1).cpu().numpy())
        plt.plot(time1, d2adt2_batches1[-1].reshape(-1).cpu().numpy())
        #plt.show()
        plt.savefig('png-r1-6-2re3/tmp1', dpi=200)
        plt.close()
    a_batches2 = []
    dadt_batches2 = []
    d2adt2_batches2 = []
    for j, (i, r, v) in enumerate(zip(i_batches2, r_batches2, v_batches2)):
        std_cutoff = 0.015
        pt = time2
        pv = voltage2
        cc = change_pt2
        dd = cap_mask2
        t_split = pt[~cc]
        t_split = np.append(t_split, pt[-1] + 1)
        t_i = 0

        a = (i / (g * r * (v - e))).cpu().numpy()

        tt = pt[dd]
        aa = a.reshape(-1)[dd]

        ao = np.zeros(pt.shape)
        dadto = np.zeros(pt.shape)
        d2adt2o = np.zeros(pt.shape)

        for t_f in t_split:
            idx = np.where((tt >= t_i) & (tt < t_f))[0]
            std = np.std(aa[idx])
            tfit = tt[idx]
            idx_full = np.where((pt >= tfit[0]) & (pt <= tfit[-1]))[0]
            if std > std_cutoff and (not is_within(tfit, 3500)):
                #afit = smooth(aa[idx], 201)[100:-100]  # smoothing with 51/10ms
                afit = aa[idx]
                t = tfit - tfit[0]
                def f(x):
                    return np.sqrt(np.mean((tri_exp(t, x) - afit)**2))
                if v[idx_full][0].item() == -90:
                    xopt, _ = pints.fmin(f, x02, method=pints.CMAES)
                    print('PINTS...')
                else:
                    xopt = optimize.fmin(f, x02, disp=False)

                ao[idx_full] = tri_exp(t, xopt)
                dadto[idx_full] = dtri_exp(t, xopt)
                d2adt2o[idx_full] = d2tri_exp(t, xopt)
                print(v[idx_full][0].item(), std, 'exp fit')
            elif is_within(tfit, 3500):
                print('Sinewave')
                afit = smooth(aa[idx], 21)[10:-10]  # smoothing with 51/10ms
                spl = UnivariateSpline(tfit, afit, k=5)  # want smooth 2nd derivate, so k>3
                spl.set_smoothing_factor(0.2)

                ao[idx_full] = spl(tfit)
                dadto[idx_full] = spl(tfit, 1)
                d2adt2o[idx_full] = spl(tfit, 2)
                print(v[idx_full][0].item(), std, 'spline fit')
            else:
                afit = smooth(aa[idx], 51)[25:-25]  # smoothing with 51/10ms
                spl = UnivariateSpline(tfit, afit, k=4)  # want smooth 2nd derivate, so k>3
                spl.set_smoothing_factor(0.2)

                ao[idx_full] = spl(tfit)
                dadto[idx_full] = spl(tfit, 1)
                d2adt2o[idx_full] = spl(tfit, 2)
                print(v[idx_full][0].item(), std, 'spline fit')

            t_i = t_f

        a_batches2.append(torch.from_numpy(ao).to(device))
        dadt_batches2.append(torch.from_numpy(dadto).to(device))
        d2adt2_batches2.append(torch.from_numpy(d2adt2o).to(device))
    if True:
        plt.plot(time2[dd], a[dd])
        plt.plot(time2, a_batches2[-1].reshape(-1).cpu().numpy())
        plt.plot(time2, dadt_batches2[-1].reshape(-1).cpu().numpy())
        plt.plot(time2, d2adt2_batches2[-1].reshape(-1).cpu().numpy())
        plt.show()
        plt.savefig('png-r1-6-2/tmp2', dpi=200)
        plt.close()
        #sys.exit()
    a_batches3 = []
    dadt_batches3 = []
    d2adt2_batches3 = []
    for j, (i, r, v) in enumerate(zip(i_batches3, r_batches3, v_batches3)):
        std_cutoff = 0.015
        pt = time3
        pv = voltage3
        cc = change_pt3
        dd = cap_mask3
        t_split = pt[~cc]
        t_split = np.append(t_split, pt[-1] + 1)
        t_i = 0

        a = (i / (g * r * (v - e))).cpu().numpy()

        tt = pt[dd]
        aa = a.reshape(-1)[dd]

        ao = np.zeros(pt.shape)
        dadto = np.zeros(pt.shape)
        d2adt2o = np.zeros(pt.shape)

        for t_f in t_split:
            idx = np.where((tt >= t_i) & (tt < t_f))[0]
            std = np.std(aa[idx])
            tfit = tt[idx]
            idx_full = np.where((pt >= tfit[0]) & (pt <= tfit[-1]))[0]
            print(std, v[idx_full][0].item())
            if std > std_cutoff:
                #afit = smooth(aa[idx], 201)[100:-100]  # smoothing with 51/10ms
                afit = aa[idx]
                t = tfit - tfit[0]
                def f(x, func):
                    return np.sqrt(np.mean((func(t, x) - afit)**2))
                if any([is_within(tfit, tflat) for tflat in [2000, 12000, 22000, 33000, 43000, 53000, 64000, 74000, 84000]]):
                    xopt = optimize.fmin(f, x02bi, args=(bi_exp,), disp=False)
                    ao[idx_full] = bi_exp(t, xopt)
                    dadto[idx_full] = dbi_exp(t, xopt)
                    d2adt2o[idx_full] = d2bi_exp(t, xopt)
                    print(v[idx_full][0].item(), std, 'bi-exp fit')
                else:
                    if v[idx_full][0].item() == -90:
                        xopt, _ = pints.fmin(f, x02, args=(tri_exp,), method=pints.CMAES, max_iter=1000)
                        print('PINTS...')
                    else:
                        xopt = optimize.fmin(f, x02, args=(tri_exp,), disp=False)

                    ao[idx_full] = tri_exp(t, xopt)
                    dadto[idx_full] = dtri_exp(t, xopt)
                    d2adt2o[idx_full] = d2tri_exp(t, xopt)
                    print(v[idx_full][0].item(), std, 'tri-exp fit')
            else:
                afit = smooth(aa[idx], 51)[25:-25]  # smoothing with 51/10ms
                spl = UnivariateSpline(tfit, afit, k=4)  # want smooth 2nd derivate, so k>3
                spl.set_smoothing_factor(0.2)

                ao[idx_full] = spl(tfit)
                dadto[idx_full] = spl(tfit, 1)
                d2adt2o[idx_full] = spl(tfit, 2)
                print(v[idx_full][0].item(), std, 'spline fit')

            t_i = t_f

        a_batches3.append(torch.from_numpy(ao).to(device))
        dadt_batches3.append(torch.from_numpy(dadto).to(device))
        d2adt2_batches3.append(torch.from_numpy(d2adt2o).to(device))
    if True:
        plt.plot(time3[dd], a[dd])
        plt.plot(time3, a_batches3[-1].reshape(-1).cpu().numpy())
        plt.plot(time3, dadt_batches3[-1].reshape(-1).cpu().numpy())
        plt.plot(time3, d2adt2_batches3[-1].reshape(-1).cpu().numpy())
        plt.savefig('png-r1-6-2re3/tmp3', dpi=200)
        #plt.show()
        plt.close()
        #sys.exit()
    # To tensors
    for i, (v, a, dadt, d2adt2) in enumerate(zip(v_batches1, a_batches1, dadt_batches1, d2adt2_batches1)):
        v_batches1[i] = v[mask1,...][skip::sparse]
        a_batches1[i] = a[mask1,...][skip::sparse]
        dadt_batches1[i] = dadt[mask1,...][skip::sparse]
        d2adt2_batches1[i] = d2adt2[mask1,...][skip::sparse]
    for i, (v, a, dadt, d2adt2) in enumerate(zip(v_batches2, a_batches2, dadt_batches2, d2adt2_batches2)):
        v_batches2[i] = v[mask2,...][skip::sparse]
        a_batches2[i] = a[mask2,...][skip::sparse]
        dadt_batches2[i] = dadt[mask2,...][skip::sparse]
        d2adt2_batches2[i] = d2adt2[mask2,...][skip::sparse]
    for i, (v, a, dadt, d2adt2) in enumerate(zip(v_batches3, a_batches3, dadt_batches3, d2adt2_batches3)):
        v_batches3[i] = v[mask3,...][skip::sparse]
        a_batches3[i] = a[mask3,...][skip::sparse]
        dadt_batches3[i] = dadt[mask3,...][skip::sparse]
        d2adt2_batches3[i] = d2adt2[mask3,...][skip::sparse]
    if False:
        plt.plot(time1[mask1,...][skip::sparse], v_batches1[-1].cpu().numpy())
        plt.plot(time1[mask1,...][skip::sparse], a_batches1[-1].cpu().numpy())
        plt.plot(time1[mask1,...][skip::sparse], dadt_batches1[-1].cpu().numpy())
        plt.plot(time1[mask1,...][skip::sparse], d2adt2_batches1[-1].cpu().numpy())
        plt.show()
        #sys.exit()
    v_batches = torch.cat(v_batches1 + v_batches3).to(device)
    a_batches = torch.cat(a_batches1 + a_batches3).to(device)
    dadt_batches = torch.cat(dadt_batches1 + dadt_batches3).to(device)
    d2adt2_batches = torch.cat(d2adt2_batches1 + d2adt2_batches3).to(device)

    t_batches = torch.cat([time1_torch[mask1,...][skip::sparse],
                           time1[-1] + time3_torch[mask3,...][skip::sparse]
                          ]).to(device)

    # Cache it
    torch.save(t_batches, 'r1/t.pt')
    torch.save(v_batches, 'r1/v.pt')
    torch.save(a_batches, 'r1/a.pt')
    torch.save(dadt_batches, 'r1/dadt.pt')
    torch.save(d2adt2_batches, 'r1/d2adt2.pt')

if args.debug and True:
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    #plt.plot(t.cpu().numpy(), r_batches[0].reshape(-1).cpu().numpy())
    #plt.plot(t.cpu().numpy(), i_batches[0].reshape(-1).cpu().numpy())
    #plt.plot(t.cpu().numpy(), drdt_batches[0].reshape(-1).cpu().numpy())
    #plt.plot(t.cpu().numpy(), didt_batches[0].reshape(-1).cpu().numpy())
    #plt.show()
    #plt.close()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(v_batches.reshape(-1).detach().cpu().numpy(), a_batches.reshape(-1).detach().cpu().numpy(),
            dadt_batches.reshape(-1).detach().cpu().numpy())
    ax.set_xlabel('V')
    ax.set_ylabel('a')
    ax.set_zlabel('da/dt')
    plt.show()
    sys.exit()
###
###
###



if __name__ == '__main__':

    ii = 0

    func = ODEFunc().to(device)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    
    best_loss = [1e10, 1e10]  # training loss, prediction loss

    #"""
    ###
    ### Pretrain
    ###
    x1 = torch.reshape(torch.linspace(-140, 80, 100).to(device), (-1, 1))
    xx1 = x1 / func.vrange
    x2 = torch.reshape(torch.linspace(-0.5, 1.5, 100).to(device), (-1, 1))
    X1, X2 = torch.meshgrid(x1.reshape(-1), x2.reshape(-1))
    XX1, X2 = torch.meshgrid(xx1.reshape(-1), x2.reshape(-1))
    p1 = torch.tensor([1.13e-4]).to(device)
    p2 = torch.tensor([7.45e-2]).to(device)
    p3 = torch.tensor([3.60e-5]).to(device)
    p4 = torch.tensor([4.49e-2]).to(device)
    k1 = p1 * torch.exp(p2 * X1)
    k2 = p3 * torch.exp(-p4 * X1)
    Y = k1 * (torch.tensor([1]).to(device) - X2) - k2 * X2

    XX = torch.stack([XX1.reshape(-1), X2.reshape(-1)]).T
    YY = Y.reshape(-1)

    opt = optim.Adam(func.net.parameters(), lr=0.001)
    for _ in range(1000):
        p = func.net(XX).to(device) / func.netscale
        loss = loss_fn(p.reshape(-1), YY)
        opt.zero_grad()
        loss.backward()
        opt.step()

    if args.debug:
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X1.cpu().numpy(), X2.cpu().numpy(), Y.cpu().numpy())
        pred = func.net(XX) / func.netscale
        ax.plot_surface(X1.cpu().numpy(), X2.cpu().numpy(), pred.reshape(50, 50).detach().cpu().numpy())
        plt.show()
        sys.exit()
    ###"""

    ###
    ### To predict
    ###
    x1 = torch.reshape(torch.linspace(-130, 70, 50).to(device), (-1, 1))
    xx1 = x1 / func.vrange
    x2 = torch.reshape(torch.linspace(0, 1, 50).to(device), (-1, 1))
    X1, X2 = torch.meshgrid(x1.reshape(-1), x2.reshape(-1))
    XX1, X2 = torch.meshgrid(xx1.reshape(-1), x2.reshape(-1))
    XX = torch.stack([XX1.reshape(-1), X2.reshape(-1)]).T

    # pretrained
    with torch.no_grad():
        pretrain_pred = func.net(XX) / func.netscale
    ###

    ###
    ### Training
    ###
    #'''
    x_av = torch.stack([v_batches.reshape(-1) / func.vrange, a_batches.reshape(-1)]).T
    y_dadt = dadt_batches.reshape(-1)

    '''
    # Remove a < 0 (probably due to the slight error in the reverseal potential)
    #to_keep = x_av[:, 1] > 0.01
    to_keep = ((x_av[:, 1] > 0) & (x_av[:, 1] < 1)).cpu().numpy()
    for i in range(10):  # remove +/- 10 points around those over or under shoot
        to_keep &= np.roll(to_keep, i + 1)
        to_keep &= np.roll(to_keep, -i - 1)
    x_av = x_av[to_keep, :]
    y_dadt = y_dadt[to_keep]
    '''
    # Remove a < 0 (probably due to the slight error in the reverseal potential)
    #to_keep = x_av[:, 1] > 0.01
    to_keep = x_av[:, 1] > 0
    x_av = x_av[to_keep, :]
    y_dadt = y_dadt[to_keep]

    if False:
        x_av[:, 1] = x_av[:, 1] + (0.002)*torch.randn(x_av[:, 1].shape)
        y_dadt = y_dadt + (0.002 * 0.005)*torch.randn(y_dadt.shape)

        import matplotlib.pyplot as plt
        plt.subplot(311)
        plt.plot(a_batches.reshape(-1).cpu().numpy())
        plt.subplot(312)
        plt.plot(dadt_batches.reshape(-1).cpu().numpy())
        plt.subplot(313)
        plt.plot(dadt_batches.reshape(-1).cpu().numpy())
        plt.ylim(-0.005, 0.01)
        plt.savefig('r1/tmp-data')
        plt.close()

    opt = optim.Adam(func.net.parameters(), lr=0.001)
    # gamma = decaying factor
    scheduler = StepLR(opt, step_size=400, gamma=0.9)  # 0.9**(4000steps/100) ~ 0.016
    for itr in range(16000):
        p = func.net(x_av.float()).to(device) / func.netscale
        loss = loss_fn(p.reshape(-1), y_dadt.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        # Decay Learning Rate
        scheduler.step()
        if (itr % 400) == 0:
            print('Iter', itr, 'LR', opt.param_groups[0]['lr'], 'Loss', loss.item())
            with torch.no_grad():
                pred_loss = []
                true_y0 = true_y0s[1]
                func.set_fixed_form_voltage_protocol(timep1, voltagep1)
                pred_y = odeint(func, true_y0, timep1_torch).to(device)
                pred_yo = g * pred_y[:, 0, 0] * pred_y[:, 0, 1] * (func._v(timep1_torch).to(device) - e)
                loss = torch.mean(torch.abs(pred_yo - torch.from_numpy(currentp1).to(device)))
                print('Validation APs | Total Loss {:.6f}'.format(loss.item()))
                pred_loss.append( loss.item() )

                func.set_fixed_form_voltage_protocol(time3, voltage3)
                pred_y = odeint(func, true_y0, time3_torch).to(device)
                pred_yo = g * pred_y[:, 0, 0] * pred_y[:, 0, 1] * (func._v(time3_torch).to(device) - e)
                loss = torch.mean(torch.abs(pred_yo - torch.from_numpy(current3).to(device)))
                print('Pr5 prediction | Total Loss {:.6f}'.format(loss.item()))
                pred_loss.append( loss.item() )

            checkpoint = {
                'epoch': itr + 1,
                'state_dict': func.state_dict(),
                'optimizer': opt.state_dict(),
                'loss': list(pred_loss)
            }
            if np.sum(pred_loss) < np.sum(best_loss):
                isbest = True
                best_loss = list(pred_loss)
                print('===== Current best model =====')
            else:
                isbest = False
            save_ckp(checkpoint, isbest)
    #'''

    # Save model
    torch.save(func.state_dict(), 'r1/model-state-dict.pt')
    torch.save(func, 'r1/model-entire.pt')
    # To load model:
    # func = TheModelClass(*args, **kwargs)
    # func.set_fixed_form_voltage_protocol(protocol[:, 0], protocol[:, 1])
    # func.load_state_dict(torch.load('r1/model-state-dict.pt'))
    # func.eval()
    #
    # Or:
    # func = torch.load('r1/model-entire.pt')
    # func.eval()

    with torch.no_grad():
        true_y0 = true_y0s[1]
        func.set_fixed_form_voltage_protocol(timep1, voltagep1)
        pred_y = odeint(func, true_y0, timep1_torch).to(device)
        pred_yo = g * pred_y[:, 0, 0] * pred_y[:, 0, 1] * (func._v(timep1_torch).to(device) - e)
        loss = torch.mean(torch.abs(pred_yo - torch.from_numpy(currentp1).to(device)))
        print('Training | Total Loss {:.6f}'.format(loss.item()))
        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        ax1.set_xlabel('t')
        ax1.set_ylabel('i')
        ax1.plot(timep1, currentp1, 'g-')
        ax1.plot(timep1, pred_yo.reshape(-1).cpu().numpy(), 'b--')
        ax1.set_xlim(timep1.min(), timep1.max())
        fig1.tight_layout()
        fig1.savefig('r1/{:03d}'.format(ii), dpi=200)
        plt.close(fig1)

        ax = plt.axes(projection='3d')
        ax.scatter(v_batches.reshape(-1).detach().cpu().numpy(), a_batches.reshape(-1).detach().cpu().numpy(),
                dadt_batches.reshape(-1).detach().cpu().numpy())
        pred = func.net(XX) / func.netscale
        ax.plot_surface(X1.cpu().numpy(), X2.cpu().numpy(), pred.reshape(50, 50).detach().cpu().numpy(), color='C1')
        ax.set_xlabel('V')
        ax.set_ylabel('a')
        ax.set_zlabel('da/dt')
        plt.savefig('r1/rates3d-{:03d}'.format(ii), dpi=200)
        #plt.show()
        plt.close()

        ii += 1
    ###
    ###
