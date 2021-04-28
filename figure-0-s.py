#!/usr/bin/env python3
import sys
import os
import argparse
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from smoothing import smooth

import torch
import torch.nn as nn

parser = argparse.ArgumentParser('Example of spline fitting.')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--cached', action='store_true')
args = parser.parse_args()

from torchdiffeq import odeint

device = 'cpu'

# Set random seed
np.random.seed(0)
torch.manual_seed(0)

noise_sigma = 0.1

t1 = torch.linspace(0., 8000, 80001).to(device)
mask1 = np.ones(len(t1), dtype=bool)
# Filter out +/- 5 time points for each step (numerical issue for estimating derivatives)
mask1[list(range(9995, 10050))
    + list(range(59995, 60050))
    + list(range(69995, 70050))
    + list(range(74995, 75050))] \
    = False

protocol_batches1 = []
pt1 = np.linspace(0., 8000., 80001)  # 8 seconds, 0.1 ms interval
template_v = np.zeros(pt1.shape)
template_v[:10000] = -80
# template_v[10000:60000] to be set
template_v[60000:70000] = -40
template_v[70000:75000] = -120
template_v[75000:] = -80
for v_i in [40]:
    v = np.copy(template_v)
    v[10000:60000] = v_i
    protocol_batches1.append(np.array([pt1, v]).T)

gt_true_y0s = [torch.tensor([[1., 0.]]).to(device),  # what you get after holding at +40mV
               torch.tensor([[0., 1.]]).to(device)]  # (roughly) what you get after holding at -80mV

#
#
#
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

makedirs('figure-0-s')


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


#
# Generate syn data from the ground truth model
#
true_model = Lambda()

true_y_batches1 = []
true_yo_batches1 = []

if args.cached:
    v_batches1 = torch.load('figure-0-s/v.pt')
    a_noisy_batches1 = torch.load('figure-0-s/a_n.pt')
    a_batches1 = torch.load('figure-0-s/a.pt')
    dadt_batches1 = torch.load('figure-0-s/dadt.pt')
    i_noisy_batches1 = torch.load('figure-0-s/i_n.pt')
    i_batches1 = torch.load('figure-0-s/i.pt')
    didt_batches1 = torch.load('figure-0-s/didt.pt')
else:
    with torch.no_grad():
        for protocol in protocol_batches1:
            true_y0 = gt_true_y0s[1]
            true_model.set_fixed_form_voltage_protocol(protocol[:, 0], protocol[:, 1])
            true_y = odeint(true_model, true_y0, t1, method='dopri5')
            true_y_batches1.append(true_y)
            true_yo_batches1.append(true_y[:, 0, 0] * true_y[:, 0, 1] * (true_model._v(t1) + 86) +
                    torch.from_numpy(np.random.normal(0, noise_sigma, t1.cpu().numpy().shape)).to(device))

    ###
    ### 'post-processing': estimating dadt and a
    ###
    dvdt_constant = torch.tensor([0]).to(device)  # for now yes
    e = torch.tensor([-86.]).to(device)  # assume we know
    g = torch.tensor([1.]).to(device)  # assume we know
    r_batches1 = []
    with torch.no_grad():
        m = Lambda().to(device)
        for protocol in protocol_batches1:
            true_y0 = gt_true_y0s[1]
            m.set_fixed_form_voltage_protocol(protocol[:, 0], protocol[:, 1])
            true_y = odeint(m, true_y0, t1, method='dopri5')
            r_batches1.append(true_y[:, 0, 1])
    v_batches1 = []
    for protocol in protocol_batches1:
        true_model.set_fixed_form_voltage_protocol(protocol[:, 0], protocol[:, 1])
        v_batches1.append(true_model._v(t1)[0])
    drdt_batches1 = []
    for r, v in zip(r_batches1, v_batches1):
        k3 = m.p5 * torch.exp(m.p6 * v)
        k4 = m.p7 * torch.exp(-m.p8 * v)
        drdt = -k3 * r + k4 * (1. - r)
        drdt_batches1.append(drdt)
    i_noisy_batches1 = true_yo_batches1
    i_batches1 = []
    didt_batches1 = []
    for j, (i, protocol) in enumerate(zip(i_noisy_batches1, protocol_batches1)):
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
            ifit = smooth(ii[idx], 61)[30:-30]  # smoothing with 151/10ms
            spl = UnivariateSpline(tfit, ifit, k=3)
            spl.set_smoothing_factor(0)#.1)  # s is the smoothing factor; if 0, spline will interpolate through all data points
            io = np.append(io, spl(tfit))
            didto = np.append(didto, spl.derivative()(tfit))
            t_i = t_f
        i_batches1.append(torch.from_numpy(io).to(device))
        didt_batches1.append(torch.from_numpy(didto).to(device))
    # Calculate a and dadt
    a_noisy_batches1 = []
    a_batches1 = []
    dadt_batches1 = []
    for j, (inoisy, i, r, v, drdt, didt) in enumerate(zip(i_noisy_batches1, i_batches1, r_batches1, v_batches1, drdt_batches1, didt_batches1)):
        ii = i.reshape(-1)
        iinoisy = inoisy.reshape(-1)
        a = ii / (g * r * (v - e))
        a_noisy = iinoisy / (g * r * (v - e))
        if np.all(v.cpu().numpy() == v.cpu().numpy()[0]) or True:  # all steps even different values
            dvdt = dvdt_constant
        else:
            spl = UnivariateSpline(range(len(v.cpu().numpy())), v.cpu().numpy(), k=3, s=0)
            dvdt = torch.from_numpy(spl.derivative()(range(len(v.cpu().numpy()))))
        dadt = r ** (-1) * (
                (didt / g - a * r * dvdt) / (v - e)
                - a * drdt
                )
        a_noisy_batches1.append(a_noisy)
        a_batches1.append(a)
        dadt_batches1.append(dadt)

    # Cache it
    torch.save(v_batches1, 'figure-0-s/v.pt')
    torch.save(a_noisy_batches1, 'figure-0-s/a_n.pt')
    torch.save(a_batches1, 'figure-0-s/a.pt')
    torch.save(dadt_batches1, 'figure-0-s/dadt.pt')
    torch.save(i_noisy_batches1, 'figure-0-s/i_n.pt')
    torch.save(i_batches1, 'figure-0-s/i.pt')
    torch.save(didt_batches1, 'figure-0-s/didt.pt')


#
#
#
zoom_in_win = {
    0: [(1000, 5000), (6005, 6600), (7005, 7095)],  # pr3
}
zoom_in_y = {
    0: [(-0.75, 3.25), (1., 22.), (-25.5, -10.0)],  # pr3
}
facecolors = [
    [sns.color_palette("Set2")[0],
     sns.color_palette("Set2")[1],
     sns.color_palette("Set2")[2]],  # pr3
]


#
# Plot
#
ds = 2
fig = plt.figure(figsize=(11, 7))
n_maxzoom = 3
grid = plt.GridSpec(4 + 1 + 7 + 1 + 7 + 4 + 7, 4 + 1 + 4 + 1 + 4,
                    hspace=0.0, wspace=0.0)
axes = np.empty([4], dtype=object)

axes[0] = fig.add_subplot(grid[:4, :])
axes[0].set_xticklabels([])
axes[1] = fig.add_subplot(grid[5:12, :])
axes[1].set_xticklabels([])
axes[2] = fig.add_subplot(grid[13:20, :])
axes[3] = np.empty(n_maxzoom, dtype=object)

for ii in range(n_maxzoom):
    axes[3][ii] = fig.add_subplot(
            grid[-7:, ii*4+ii*1:(ii+1)*4+ii*1])
    axes[3][ii].set_xticklabels([])
    axes[3][ii].set_xticks([])
    axes[3][ii].set_yticklabels([])
    axes[3][ii].set_yticks([])

# Set labels
axes[0].set_ylabel('Voltage\n(mV)', fontsize=12)
axes[1].set_ylabel('Current\n(nA)', fontsize=12)
axes[2].set_ylabel(r'$dI/dt$' + '\n(nA/ms)', fontsize=12)
axes[3][0].set_ylabel('Zoom in', fontsize=12)
axes[2].set_xlabel('Time (ms)', fontsize=12)

# Plot!
for i, (voltage1, current1, current1_smooth, didt) in enumerate(zip(v_batches1, i_noisy_batches1, i_batches1, didt_batches1)):
    axes[0].plot(t1, voltage1, c='#7f7f7f', ds='steps')
    axes[1].plot(t1[::ds], current1.reshape(-1)[::ds], c='#7f7f7f', label='__nolegend__' if i else 'Data')
    axes[1].plot(t1[::ds], current1_smooth.reshape(-1)[::ds], '--', c='C0', label='__nolegend__' if i else 'Spline')
    axes[2].plot(t1[::ds], didt.reshape(-1)[::ds], c='C3')
    
    axes[0].set_xlim([t1[::ds][0], t1[::ds][-1]])
    axes[1].set_xlim([t1[::ds][0], t1[::ds][-1]])
    axes[2].set_xlim([t1[::ds][0], t1[::ds][-1]])

    # Zooms
    for i_z, (t_i, t_f) in enumerate(zoom_in_win[0]):
        # Find closest time
        idx_i = np.argmin(np.abs(t1[::ds] - t_i))
        idx_f = np.argmin(np.abs(t1[::ds] - t_f))
        # Data
        t = t1[::ds][idx_i:idx_f]
        cn = current1.reshape(-1)[::ds][idx_i:idx_f]
        cs = current1_smooth.reshape(-1)[::ds][idx_i:idx_f]

        # Work out third panel plot
        axes[3][i_z].plot(t, cn, c='#7f7f7f')
        axes[3][i_z].plot(t, cs, '--', c='C0')

        y_min, y_max = zoom_in_y[0][i_z]
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
        plt.sca(axes[1])
        pyplot_axes = plt.gca()
        pyplot_axes.add_patch(pathpatch)

        axes[3][i_z].set_xlim([t[0], t[-1]])
        # Re-adjust the max and min
        axes[3][i_z].set_ylim([y_min, y_max])
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
        plt.sca(axes[3][i_z])
        pyplot_axes = plt.gca()
        pyplot_axes.add_patch(pathpatch)

        # Set arrow and time duration
        axes[3][i_z].arrow(1, -0.05, -1, 0,
                              length_includes_head=True,
                              head_width=0.03, head_length=0.05,
                              clip_on=False, fc='k', ec='k',
                              transform=axes[3][i_z].transAxes)
        axes[3][i_z].arrow(0, -0.05, 1, 0,
                              length_includes_head=True,
                              head_width=0.03, head_length=0.05,
                              clip_on=False, fc='k', ec='k',
                              transform=axes[3][i_z].transAxes)
        axes[3][i_z].text(0.5, -0.15,
                '%s ms' % np.around(t_f - t_i, decimals=0),
                transform=axes[3][i_z].transAxes,
                horizontalalignment='center',
                verticalalignment='center')
        # Set arrow and current range
        axes[3][i_z].arrow(-0.05, 1, 0, -1,
                              length_includes_head=True,
                              head_width=0.03, head_length=0.05,
                              clip_on=False, fc='k', ec='k',
                              transform=axes[3][i_z].transAxes)
        axes[3][i_z].arrow(-0.05, 0, 0, 1,
                              length_includes_head=True,
                              head_width=0.03, head_length=0.05,
                              clip_on=False, fc='k', ec='k',
                              transform=axes[3][i_z].transAxes)
        axes[3][i_z].text(-0.1, 0.5,
                '%s nA' % np.around(y_max - y_min, decimals=1),
                rotation=90,
                transform=axes[3][i_z].transAxes,
                horizontalalignment='center',
                verticalalignment='center')

axes[1].legend(loc='lower left', bbox_to_anchor=(0., 1.7), ncol=4,
                  columnspacing=4, #handletextpad=1,
                  bbox_transform=axes[1].transAxes)

fig.align_ylabels([axes[0], axes[1], axes[2], axes[3][0]])
#grid.tight_layout(fig, pad=0.1, rect=(0, 0, -0.8, 0.95))

plt.savefig('figure-0-s/fig0-s.pdf', format='pdf', pad_inches=0.02,
            bbox_inches='tight')
fig.canvas.start_event_loop(sys.float_info.min)  # Silence Tkinter callback
plt.savefig('figure-0-s/fig0-s', pad_inches=0.02, dpi=300, bbox_inches='tight')
#plt.show()
plt.close()
