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

import pints

import torch
import torch.nn as nn

parser = argparse.ArgumentParser('IKr real data fit with the candidate model with Pr3 and Pr4.')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--myokit', action='store_true', help='Use Myokit for speed up.')
args = parser.parse_args()

from torchdiffeq import odeint

# Set random seed
np.random.seed(0)
torch.manual_seed(0)

device = 'cpu'

p0 = np.array([
    # https://github.com/CardiacModelling/hERGRapidCharacterisation/blob/master/room-temperature-only/out/herg25oc1/herg25oc1-staircaseramp-B06-solution-542811797.txt
    1.12592345582957387e-01 * 1e-3,
    8.26751134920666146e+01 * 1e-3,
    3.38768033864048357e-02 * 1e-3,
    4.67106147665183542e+01 * 1e-3,
])


prediction_protocol = np.loadtxt('test-protocols/ap2hz.csv', skiprows=1, delimiter=',')
prediction_protocol[:, 0] *= 1e3  # s -> ms
prediction_t = torch.linspace(0., 3000, 1501).to(device)


raw_data1 = np.loadtxt('data/pr3-steady-activation-cell-5.csv', delimiter=',', skiprows=1)
raw_data2 = np.loadtxt('data/pr4-inactivation-cell-5.csv', delimiter=',', skiprows=1)
time1 = raw_data1[:, 0]
time1_torch = torch.from_numpy(raw_data1[:, 0]).to(device)
current1 = raw_data1[:, 1]
voltage1 = raw_data1[:, 2]
time2 = raw_data2[:, 0]
time2_torch = torch.from_numpy(raw_data2[:, 0]).to(device)
current2 = raw_data2[:, 1]
voltage2 = raw_data2[:, 2]

#
#
#
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

makedirs('r0-bad')


#
#
#
if args.myokit:
    try:
        import myokit
    except ModuleNotFoundError:
        # Install myokit
        import subprocess
        import sys
        def install(package):
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        install('myokit')
        import myokit

    #
    # Time out handler
    #
    class Timeout(myokit.ProgressReporter):
        """
        A :class:`myokit.ProgressReporter` that halts the simulation after
        ``max_time`` seconds.
        """
        def __init__(self, max_time):
            self.max_time = float(max_time)
        def enter(self, msg=None):
            self.b = myokit.Benchmarker()
        def exit(self):
            pass
        def update(self, progress):
            return self.b.time() < self.max_time


    class Model(pints.ForwardModel):
        """
        # A voltage clamp model linking Myokit and Pints ForwardModel.
        """

        def __init__(self, max_evaluation_time=60):
            """
            # max_evaluation_time: maximum time (in second) allowed for one
            #                      simulate() call.
            """
            self._model = myokit.parse_model('''
            [[model]]
            # Initial values
            ikr.act = 0
            ikr.rec = 1

            # Simulation engine variables
            [engine]
            time = 0 [ms]
                bind time

            # Membrane potential
            [membrane]
            V = 0 [mV]
                bind pace
                label membrane_potential

            [nernst]
            EK = -86 [mV]

            # Hodgkin-Huxley current model
            [ikr]
            use membrane.V
            IKr = g * act * rec * (V - nernst.EK)
                in [nA]
            dot(act) = (inf - act) / tau
                inf = k1 * tau
                tau = 1 / (k1 + k2)
                    in [ms]
                k1 = p1 * exp(p2 * V)
                    in [1/ms]
                k2 = p3 * exp(-p4 * V)
                    in [1/ms]
            dot(rec) = (inf - rec) / tau
                inf = k4 * tau
                tau = 1 / (k3 + k4)
                    in [ms]
                k3 = p5 * exp(p6 * V)
                    in [1/ms]
                k4 = p7 * exp(-p8 * V)
                    in [1/ms]
            # https://github.com/CardiacModelling/hERGRapidCharacterisation/blob/master/room-temperature-only/out/herg25oc1/herg25oc1-staircaseramp-B06-solution-542811797.txt
            p1 = 1.12592345582957387e-01 * 1e-3 [1/ms]
            p2 = 8.26751134920666146e+01 * 1e-3 [1/mV]
            p3 = 3.38768033864048357e-02 * 1e-3 [1/ms]
            p4 = 4.67106147665183542e+01 * 1e-3 [1/mV]

            # https://physoc.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1113%2FJP275733&file=tjp12905-sup-0001-textS1.pdf#page=20
            # Table F11, Cell #5 parameters
            p5 = 8.73e-2 [1/ms]
            p6 = 8.90e-3 [1/mV]
            p7 = 5.20e-3 [1/ms]
            p8 = 3.16e-2 [1/mV]
            p9 = 0.1524 [uS]
            g = p9
            ''')
            self._vhold = -80

            # maximum time allowed
            self.max_evaluation_time = max_evaluation_time

            # Create simulation protocol
            self.simulation = myokit.Simulation(self._model)
            self.simulation.set_tolerance(1e-6, 1e-8)
            # self.simulation.set_max_step_size(1e-2)  # ms

            # Init states
            self.init_state = self.simulation.state()

        def n_parameters(self):
            return 4

        def set_init_state(self, v):
            self.init_state = v

        def set_voltage_protocol(self, p, prt_mask=None):
            # Assume protocol p is
            # [step_1_voltage, step_1_duration, step_2_voltage, ...]
            # prt_mask: (numpy) mask function that remove part of the measurement;
            #           can be used as a capacitive filter, or to make the fitting
            #           harder
            protocol = myokit.Protocol()
            duration = 0
            for i in range(len(p) // 2):
                protocol.add_step(p[2 * i], p[2 * i + 1])
                duration += p[2 * i + 1]
            self.simulation.set_protocol(protocol)
            del(protocol)
            self.prt_mask = prt_mask

        def set_fixed_form_voltage_protocol(self, t, v, prt_mask=None):
            # v, t: voltage, time to be set in ms, mV
            # prt_mask: (numpy) mask function that remove part of the measurement;
            #           can be used as a capacitive filter, or to make the fitting
            #           harder
            self.simulation.set_fixed_form_protocol(
                t, v  # ms, mV
            )
            self.prt_mask = prt_mask

        def simulate(self, parameters, times):
            # simulate() method for Pints

            p1, p2, p3, p4 = parameters
            self.simulation.set_constant('ikr.p1', p1)
            self.simulation.set_constant('ikr.p2', p2)
            self.simulation.set_constant('ikr.p3', p3)
            self.simulation.set_constant('ikr.p4', p4)

            # Reset to ensure each simulate has same init condition
            self.simulation.reset()
            self.simulation.set_state(self.init_state)

            # Run!
            try:
                p = Timeout(self.max_evaluation_time)
                d = self.simulation.run(np.max(times)+1e-3,
                    log_times=times,
                    log=['ikr.IKr'],
                    progress=p,
                    ).npview()
                del(p)
            except (myokit.SimulationError, myokit.SimulationCancelledError):
                return np.full(times.shape, float('inf'))

            # Apply capacitance filter and return
            if self.prt_mask is not None:
                fcap = np.zeros(times.shape)
                fcap[self.prt_mask] = 1
                d['ikr.IKr'] = d['ikr.IKr'] * self.fcap

            return d['ikr.IKr']

else:  # use torchdiffeq odeint

    #
    # Timer
    #
    import signal
    from contextlib import contextmanager

    class TimeoutException(Exception): pass

    @contextmanager
    def time_limit(seconds):
        def signal_handler(signum, frame):
            raise TimeoutException('Simulation time out.')
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)


    class ODEFunc(nn.Module):

        def __init__(self):
            super(ODEFunc, self).__init__()

            self.p1 = 1.13e-4
            self.p2 = 7.45e-2
            self.p3 = 3.60e-5
            self.p4 = 4.49e-2
            # https://physoc.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1113%2FJP275733&file=tjp12905-sup-0001-textS1.pdf#page=20
            # Table F11, Cell #5 parameters
            self.p5 = 8.73e-2
            self.p6 = 8.90e-3
            self.p7 = 5.20e-3
            self.p8 = 3.16e-2

            self.unity = torch.tensor([1]).to(device)

        def set_parameters(self, x):
            #self.p1, self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8 = x
            self.p1, self.p2, self.p3, self.p4 = x

        def set_fixed_form_voltage_protocol(self, t, v):
            # Regular time point voltage protocol time series
            self._t_regular = t
            self._v_regular = v
            self.__v = interp1d(t, v)

        def _v(self, t):
            #return torch.from_numpy(np.interp([t.cpu().detach().numpy()], self._t_regular,
            #                                  self._v_regular))
            #return self.__v([t.cpu().detach().numpy()])
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

            k1 = self.p1 * torch.exp(self.p2 * v)
            k2 = self.p3 * torch.exp(-self.p4 * v)
            k3 = self.p5 * torch.exp(self.p6 * v)
            k4 = self.p7 * torch.exp(-self.p8 * v)

            dadt = k1 * (self.unity - a) - k2 * a
            drdt = -k3 * r + k4 * (self.unity - r)

            return torch.stack([dadt[0], drdt[0]]).reshape(1, -1)


    class Model(pints.ForwardModel):
        def __init__(self, ode_func=None):
            super(Model, self).__init__()
            if ode_func is None:
                self._ode = ODEFunc().to(device)
            else:
                self._ode = ode_func
            self.set_y0()
            self.set_voltage_protocol_batches()
            self.set_discontinous()

        def n_parameters(self):
            #return 9
            return 4

        def n_outputs(self):
            if self._ps is not None:
                return len(self._ps)
            else:
                return 1

        def set_fixed_form_voltage_protocol(self, t, v):
            self._ode.set_fixed_form_voltage_protocol(t, v)

        def set_voltage_protocol_batches(self, ps=None):
            "ps: list of voltage time series [times, voltages]"
            self._ps = ps

        def set_y0(self, y0=np.asarray([0, 1])):
            self._y0 = y0.reshape(1, -1)
            self._y0_torch = torch.from_numpy(self._y0).float()

        def set_discontinous(self, discontn=None):
            if discontn is not None:
                self.discontn = torch.from_numpy(discontn)
            else:
                self.discontn = discontn

        def simulate(self, x, t):
            "Pints's forward simulation, x parameters, t time series"
            t = torch.from_numpy(np.copy(t))
            #g = x[0]
            g = 0.1524
            #self._ode.set_parameters(x[1:])
            self._ode.set_parameters(x[:])
            if self._ps is not None:
                out = []
                for p in self._ps:
                    self.set_fixed_form_voltage_protocol(p[:, 0], p[:, 1])
                    try:
                        with time_limit(600):
                            o = odeint(self._ode, self._y0_torch, t, method='dopri5')
                        out.append((g * o[:, 0, 0] * o[:, 0, 1] * (self._ode._v(t).to(device) + 86)).cpu().detach().numpy().reshape(-1))
                    except TimeoutException as e:
                        out.append(np.ones(t.shape) * np.inf)
                return np.asarray(out).T
            else:
                try:
                    with time_limit(600):
                        o = odeint(self._ode, self._y0_torch, t, method='dopri5', options={"grid_points": self.discontn, "eps": 1e-6})
                except TimeoutException as e:
                    return np.ones(t.shape) * np.inf
                return (g * o[:, 0, 0] * o[:, 0, 1] * (self._ode._v(t).to(device) + 86)).cpu().detach().numpy().reshape(-1)


#
#
#
if args.debug:
    if args.myokit:
        model = Model()
    else:
        func = ODEFunc().to(device)
        model = Model(func)
    model.set_fixed_form_voltage_protocol(time1, voltage1)
    o = model.simulate(p0, time1)
    print(o.shape)
    l = int(len(time1) / 7)
    for i in range(7):
        plt.plot(time1[:l], o[l*i:l*(i+1)])
    plt.show()


if __name__ == '__main__':

    ii = 0

    timet = np.append(time1, time1[-1] + time2[1] + time2)
    voltaget = np.append(voltage1, voltage2)
    datat = np.append(current1, current2)

    if args.myokit:
        model = Model()
    else:
        func = ODEFunc().to(device)
        model = Model(func)
        change_pt = np.append([True], ~(voltaget[1:] != voltaget[:-1]))
        discontinuous_time = timet[~np.roll(change_pt, -1)]
        model.set_discontinous(discontinuous_time)
    model.set_fixed_form_voltage_protocol(timet, voltaget)

    timet = timet[::10]
    datat = datat[::10]

    # PINTS
    problem = pints.SingleOutputProblem(model, timet, datat)
    error = pints.SumOfSquaresError(problem)
    print('Initial guess error:', error(p0))
    transform = pints.LogTransformation(n_parameters=problem.n_parameters())

    #"""
    import time
    for _ in range(1):
        start_time = time.time()
        print(error(p0))
        print("--- %s seconds ---" % (time.time() - start_time))
    #sys.exit()
    #"""

    if True:
        plt.plot(timet, datat)
        plt.plot(timet, problem.evaluate(p0))
        plt.savefig('r0-bad/data', dpi=300)
        plt.close()

    opt = pints.OptimisationController(
        error,
        p0,
        sigma0=p0 * 1e-1,
        boundaries=pints.RectangularBoundaries(p0 * 0.1, p0 * 10),
        method=pints.CMAES,
        transform=transform,
    )
    opt.set_max_iterations(None)
    opt.set_max_unchanged_iterations(iterations=100, threshold=1e-3)
    opt.set_parallel(True)

    found_parameters, found_value = opt.run()

    np.savetxt('r0-bad/model-parameters.txt', found_parameters)
