
import numpy as np
import ruamel.yaml as yaml
from .common import extwrapper
from .data_calculations import ideality_factor, _Vth


class IVLight():
    J = None   # in Amps
    V = None  # in Voltage
    Jsc = None  # in Amps
    Voc = None  # in Volts
    Jmp = None  # in Volts
    Vmp = None  # in Volts
    efficency = None  # in percent
    temp = None
    FF = None  # unitless, < 1
    Rs = 0

    other = None  # other values

    def __init__(self, loader=None, fname=None):

        self.J, self.V, self.other = globals().get(loader)(fname)

        keys = dict(self.other).keys()
        for key in keys:
            if hasattr(self, key):
                setattr(self, key, self.other[key])
                del self.other[key]

        self.process()

    def process(self):

        self.m = ideality_factor(
            self.V, -1 * (self.J - self.Jsc), _Vth(self.temp))

        if None not in (self.FF, self.Jmp, self.Vmp, self.Jsc, self.Voc):
            assert np.isclose(
                [self.FF], [self.Jmp * self.Vmp / self.Jsc / self.Voc], rtol=0.1), '{0} \t {1}'.format([self.FF], [self.Jmp * self.Vmp / self.Jsc / self.Voc])

    def plot_JV(self, ax):
        '''
        Plots the current voltage curve

        inputs:
            ax: A figure axes to which is plotted
        '''
        ax.plot(self.V, self.J, '.-', label='light IV')
        ax.set_xlabel('Voltage [$V$]')
        ax.set_ylabel('Current Density [$A cm^{-2}$]')
        ax.grid(True)
        ax.set_ylim(bottom=0)

    def plot_mV(self, ax):
        ax.plot(self.V, self.m, '.-', label='Light IV')
        ax.set_xlabel('Voltage [$V$]')
        ax.set_ylabel('Ideality Factor []')
        ax.grid(True)
        ax.legend(loc='best')
        ax.set_ylim(ymin=0)

loader_file_ext = {
    'darkstar_UNSW': '.lgv',
}


@extwrapper(loader_file_ext)
def darkstar_UNSW(file_path):
    '''Loads Light IV data from UNSW darkstart into the light IV object'''

    # d = OrderedDict()
    with open(file_path, 'r') as f:
        contence = '\n'.join(f.readlines()[1:19])

    # remove tabs and colon before spaces for yaml reader
    contence = contence.replace('\t', ' ')
    contence = contence.replace(' :', ':')
    details = yaml.safe_load(contence)

    # get raw data
    V, J = np.genfromtxt(file_path, skip_header=20, unpack=True)

    # convert I into J
    J = J / details['Cell Area (sqr cm)']

    details['temp'] = details.pop('Temperature (\'C)')
    details['temp'] += 273.15

    details['efficiency'] = details.pop('Eff')

    return J, V, details
