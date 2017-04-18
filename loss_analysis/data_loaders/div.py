
import numpy as np
import ruamel.yaml as yaml
from .common import extwrapper
from .data_calculations import ideality_factor, _Vth, find_nearest


class IVDark():
    J = None   # in Amps
    V = None  # in Voltage
    temp = 300  # in Kelvin
    Rsh = None  # the shunt resistance
    area = None  # the area of the device
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

        self.m = ideality_factor(self.V, self.J, _Vth(self.temp))

        # TODO: Fix hack to get shunt resistance
        if self.Rsh is None:
            self.Rsh = 0.03 / find_nearest(0.03, self.V, self.J)

    def plot(self, ax):

        ax.plot(self.V, self.J, '.-', label='light IV')
        ax.set_xlabel('Voltage [$V$]')
        ax.set_ylabel('Current Density [$A cm^{-2}$]')
        ax.grid(True)

    def plot_log_JV(self, ax):
        '''
        Plots the current voltage curve

        inputs:
            ax: A figure axes to which is plotted
        '''
        ax.semilogy(self.V, self.J, '.-', label='Dark IV')
        ax.set_xlabel('Voltage [$V$]')
        ax.set_ylabel('Current Density [$A cm^{-2}$]')
        ax.grid(True)
        ax.legend(loc='best')

    def plot_mV(self, ax):
        ax.plot(self.V, self.m, '.-', label='dark IV')
        ax.set_xlabel('Voltage [$V$]')
        ax.set_ylabel('Ideality Factor []')
        ax.grid(True)
        ax.legend(loc='best')

loader_file_ext = {
    'darkstar_UNSW': '.drk',
}


@extwrapper(loader_file_ext)
def darkstar_UNSW(file_path):
    '''Loads Light IV data from UNSW darkstart into the dark IV object'''

    with open(file_path, 'r') as f:
        contence = '\n'.join(f.readlines()[1:10])

    # remove tabs and colon before spaces for yaml reader
    contence = contence.replace('\t', ' ')
    contence = contence.replace(' :', ':')
    details = yaml.safe_load(contence)

    # get raw data
    V, J = np.genfromtxt(file_path, skip_header=11, unpack=True)

    # convert I into J
    J = J / details['Cell Area in sqr cm']

    # rename interested vairables to "proper" names
    details['temp'] = details.pop('Temperature') + 273.15
    details['Rsh'] = details.pop('Rshunt in Reverse Bias')
    details['area'] = details.pop('Cell Area in sqr cm')

    return J, V, details
