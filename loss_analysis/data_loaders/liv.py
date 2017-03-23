
import numpy as np
import ruamel.yaml as yaml
from .common import extwrapper


class IVLight():
    J = None   # in Amps
    V = None  # in Voltage
    other = None  # other values

    def __init__(self, loader=None, fname=None):

        self.J, self.V, self.other = globals().get(loader)(fname)

    def plot(self, ax):
        '''
        Plots the current voltage curve

        inputs:
            ax: A figure axes to which is plotted
        '''
        ax.plot(self.V, self.J, '-o', label='light IV')
        ax.set_xlabel('Voltage [$V$]')
        ax.set_ylabel('Current Density [$A cm^{-2}$]')
        ax.grid(True)

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

    return J, V, details
