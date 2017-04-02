import numpy as np
import ruamel.yaml as yaml
from .common import extwrapper


class QE():
    wl = None   # in nm
    EQE = None  # in % (0-100)
    Jsc = None  # in A/cm^-2
    output = None

    def __init__(self, loader=None, fname=None):

        self.wl, self.EQE, self.output = globals().get(loader)(fname)

    def plot_EQE(self, ax):

        line_EQE = ax.plot(self.wl, self.EQE, '.-', label='EQE')
        ax.set_xlabel('Wavelength [$nm$]')
        ax.set_ylabel('QE [%]')
        ax.legend(loc='best')
        ax.grid(True)
        return line_EQE


loader_file_ext = {
    'PVInstruments_QEX10': '.txt',
}


@extwrapper(loader_file_ext)
def PVInstruments_QEX10(file_path):
    '''Loads EQE data into attributes'''

    # the other columns are ignored
    data_array = np.genfromtxt(file_path, usecols=(0, 1),
                               skip_header=1, skip_footer=8)
    wl = data_array[:, 0]
    EQE = data_array[:, 1]

    f = open(file_path, 'r')

    with open(file_path, 'r') as f:
        d = yaml.safe_load('\n'.join(f.readlines()[-7:-1]))

    # put Jsc in mA
    d['Jsc'] = round(float(d['Jsc']) / 1e3, 7)
    output = d

    return wl, EQE, output
