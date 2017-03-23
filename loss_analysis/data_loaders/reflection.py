
import numpy as np
import ruamel.yaml as yaml
from .common import extwrapper


class Relf():
    wl = None   # in nm from lowest to highest
    reflection = None  # in % (0-100)

    def __init__(self, loader=None, fname=None):

        self.wl, self.reflection, self.other = globals().get(loader)(fname)

    def _check_wavelength_assending_oder(self):
        if self.wl[0] > self.wl[-1]:
            self.wl = self.wl[::-1]
            self.ref = self.ref[::-1]


loader_file_ext = {
    'PerkinElma_lambda_1050': '.csv',
}


@extwrapper(loader_file_ext)
def PerkinElma_lambda_1050(file_path):
    '''Loads Reflectance data in attributes'''

    wl, ref = np.genfromtxt(file_path, usecols=(0, 1), skip_header=1,
                            delimiter=',', unpack=True)

    other = {}
    return wl, ref, other
