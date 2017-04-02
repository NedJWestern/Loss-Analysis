
import numpy as np
import ruamel.yaml as yaml
from .common import extwrapper


def refl_metal_from_refl_total(wl, refl_total, wl_bounds=(400, 1000)):
    '''
    Determines the relfection from metal on the surface as the minimum
    reflectance on the reflectance curve. Implicitly this assume that the
    anti reflection curve and texturing results in the reflection of the sample
    to be &asymp; 0\% for a signle wavelength.
    '''
    index = wl >= wl_bounds[0]
    index *= wl <= wl_bounds[1]
    refl_min = np.amin(refl_total[index])
    return refl_min


def refl_front_from_refl_total(wl, refl_total, wl_bounds=(850, 950)):
    '''
    Calculates the amount of light that is reflected from the front surface
    from a measured total reflectance. This is done by assuming the reflection
    linearly with wavelength towards longer  wavelengths. A stright line is fitted
    in the wavelength range provided by the wl_bounds, and extended to longer wavelengths
    '''

    # get the index where these wavelengths occur
    index_l = (wl >= wl_bounds[0])
    index = (wl <= wl_bounds[1]) * index_l

    # fit in the desirbed wavelength range
    popt, pcov = np.polyfit(wl[index], refl_total[index], 1, cov=True)

    # before this wavelength range, the front reflection is the measured.
    # after it is provided by the fitted data
    refl_front = np.copy(refl_total)
    refl_front[index_l] = np.polyval(popt, wl[index_l])

    # return the calculated data.
    return refl_front


class Refl():
    wl = None   # in nm from lowest to highest
    reflection = None  # in % (0-100)

    _refl_front = None
    _refl_metal = None

    def __init__(self, loader=None, fname=None):

        self.wl, self.reflection, self.other = globals().get(loader)(fname)

        for key in self.other:
            if hasattr(self, key):
                setattr(self, key, self.other[key])

    @property
    def refl_front(self):
        if self._refl_front is None:
            ret = refl_front_from_refl_total(
                self.wl, self.reflection)
        else:
            ret = self._refl_front

        return ret

    @refl_front.setter
    def refl_front(self, value):
        self._refl_front = value

    @property
    def refl_metal(self):
        if self._refl_metal is None:
            ret = refl_metal_from_refl_total(
                self.wl, self.reflection)
        else:
            ret = self._refl_metal
        return ret

    @refl_metal.setter
    def refl_metal(self, value):
        self._refl_metal = value

    def _check_wavelength_assending_oder(self):
        if self.wl[0] > self.wl[-1]:
            self.wl = self.wl[::-1]
            self.ref = self.ref[::-1]

    def plot(self, ax):
        ax.plot(self.wl, self.reflection, '.-', label='Reflection')
        ax.plot(self.wl, self.refl_front, '.-', label='Front reflection')
        ax.plot(self.wl, np.ones(len(self.wl)) *
                self.refl_metal, 'r-', label='Metal reflection')
        ax.set_ylabel('Reflectance [%]')
        ax.legend(loc='best')
        ax.grid(True)

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
