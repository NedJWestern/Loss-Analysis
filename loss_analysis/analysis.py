import numpy as np
from scipy import constants
from scipy.optimize import curve_fit
import os
from numpy.polynomial import polynomial as poly

path_const = os.path.join(os.pardir, 'constants')


def AM15G_resample(wl):
    '''Returns AM1.5G spectrum at given wavelengths'''
    AM15G_wl = np.genfromtxt(os.path.join(path_const, 'AM1.5G_spectrum.dat'),
                             usecols=(0,), skip_header=1)
    AM15G_Jph = np.genfromtxt(os.path.join(path_const, 'AM1.5G_spectrum.dat'),
                              usecols=(1,), skip_header=1)
    return np.interp(wl, AM15G_wl, AM15G_Jph)


def find_nearest(x_val, xdata, ydata=None):
    '''
    TODO: check xdata and ydata are equal lengths
    Finds the nearest index in 'xdata' to 'value'
    Returns corresponding 'ydata' value if given
    '''
    idx = (np.abs(xdata - x_val)).argmin()
    if ydata is None:
        return idx
    else:
        return ydata[idx]


def wl_to_alpha(given_wl):
    '''Returns alpha for a given wavelength in [nm] xxx? in Si'''
    alpha_data = np.genfromtxt(os.path.join(path_const, 'Si_alpha_Green_2008.dat'),
                               usecols=(0, 1), skip_header=1).transpose()
    wl = alpha_data[0]
    alpha = alpha_data[1]
    return np.interp(given_wl, wl, alpha)


def fit_Basore(wavelength, IQE, theta=0, wlbounds=(1040, 1100)):
    '''
    Linear fit of IQE to extract effective bulk lifetime

    This is just a linear fit over limited wavelengths
    Extracts an effective bulk diffusion length.

    Returns:
        a tuple of
            a dictionary containing
                L_eff: the effective diffusion length (cm)
            a plotting function

    See Basore 1993
    doi:10.1109/PVSC.1993.347063
    '''
    index = (wavelength >= wlbounds[0]) * (wavelength <= wlbounds[1])

    IQE = np.copy(IQE[index])
    wavelength = np.copy(wavelength[index])
    # print(1/ IQE)
    # print(1/wavelength)

    fit_params = ['Leff']
    alpha = wl_to_alpha(wavelength) / float(np.cos(np.radians(theta)))
    coefs = poly.polyfit(1. / alpha, 1. / IQE, 1)

    # xxx check these calcs
    fit_output = {'Leff': coefs[1],
                  'eta_c': 1 / coefs[0]}

    def plot_Basore_fit(ax):
        ax.plot(1. / alpha, 1. / IQE, '-o', label='data')
        ax.plot(1. / alpha, poly.polyval(1. / alpha, coefs), label='fit_Basore')
        ax.set_xlabel('$1/ \\alpha$ [$cm^2$]')
        ax.set_ylabel('$1/IQE$ []')
        ax.grid(True)
        ax.legend(loc='best')

    return fit_output, plot_Basore_fit


def Rs_calc_1(Vmp, Jmp, sunsVoc_V, sunsVoc_J):
    # TODO: not finished

    # sunsVoc method
    V_sunsVoc = find_nearest(Jmp, sunsVoc_J, sunsVoc_V)
    return (V_sunsVoc - Vmp) / Jmp


def Rs_calc_2(Voc, Jsc, FF, pFF):
    '''
    TODO: improve

    From:
    Solar Cells: Operating Principles, Technology and System Applications
    taken from ernst2016efficiency
    '''
    return Voc / Jsc * (1 - FF / pFF)


def FF_ideal(Voc, Jsc=None, Rs=None, Rsh=None, T=300):
    '''
    Calculates the ideal Fill Factor of a solar cell.
    Gives option to account for series and shunt resistance.
    Valid for:
        Voc * q / k / T > 10
        rs + 1 / rsh < 0.4
    Accuracy:
        FFo:  1e-4
        FFs:  4e-3
        FF:   3e-2

    TODO: check input ranges?
    I've used Rs_2 because Rs_1 gave me artificially high FF

    Source: Green, 1982
    http://dx.doi.org/10.1016/0379-6787(82)90057-6
    '''
    # Rs -> infty,  Rsh -> 0
    voc = constants.e * Voc / constants.k / T
    FFo = (voc - np.log(voc + 0.72)) / (voc + 1)

    # Rs -> finite,  Rsh -> 0
    if not (Jsc is None or Rs is None):
        rs = Rs / Voc * Jsc
        FFs = FFo * (1 - 1.1 * rs) + rs**2 / 5.4
    else:
        # TODO: not sure if this is the best method?
        FFs = None

    # Rs -> finite,  Rsh -> finite
    if not (FFs is None or Rsh is None):
        rsh = Rsh / Voc * Jsc
        FF = FFs * (1 - (voc - 0.7) / voc * FFs / rsh)
    else:
        FF = None

    return FFo, FFs, FF


def FF_loss(Voc, Jsc, Vmp, Jmp, FFm, Rs, Rsh):
    '''
    Calculates the loss in fill factor from shunt and series resistance
    TODO: check theory
    '''
    FFo, _, _ = FF_ideal(Voc)

    FF_Rs = Jmp**2 * Rs / (Voc * Jsc)
    FF_Rsh = (Vmp + Rs * Jmp) / (Voc * Jsc * Rsh)
    FF_other = FFo - FFm - FF_Rs - FF_Rsh

    return FF_Rs, FF_Rsh, FF_other


def ideality_factor(V, J, Vth):
    '''
    Calculates the ideality factor

    This assumes that: $e^{V/mVt} >> 1$

    This log form is used as it appears to be more robust against noise.

    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        m = 1. / Vth /\
            np.gradient(np.log(J)) * np.gradient(V)
    return m
