import numpy as np
from scipy import constants
from scipy.optimize import curve_fit
import os
from numpy.polynomial import polynomial as poly

# use absolute file path so tests work
path_const = os.path.join(os.path.dirname(__file__), '..', 'constants')


def AM15G_resample(wl):
    '''Returns AM1.5G spectrum at given wavelengths'''
    AM15G_wl = np.genfromtxt(os.path.join(path_const, 'AM1.5G_spectrum.dat'),
                             usecols=(0,), skip_header=1)
    AM15G_Jph = np.genfromtxt(os.path.join(path_const, 'AM1.5G_spectrum.dat'),
                              usecols=(1,), skip_header=1)
    return np.interp(wl, AM15G_wl, AM15G_Jph)


def find_nearest(x_val, xdata, ydata=None):
    '''
    Finds the nearest index in 'xdata' to 'value'
    Returns corresponding 'ydata' value if given
    '''

    xdata = np.array(xdata)

    nearest = (np.abs(xdata - x_val)).argmin()

    if ydata is not None:
        ydata = np.array(ydata)

        assert xdata.shape[0] == ydata.shape[0]
        nearest = ydata[nearest]

    return nearest


def wl_to_alpha(wavelength):
    '''
    Returns the band to band absorption coefficient for Silicon given a wavelength. Linear interpolation is performed if the exact values are not provided.

    The values are taken from Green 2008
    DOI:10.1016/j.solmat.2008.06.009

    inputs:
        wavelength: (float)
                wavelength in nm
    outputs:
        wavelength: (float)
                wavelength in nm

    '''
    alpha_data = np.genfromtxt(
        os.path.join(path_const, 'Si_alpha_Green_2008.dat'),
        usecols=(0, 1), skip_header=1).transpose()

    wl = alpha_data[0]
    alpha = alpha_data[1]

    return np.interp(wavelength, wl, alpha)


def fit_Basore(wavelength, IQE, theta=0, wlbounds=(1040, 1100)):
    '''
    Linear fit of IQE to extract effective bulk lifetime

    This is just a linear fit over limited wavelengths
    Extracts an effective bulk diffusion length.

    inputs:
        wavelength: (array like)
                the measured wavelengths in nano meters.
        IQE:    (array like)
                the measured internal quantum efficiency in units %.
        theta: (float, optional)
                The average angle the light travels through the sample. This can be used to partially correct for textured surfaces. The default is 0. In units of degrees,
        wlbounds: (tuple, optional)
                The bounds between which the linear fit is performed. The first touple should be the min and then the max. The default is 1040 nm to 1100 nm.
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

    inputs:
        Voc: (float)
            The open circuit voltage of the device in volts
        Jsc: (float, optional)
            The short circuit current of the device in amps
        Rs: (float, optional)
            The lumped series resistance of the device in Ohms?
        Rsh: (float, optional)
            The lumped shunt resistor for the device in Ohms?
        T: (float, optional)
            The temperature, in Kelvin, of the device, the default is 300 K.

    outputs:
        FF0: The highest obtainable fill factor
        FFs: The highest obtainable fill factor including the series resistance limit
        FF:  The highest obtainable fill factor including the effects of series and shunt resistance.

    Notes:
    I've used Rs_2 because Rs_1 gave me artificially high FF

    Source: Green, 1982
    http://dx.doi.org/10.1016/0379-6787(82)90057-6
    '''

    if Rsh is not None:
        # Rs -> finite,  Rsh -> 0
        FF = _ideal_FF_shunt_series(Voc, T, Jsc, Rs, Rsh)
    elif not (Jsc is None or Rs is None):
        # Rs -> finite,  Rsh -> finite
        FF = _ideal_FF_series(Voc, T, Jsc, Rs)
    else:
        # Rs -> infty,  Rsh -> 0
        FF = _ideal_FF(Voc, T)

    return FF


def _ideal_FF(Voc, T):
    '''
    Calculates the ideal fill factor.

    inputs:
        Voc: (float)
            Open circuit voltage in volts
        T: (float)
            Temperature in Kelvin

    output:
        FF:
            The ideal fill factor
    '''
    voc = constants.e * Voc / constants.k / T
    FF = (voc - np.log(voc + 0.72)) / (voc + 1)
    return FF


def _ideal_FF_series(Voc, T, Jsc, Rs):
    '''
    Calculates the ideal fill factor accounting for series resistance

        inputs:
            Voc: (float)
                Open circuit voltage in volts
            T: (float)
                Temperature in Kelvin
            Jsc: (float)
                The short circuit current in amps
            Rs: (float)
                The series resistance in Ohms?

        output:
            FF:
                The ideal fill factor
    '''
    FF = _ideal_FF(Voc, T)
    rs = Rs / Voc * Jsc
    FF = FF * (1 - 1.1 * rs) + rs**2 / 5.4
    return FF


def _ideal_FF_shunt_series(Voc, T, Jsc, Rs, Rsh):
    '''
    Calculates the ideal fill factor, accounting for shunt and series resistance.
    inputs:
        Voc: (float)
            Open circuit voltage in volts
        T: (float)
            Temperature in Kelvin
        Jsc: (float)
            The short circuit current in amps
        Rs: (float)
            The series resistance in Ohms?
        Rsh: (float)
            The shunt resistance in Ohms?
    output:
        FF:
            The ideal fill factor
    '''
    FF = _ideal_FF_series(Voc, Jsc, Rs, T)
    voc = constants.e * Voc / constants.k / T
    rsh = Rsh / Voc * Jsc
    FF = FF * (1 - (voc - 0.7) / voc * FF / rsh)
    return FF


def FF_loss(Voc, Jsc, Vmp, Jmp, FFm, Rs, Rsh):
    '''
    Calculates the loss in fill factor from shunt and series resistance
    TODO: check theory
    '''
    FF = FF_ideal(Voc)

    FF_Rs = Jmp**2 * Rs / (Voc * Jsc)
    FF_Rsh = (Vmp + Rs * Jmp) / (Voc * Jsc * Rsh)
    FF_other = FF - FFm - FF_Rs - FF_Rsh

    return FF_Rs, FF_Rsh, FF_other


def ideality_factor(V, J, Vth):
    '''
    Calculates the ideality factor

    This assumes that: $e^{V/mVt} >> 1$

    This log form is used as it appears to be more robust against noise.

    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        m = 1. / Vth / np.gradient(np.log(J)) * np.gradient(V)
    return m
