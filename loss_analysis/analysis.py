import numpy as np
from scipy import constants
from scipy.optimize import curve_fit
import os
from numpy.polynomial import polynomial as poly
from scipy.special import lambertw

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
    Returns the band to band absorption coefficient for Silicon given a
    wavelength. Linear interpolation is performed if the exact values are
    not provided.

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
                The average angle the light travels through the sample.
                This can be used to partially correct for textured surfaces.
                The default is 0. In units of degrees,
        wlbounds: (tuple, optional)
                The bounds between which the linear fit is performed.
                The first touple should be the min and then the max.
                The default is 1040 nm to 1100 nm.
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


def _Vth(T):
    # this is here so it is the only place I need to define a default
    # temperature
    # if T == None:
        # T = 300
    return constants.k * T / constants.e


def ideal_FF(Voc, T=None):
    '''
    Calculates the ideal fill factor.

    inputs:
        Voc: (float)
            Open circuit voltage in volts
        T: (float, optional)
            Temperature in Kelvin, default of 300K

    output:
        FF_0:
            The ideal fill factor

    Valid for:
        Voc * q / k / T > 10
    Accuracy: 1e-4

    Source: Green, 1982
    http://dx.doi.org/10.1016/0379-6787(82)90057-6
    '''
    voc = Voc / _Vth(T)
    FF_0 = (voc - np.log(voc + 0.72)) / (voc + 1)
    return FF_0


def ideal_FF_2016(Voc, T=None):
    '''
    Calculates the ideal fill factor.

    inputs:
        Voc: (float)
            Open circuit voltage in volts
        T: (float, optional)
            Temperature in Kelvin, default of 300K

    output:
        FF_0:
            The ideal fill factor

    Valid for:
        ??
    Accuracy: ??

    Source: Green, 2016
    http://dx.doi.org/10.1063/1.4942660
    '''

    voc = Voc / _Vth(T)

    z0 = np.exp(voc + 1)
    # inverse f0
    if0 = 1. - np.exp(-voc)
    FF_0 = (lambertw(z0) - 1)**2 / if0 / voc / lambertw(z0)
    return FF_0.real


def ideal_FF_series(Voc, Jsc, Rs, T=None):
    '''
    Calculates the ideal fill factor accounting for series resistance

        inputs:
            Voc: (float)
                Open circuit voltage in volts
            Jsc: (float)
                The short circuit current in amps
            Rs: (float)
                The series resistance in Ohms?
            T: (float)
                Temperature in Kelvin

        output:
            FF_s:
                The ideal fill factor accounting for series resistance

    Valid for:
        Voc * q / k / T > 10
        Rs * Jsc / Voc < 0.4
    Accuracy: 4e-3

    Source: Green, 1982
    http://dx.doi.org/10.1016/0379-6787(82)90057-6
    '''
    FF_0 = ideal_FF(Voc, T)
    rs = Rs / Voc * Jsc
    FF_s = FF_0 * (1 - 1.1 * rs) + rs**2 / 5.4
    return FF_s


def ideal_FF_series_2016(Voc, Jsc, Rs, T=None):
    '''
    Calculates the ideal fill factor.

    inputs:
        Voc: (float)
            Open circuit voltage in volts
        T: (float, optional)
            Temperature in Kelvin, default of 300K

    output:
        FF_0:
            The ideal fill factor

    Valid for:
        ??
    Accuracy: Approximately 4 digit accuracy is maintained in
    technologically interesting cases, where losses are <5% for
    normalised Voc>10.

    Source: Green, 2016
    http://dx.doi.org/10.1063/1.4942660
    '''

    FF_0 = ideal_FF_2016(Voc, T)
    # normalised values
    voc = Voc / _Vth(T)
    rs = Rs / Voc * Jsc

    # other factors
    if0 = 1. - np.exp(-voc)
    ifs = 1. - np.exp(-voc * (1 - rs))
    z0 = np.exp(voc + 1)

    # calculate it
    FF_s = FF_0 * (1 - voc / lambertw(z0) * rs / if0) * if0 / ifs
    return FF_s.real


def ideal_FF_series_shunt(Voc, Jsc, Rs, Rsh, T=None):
    '''
    Calculates the ideal fill factor, accounting for shunt and series resistance.
    inputs:
        Voc: (float)
            Open circuit voltage in volts
        Jsc: (float)
            The short circuit current in amps
        Rs: (float)
            The series resistance in Ohms?
        Rsh: (float)
            The shunt resistance in Ohms?
        T: (float)
            Temperature in Kelvin

    output:
        FF_sh_s:
            The ideal fill factor accounting for shunt and series resistance

    Valid for:
        Voc * q / k / T > 10
         < 0.4
        Rs * Jsc / Voc + Voc / Rsh / Jsc < 0.4
    Accuracy: 3e-2

    Source: Green, 1982
    http://dx.doi.org/10.1016/0379-6787(82)90057-6
    '''
    FF_s = ideal_FF_series(Voc, Jsc, Rs, T)
    voc = Voc / _Vth(T)
    rsh = Rsh / Voc * Jsc
    FF_s_sh = FF_s * (1 - (voc - 0.7) / voc * FF_s / rsh)
    return FF_s_sh


def ideal_FF_shunt_2016(Voc, Rsh, T=None):
    '''
    Calculates the ideal fill factor, accounting for shunt and series resistance.
    inputs:
        Voc: (float)
            Open circuit voltage in volts
        Jsc: (float)
            The short circuit current in amps
        Rs: (float)
            The series resistance in Ohms?
        Rsh: (float)
            The shunt resistance in Ohms?
        T: (float)
            Temperature in Kelvin

    output:
        FF_sh_s:
            The ideal fill factor accounting for shunt and series resistance

    Valid for:
        Voc * q / k / T > 10
         < 0.4
        Rs * Jsc / Voc + Voc / Rsh / Jsc < 0.4
    Accuracy: 3e-2

    Source: Green, 1982
    http://dx.doi.org/10.1016/0379-6787(82)90057-6
    '''
    FF_0 = ideal_FF_2016(Voc, T)
    # normalised values
    voc = Voc / _Vth(T)
    rsh = Rsh / Voc * Jsc

    # other factors
    if0 = 1. - np.exp(-voc)
    z0 = np.exp(voc + 1)

    # calculate it
    FF_sh = FF_0 * (1 - lambertw(z0) * if0 / voc /
                    rsh / if0) / (1 - 1 / (voc * rsh))
    return FF_sh.real


def FF_loss_series(Voc, Jsc, Jmp, Rs):
    '''
    Calculates the loss in fill factor from series resistance

    inputs:
        Voc: (float)
            Open circuit voltage in [V]
        Jsc: (float)
            Short circuit current density in [A cm^{-1}]
        Jmp: (float)
            Maximum power point current density in [A cm^{-2}]
        Rs: (float)
            Series resistance in [Ohm cm^2]

    output:
        FF_Rs: (float)
            The increase in fill factor expected by removing the series resistance
            Dimensionless units

    Source: Khanna, 2013
    http://dx.doi.org/10.1109/JPHOTOV.2013.2270348
    '''
    FF_Rs = Jmp**2 * Rs / (Voc * Jsc)

    return FF_Rs


def FF_loss_shunt(Voc, Jsc, Vmp, Jmp, Rs, Rsh):
    '''
    Calculates the loss in fill factor from shunt resistance

    inputs:
        Voc: (float)
            Open circuit voltage in [V]
        Jsc: (float)
            Short circuit current density in [A cm^{-1}]
        Vmp: (float)
            Maximum power point voltage in [V]
        Jmp: (float)
            Maximum power point current density in [A cm^{-2}]
        Rs: (float)
            Series resistance in [Ohm cm^2]
        Rsh: (float)
            Shunt resistance in [Ohm cm^2]

    output:
        FF_Rs: (float)
            The increase in fill factor expected by removing the series resistance
            Dimensionless units

    Source: Khanna, 2013
    http://dx.doi.org/10.1109/JPHOTOV.2013.2270348
    '''
    FF_Rsh = (Vmp + Rs * Jmp)**2 / (Voc * Jsc * Rsh)

    return FF_Rsh


def ideality_factor(V, J, Vth):
    '''
    Calculates the ideality factor

    This assumes that: $e^{V/mVt} >> 1$

    This log form is used as it appears to be more robust against noise.

    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        m = 1. / Vth / np.gradient(np.log(J)) * np.gradient(V)
    return m


if __name__ == '__main__':
    print(ideal_FF(0.6, 300))
    print(ideal_FF_2016(0.6, 300))

    print(ideal_FF_series_2016(0.6, 0.04, 1, 300))
    print(ideal_FF_series(0.6, 0.04, 1, 300))
