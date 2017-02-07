import numpy as np
from scipy import constants
from scipy.optimize import curve_fit
import os

# helper functions ###########################################################

# def fit_long_wavelengths(self, wavelength, IQE, theta=0, model='Basore',
#                         **kwargs):
#     '''
#     Fits to the long wavelengths to determine
#     the bulk diffusion length and the rear surface recombination
#     '''
#     try:
#         vals = getattr(
#             self, 'fit_' + model)(wavelength, IQE, theta=0, **kwargs)
#     except:
#         print ('Incorrect or model failed ')
#
#     #TODO: load into attributes instead of returning?
#     return vals

# def Isenberg_function(self, alpha, L, S, emitter_width=5e-5):
#     '''Used in 'fit_Isenberg' function'''
#     # TODO: check these calcs
#     D = 27.  # [cm^2/s] diffusion constant
#     W = self.thickness - emitter_width
#
#     SigmaW = (S * L / D * np.cosh(W / L) + np.sinh(W / L)) / \
#         (S * L / D * np.sinh(W / L) + np.cosh(W / L))
#
#     f0 = L ** 2. / D * alpha / (1. - (L * alpha)**2.)
#     fd0 = -alpha * f0
#     fW = f0 * np.exp(-alpha * W)
#     fdW = -alpha * fW
#
#     IQE = D / L * (SigmaW * f0 + L * fd0) - D * (S * fW / D + fdW) / (
#         S * L / D * np.sinh(W / L) + np.cosh(W / L))
#
#     if S < 0 or L < 0 or emitter_width < 0:
#         IQE *= 4    # TODO: what's this for??
#
#     return IQE * np.exp(-emitter_width * alpha)

# def fit_Isenberg(self, wavelength, IQE, theta=0, wlbounds=[700, 940]):
#     '''
#     TODO: not finished
#
#     Performs the fit using the Isenberg_function. This is considered an
#     improvement of the Basore method.
#     doi:10.1109/PVSC.2002.1190525
#
#     Returns:
#         a dictionary containing
#             L: the diffusion length (cm)
#             Srear: and the rear surface
#                    recombination velocity (cm/s)
#             WJ: width of the junction
#     '''
#
#     fit_params = ['L', 'Srear', 'Wj']
#
#     index = wavelength > wlbounds[0]
#     index *= wavelength < wlbounds[1]
#
#     IQE = IQE[index]
#     wavelength = wavelength[index]
#
#     alpha = self.wl_to_alpha(wavelength) / np.cos(np.radians(theta))
#
#     p0 = (.09, 10000, 7e-06)
#     popt, pcov = curve_fit( self.Isenberg_function, alpha, IQE, p0=p0,
#                            method='trf', bounds=(0, [100, 1e6, 0.02]))
#
#     #TODO: load into attributes instead of returning?
#     return {elem:popt[i] for i, elem in enumerate(fit_params)}

path = os.sep.join(os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-1])

def AM15G_resample(wl):
    '''Returns AM1.5G spectrum at given wavelengths'''
    AM15G_wl = np.genfromtxt(os.path.join(path,'constants', 'AM1.5G_spectrum.dat'), usecols=(0,),
                                  skip_header=1)
    AM15G_Jph = np.genfromtxt(os.path.join(path,'constants', 'AM1.5G_spectrum.dat'), usecols=(1,),
                                   skip_header=1)
    return np.interp(wl, AM15G_wl, AM15G_Jph)

def find_nearest(x_val, xdata, ydata=None):
    '''
    TODO: check xdata and ydata are equal lengths
    Finds the nearest index in 'xdata' to 'value'
    Returns corresponding 'ydata' value if given
    '''
    idx = (np.abs(xdata - x_val)).argmin()
    if ydata is None:
        return xdata[idx]
    else:
        return ydata[idx]

def wl_to_alpha(given_wl):
    '''Returns alpha for a given wavelength in [nm] xxx? in Si'''
    alpha_data = np.genfromtxt(os.path.join(path,'constants', 'Si_alpha_Green_2008.dat'),
                               usecols=(0,1), skip_header=1).transpose()
    wl = alpha_data[0]
    alpha = alpha_data[1]
    return np.interp (given_wl, wl, alpha)

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

    See Basore (xxx year)
    doi:10.1109/PVSC.1993.347063
    '''
    index = (wavelength >= wlbounds[0]) * (wavelength <= wlbounds[1])

    IQE = np.copy(IQE[index])
    wavelength = np.copy(wavelength[index])
    # print(1/ IQE)
    # print(1/wavelength)

    fit_params = ['Leff']
    # TODO: is this not already a float?
    alpha = wl_to_alpha(wavelength) / float(np.cos(np.radians(theta)))
    popt, pcov = np.polyfit(1. / alpha, 1. / IQE,1,cov=True)

    vals =  {elem: popt[i] for i, elem in enumerate(fit_params)}

    def plot_Basore_fit(ax):
        ax.plot(1. / alpha, 1. / IQE, '-o', label='data')
        ax.plot(1. / alpha, np.polyval(popt, 1. / alpha),
                label='fit_Basore')
        ax.set_xlabel('$1/ \\alpha$ [$cm^2$]')
        ax.set_ylabel('$1/IQE$ []')
        ax.grid(True)
        ax.legend(loc='best')

    return vals, plot_Basore_fit

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

def FF_ideal(Voc, Jsc = None, Rs = None, Rsh = None, T=300):
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
        #TODO: not sure if this is the best method?
        FFs = None

    # Rs -> finite,  Rsh -> finite
    if not (FFs is None or Rsh is None):
        rsh = Rsh / Voc * Jsc
        FF = FFs * (1 - (voc - 0.7) / voc * FFs / rsh)
    else:
        FF = None

    return (FFo, FFs, FF)
