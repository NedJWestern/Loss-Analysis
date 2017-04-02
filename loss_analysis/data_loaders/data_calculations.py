
from scipy import constants
import numpy as np


def ideality_factor(V, J, Vth):
    '''
    Calculates the ideality factor

    This assumes that: $e^{V/mVt} >> 1$

    This log form is used as it appears to be more robust against noise.

    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        m = 1. / Vth / np.gradient(np.log(J)) * np.gradient(V)
    return m


def _Vth(T=None):
    # this is here so it is the only place I need to define a default
    # temperature
    if T == None:
        T = 300
    return constants.k * T / constants.e


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
