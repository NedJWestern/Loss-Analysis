
import numpy as np
import ruamel.yaml as yaml
import openpyxl
import warnings
from .common import extwrapper
from .data_calculations import ideality_factor, _Vth


class IVSuns():
    suns = None   # in Amps
    V = None  # in Voltage
    temp = 300  # in Kelvin
    other = None  # other values

    def __init__(self, loader=None, fname=None):

        self.suns, self.V, self.other = globals().get(loader)(fname)

        keys = dict(self.other).keys()
        for key in keys:
            if hasattr(self, key):
                setattr(self, key, self.other[key])
                del self.other[key]

        self.process()

    def process(self):
        if 'J' in self.other:
            J = self.other['J']
            self.m = ideality_factor(self.V, J, _Vth(self.temp))

    def plot_JV(self, ax):
        '''
        Plots the current voltage curve

        inputs:
            ax: A figure axes to which is plotted
        '''
        if 'J' in self.other:
            J = self.other['J']
            ax.plot(self.V, J, '.-', label='Suns Voc')
            ax.set_xlabel('Voltage [$V$]')
            ax.set_ylabel('Current Density [$A cm^{-2}$]')
            ax.grid(True)

    def plot_log_JV(self, ax):
        '''
        Plots the current voltage curve

        inputs:
            ax: A figure axes to which is plotted
        '''
        if 'J' in self.other:
            J = self.other['J']
            index = np.isfinite(J)
            index *= self.V[index] > -0.1
            Jsc_index = abs(self.V[index]) == np.min(abs(self.V[index]))
            print(J[Jsc_index])

            ax.plot(self.V, -J +
                    abs(J[Jsc_index]), '.-', label='Suns Voc')
            ax.set_xlabel('Voltage [$V$]')
            ax.set_ylabel('Current Density [$A cm^{-2}$]')
            ax.grid(True)

    def plot_mV(self, ax):
        '''
        Plots the current voltage curve

        inputs:
            ax: A figure axes to which is plotted
        '''
        if 'Vth' in self.other:
            ax.plot(self.V, self.m, '.-', label='Suns Voc')
            ax.set_xlabel('Voltage [$V$]')
            ax.set_ylabel('Current Density [$A cm^{-2}$]')
            ax.grid(True)

    def plot_tau(self, ax):
        if'nxc' in self.other:
            nxc = self.other['nxc']
            tau_eff = self.other['tau_eff']

            ax.loglog(Dn, tau_eff, '.-',
                      label='Suns Voc')
            ax.set_xlabel('$\Delta n$ [$cm^{-3}$]')
            ax.set_ylabel(r'$\tau_{eff}$ [s]')
            ax.grid(True)
            ax.legend(loc='best')

loader_file_ext = {
    'plain_text': '.txt',
    'sinton': '.xlsm'
}


@extwrapper(loader_file_ext)
def plain_text(fname):
    '''Loads Suns Voc data from a plain text file '''

    if text_format:
        Suns, Voc = np.genfromtxt(raw_data_file, usecols=(0, 1, 2, 3, 4),
                                  skip_header=1, unpack=True)

    return suns, Voc, other


@extwrapper(loader_file_ext)
def sinton(fname):

    # suppress annoying warning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        wb = openpyxl.load_workbook(fname, read_only=True,
                                    data_only=True)
        # make references to the sheets
        ws_RawData = wb.get_sheet_by_name('RawData')
        ws_User = wb.get_sheet_by_name('User')

        # grab the data
        last_cell = 'J' + str(ws_RawData.max_row)
        data_array = np.array([[i.value for i in j] for j in
                               ws_RawData['E2':last_cell]])

        suns = data_array[:, 0][5:]     # Effective Suns
        V = data_array[:, 1][5:]

        # now just grabbing some other stuff from the excel
        _params = [i.value for i in ws_User['A5':'F5'][0]]
        vals = [i.value for i in ws_User['A6':'F6'][0]]

        params = dict(zip(_params, vals))

        _params = [i.value for i in ws_User['A8':'L8'][0]]
        # Reduce 13 significant figures in .xlsx file to 6 (default of .format())
        # vals = [float('{:f}'.format(i.value)) for i in
        # ws_User['A6':'F6'][0]]
        vals = [float('{:e}'.format(i.value))
                for i in ws_User['A9':'L9'][0]]
    other = dict(zip(params, vals))
    other.update(params)
    other['J'] = data_array[:, 2][5:]
    other['P'] = data_array[:, 3][5:]
    other['nxc'] = data_array[:, 4][5:-5]
    other['tau_eff'] = data_array[:, 5][5:-5]

    return suns, V, other
