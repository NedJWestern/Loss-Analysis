
import numpy as np
import ruamel.yaml as yaml
import openpyxl
import warnings
from .common import extwrapper


class IVSuns():
    suns = None   # in Amps
    V = None  # in Voltage
    other = None  # other values

    def __init__(self, loader=None, fname=None):

        self.suns, self.V, self.other = globals().get(loader)(fname)

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

        suns = data_array[:, 0]     # Effective Suns
        V = data_array[:, 1]
        J = data_array[:, 2]
        P = data_array[:, 3]
        Dn = data_array[:, 4]
        tau_eff = data_array[:, 5]

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
    other['J'] = data_array[:, 2]
    other['J'] = data_array[:, 3]
    other['Dn'] = data_array[:, 4]
    other['tau_eff'] = data_array[:, 5]

    return suns, V, other
