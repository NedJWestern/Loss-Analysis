# port "loss analysis v5.xlsx" by Ziv Hameiri to python3

import openpyxl
import numpy as np
import os
import re
from collections import OrderedDict
import matplotlib.pyplot as plt
import warnings
# modules for this package
import analysis
from scipy import constants

T = 300   # TODO: make optional input?
Vth = constants.k * T / constants.e


class Refl(object):

    def __init__(self, fname):
        self.load(fname)

    def process(self, f_metal=None, wlbounds=(900, 1000)):
        '''
        Performs several calculations including:
        - Weighted Average Reflection (WAR)
        - Light lost from front surface escape
        the results are loaded into attributes
        '''

        self.AM15G_Jph = analysis.AM15G_resample(self.wl)
        i_upper = (self.wl <= 1000)
        self.WAR = (np.dot(self.refl[i_upper], self.AM15G_Jph[i_upper])
                    / np.sum(self.AM15G_Jph[i_upper]))

        if f_metal is None:
            index = (self.wl >= 400) * i_upper
            refl_min = np.amin(self.refl[index])
            self.f_metal = refl_min
        else:
            self.f_metal = f_metal

        index_l = (self.wl >= wlbounds[0])
        index = (self.wl <= wlbounds[1]) * index_l

        # use numpys implementation for line fitting
        popt, pcov = np.polyfit(self.wl[index], self.refl[index], 1, cov=True)

        self.refl_wo_escape = np.copy(self.refl)
        self.refl_wo_escape[index_l] = np.polyval(popt, self.wl[index_l])

        Jloss = {}
        Jloss['R'] = np.dot(self.refl, self.AM15G_Jph)
        Jloss['R_wo_escape'] = np.dot(self.refl_wo_escape, self.AM15G_Jph)
        self.Jloss = Jloss

    def plot(self, ax):
        ax.plot(self.wl, self.refl, '-o')
        ax.plot(self.wl, self.refl_wo_escape, '-o')
        ax.plot(self.wl, np.ones(len(self.wl)) * self.f_metal, 'r-')
        ax.set_ylabel('Reflectance [%]')
        ax.grid(True)

    def plot_QE(self, ax):
        ax.fill_between(self.wl, 100 - self.refl,
                        100 - self.refl_wo_escape)
        ax.legend(loc='best')
        # ax.set_ylabel('Reflectance [%]')
        # ax.grid(True)

    def load(self, raw_data_file):
        '''Loads Reflectance data in attributes'''
        self.filepath = raw_data_file
        self.filename = os.path.basename(raw_data_file)

        data_array = np.genfromtxt(raw_data_file, usecols=(0, 1), skip_header=1,
                                   delimiter=',').transpose()

        # is this needed?
        if data_array[0, 0] > data_array[0, -1]:
            data_array = data_array[:, ::-1]

        self.wl = data_array[0, :]
        self.refl = data_array[1, :]

class QE(object):

    def __init__(self, fname):
        self.load(fname)

    def process(self, refl):
        '''
        Performs several calculations from QE and Reflectance data including:
        - IQE
        - Leff and SRV_rear
        the results are saved into attributes
        '''
        self.IQE = 100 * self.EQE / (100 - refl)

        self.output_Basore_fit, self.plot_Basore_fit = analysis.fit_Basore(
            self.wl, self.IQE)

    def plot_EQE(self, ax):

        line_EQE = ax.plot(self.wl, self.EQE, '-o', label='EQE')
        ax.set_xlabel('Wavelength [$nm$]')
        ax.set_ylabel('QE [%]')
        ax.legend(loc='best')
        ax.grid(True)
        return line_EQE     # currently not working

    def plot_IQE(self, ax):
        ax.plot(self.wl, self.IQE, '-o', label='IQE')
        ax.set_xlabel('Wavelength [$nm$]')
        ax.set_ylabel('QE [%]')
        ax.legend(loc='best')
        ax.grid(True)

    def load(self, raw_data_file):
        '''Loads EQE data into attributes'''
        self.filepath = raw_data_file
        self.filename = os.path.basename(raw_data_file)

        # the other columns are ignored
        data_array = np.genfromtxt(raw_data_file, usecols=(0, 1),
                                   skip_header=1, skip_footer=8)
        self.wl = data_array[:, 0]
        self.EQE = data_array[:, 1]

        f = open(raw_data_file, 'r')
        d = {}
        for line in f.readlines()[-7:-1]:
            d.update(dict([line.strip('\n').split(':')]))

        d['Jsc'] = round(float(d['Jsc']) / 1e3, 7)
        self.output = d

class IVLight(object):

    def __init__(self, fname):
        self.load(fname)

    def process(self, Rsh, Rs):
        '''Light IV calculations'''

        FFo, FFs, FF = analysis.FF_ideal(self.output['Voc'],
                                         Jsc=self.output['Jsc'], Rs=Rs, Rsh=Rsh)

        self.FF_vals = {}
        self.FF_vals['FFo'] = FFo
        self.FF_vals['FFs'] = FFs
        self.FF_vals['FF'] = FF

    def plot(self, ax):
        ax.plot(self.V, self.J, '-o', label='light IV')
        ax.set_xlabel('Voltage [$V$]')
        ax.set_ylabel('Current Density [$A cm^{-2}$]')
        ax.grid(True)
        # ax.legend(loc='best')

    def load(self, raw_data_file):
        '''Loads Light IV data in attributes'''
        self.filepath = raw_data_file
        self.filename = os.path.basename(raw_data_file)

        f = open(raw_data_file, 'r')
        d = OrderedDict()
        # rows which contain floats in lightIV data file header
        float_rows = [2]
        float_rows.extend(list(range(6, 18)))
        for i, line in enumerate(f.readlines()[1:19]):
            # convert to float for future calculations
            if i in float_rows:
                key_temp, val = line.strip('\n').split(':\t')
                key = key_temp.strip()
                d[key] = float(val)
            else:
                # d.update(dict(re.findall(r'([\s\S]+)\s*:\t([^\n]+)', line)))
                d.update(dict([line.strip('\n').split(':\t')]))

        data_array = np.genfromtxt(raw_data_file, skip_header=20)
        self.V = data_array[:, 0]
        self.J = data_array[:, 1] / d['Cell Area (sqr cm)']

        self.output = d

class IVSuns(object):
    filepath = None
    filename = None

    def __init__(self, fname):
        self.load(fname)

    def process(self):
        '''Suns Voc calculations'''

        with np.errstate(divide='ignore', invalid='ignore'):
            self.m = 1 / Vth * self.effsuns \
                / (np.gradient(self.effsuns) / np.gradient(self.V))

    def plot_IV(self, ax):
        ax.plot(self.V, self.J, '-o', label='suns Voc')
        ax.set_xlabel('Voltage [$V$]')
        ax.set_ylabel('Current Density [$A cm^{-2}$]')
        ax.grid(True)
        ax.legend(loc='best')
        ax.set_ylim(ymin=0)

    def plot_tau(self, ax):
        # TODO: trims off some noise, use better method?
        ax.loglog(self.Dn[5:-5], self.tau_eff[5:-5], '-o',
                  label='Suns Voc')
        ax.set_xlabel('$\Delta n$ [$cm^{-3}$]')
        ax.set_ylabel(r'$\tau_{eff}$ [s]')
        ax.grid(True)
        ax.legend(loc='best')
        # ax.set_xlim(xmin=1e11)

    def plot_m(self, ax):
        # trims some noise at ends of array
        ax.plot(self.V[10:-5], self.m[10:-5], '-o', label='suns Voc')
        ax.set_xlabel('Voltage [$V$]')
        ax.set_ylabel('Ideality Factor []')
        ax.grid(True)
        ax.legend(loc='best')
        ax.set_ylim(ymin=0)

    def load(self, raw_data_file, text_format=False):
        '''Loads Suns Voc data in attributes'''

        self.filepath = raw_data_file
        self.filename = os.path.basename(raw_data_file)

        if text_format:
            data_array = np.genfromtxt(raw_data_file, usecols=(0, 1, 2, 3, 4),
                                       skip_header=1)
        else:
            # suppress annoying warning
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                wb = openpyxl.load_workbook(raw_data_file, read_only=True,
                                            data_only=True)
            ws_RawData = wb.get_sheet_by_name('RawData')
            ws_User = wb.get_sheet_by_name('User')

            last_cell = 'J' + str(ws_RawData.max_row)
            data_array = np.array([[i.value for i in j] for j in
                                   ws_RawData['E2':last_cell]])

            # try: ??
            # np.asarray(xlSheet.Range("A9:I133").Value, dtype=np.float64)

            params = [i.value for i in ws_User['A5':'F5'][0]]
            vals = [i.value for i in ws_User['A6':'F6'][0]]
            self.params = dict(zip(params, vals))

            params = [i.value for i in ws_User['A8':'L8'][0]]
            # Reduce 13 significant figures in .xlsx file to 6 (default of .format())
            # vals = [float('{:f}'.format(i.value)) for i in ws_User['A6':'F6'][0]]
            vals = [float('{:e}'.format(i.value))
                    for i in ws_User['A9':'L9'][0]]
            self.output = dict(zip(params, vals))

        self.effsuns = data_array[:, 0]     # Effective Suns
        self.V = data_array[:, 1]
        self.J = data_array[:, 2]
        self.P = data_array[:, 3]
        self.Dn = data_array[:, 4]
        self.tau_eff = data_array[:, 5]

class IVDark(object):

    def __init__(self, fname):
        self.load(fname)

    def process(self):
        '''Dark IV calculations'''

        # Ideality factor
        with np.errstate(divide='ignore', invalid='ignore'):
            self.m = 1 / Vth * self.J / (np.gradient(self.J) / np.gradient(self.V))

        # Shunt resistance, at 30mV
        # TODO: do linear fit with zero intercept?
        Rsh = 0.03 / analysis.find_nearest(0.03, self.V, self.J)

        return Rsh

    def plot_IV(self, ax):
        ax.semilogy(self.V, self.J, '-o', label='data')
        ax.set_xlabel('Voltage [$V$]')
        ax.set_ylabel('Current Density [$A cm^{-2}$]')
        ax.grid(True)
        # ax.legend(loc='best')

    def plot_m(self, ax):
        ax.plot(self.V, self.m, '-o', label='dark IV')
        ax.set_xlabel('Voltage [$V$]')
        ax.set_ylabel('Ideality Factor []')
        ax.grid(True)
        ax.legend(loc='best')

    def load(self, raw_data_file):
        '''Loads Dark IV data in attributes'''
        self.filepath = raw_data_file
        self.filename = os.path.basename(raw_data_file)

        f = open(raw_data_file, 'r')
        d = OrderedDict()
        # rows which contain floats in lightIV data file header
        float_rows = [1, 6, 7, 8]
        for i, line in enumerate(f.readlines()[1:10]):
            # convert to float for future calculations
            key, val = line.strip('\n').split(':\t')
            if i in float_rows:
                d[key] = float(val)
            else:
                d[key] = val
                # d.update(dict(re.findall(r'([\s\S]+)\s*:\t([^\n]+)', line)))
                # d.update(dict([line.strip('\n').split(':\t')]))

        # for line in f.readlines()[1:10]:
        #     d.update(dict(re.findall(r'([\s\S]+)\s*:\t([^\n]+)', line)))

        # d['Cell Area in sqr cm'] = float(d['Cell Area in sqr cm'])
        self.output = d

        data_array = np.genfromtxt(
            raw_data_file, usecols=(0, 1), skip_header=11)
        self.V = data_array[:, 0]
        self.J = data_array[:, 1] / d['Cell Area in sqr cm']

class Cell(object):

    def __init__(self, thickness=0.019, **kwargs):
        self.thickness = thickness  # [cm]
        self.sample_names = {}
        self.input_errors = {}
        self.refl = Refl(kwargs['reflectance_fname'])
        self.qe = QE(kwargs['EQE_fname'])
        self.sunsVoc = IVSuns(kwargs['suns Voc_fname'])
        self.div = IVDark(kwargs['dark IV_fname'])
        self.liv = IVLight(kwargs['light IV_fname'])
        self.check_input_vals()

        self.example_dir = os.path.join(os.pardir, 'example_cell')

    def check_input_vals(self):
        '''
        Check the input cell parameters are consistent between measurements.
        Gives the error as a percentage.
        '''

        # sample names
        self.sample_names['Light IV'] = self.liv.output['Cell Name ']
        self.sample_names['Suns Voc'] = self.sunsVoc.params['Sample Name']
        self.sample_names['Dark IV'] = self.div.output['Cell Name']

        # Cell area
        # tolerance = 1e-3
        liv = self.liv.output['Cell Area (sqr cm)']
        div = self.div.output['Cell Area in sqr cm']
        delta = (div - liv) / liv
        self.input_errors['Cell Area'] = delta

        # thickness
        user_input_t = self.thickness
        sunsVoc_t = self.sunsVoc.params['Wafer Thickness (cm)']
        delta = (sunsVoc_t - user_input_t) / user_input_t
        self.input_errors['Cell thickness'] = delta

        # Voc
        liv = self.liv.output['Voc']
        div = self.sunsVoc.output['Voc (V)']
        delta = (div - liv) / liv
        self.input_errors['Voc'] = delta

        # Jsc
        liv = self.liv.output['Jsc']
        iqe = self.qe.output['Jsc']
        delta = (iqe - liv) / liv
        self.input_errors['Jsc'] = delta

    def collect_outputs(self):
        '''Collects input and output parameters into self.output_list'''

        output_list = []

        def quick_print(key, val):
            output_list.append('{:>30}, {:<20}'.format(key, val))

        output_list.append('\n')
        quick_print('##### Check inputs', '',)

        for key, val in self.sample_names.items():
            quick_print(key, '{:s}'.format(val))
        for key, val in self.input_errors.items():
            quick_print(key, '{:.3e}'.format(val))

        output_list.append('\n')
        quick_print('##### Reflectance', '')
        quick_print('filename', self.refl.filename)
        quick_print('WAR', '{:.3f}'.format(self.refl.WAR))
        quick_print('f_metal', '{:.3f}'.format(self.refl.f_metal))
        for key, val in self.refl.Jloss.items():
            quick_print(key, '{:.3f}'.format(val))

        output_list.append('\n')
        quick_print('##### QE', '')
        quick_print('filename', self.qe.filename)
        for key, val in self.qe.output.items():
            quick_print(key, val)
        quick_print('Basore fit Leff', '{:.3e}'.format(
            self.qe.output_Basore_fit['Leff']))
        quick_print('Basore fit eta_c', '{:.3f}'.format(
            self.qe.output_Basore_fit['eta_c']))

        output_list.append('\n')
        quick_print('##### Light IV', '')
        quick_print('filename', self.liv.filename)
        for key, val in self.liv.output.items():
            quick_print(key, val)

        output_list.append('\n')
        quick_print('##### Suns Voc', '')
        quick_print('filename', self.sunsVoc.filename)
        for key, val in self.sunsVoc.params.items():
            quick_print(key, val)
        for key, val in self.sunsVoc.output.items():
            quick_print(key, val)

        output_list.append('\n')
        quick_print('##### Dark IV', '')
        quick_print('filename', self.div.filename)
        for key, val in self.div.output.items():
            quick_print(key, val)

        output_list.append('\n')
        quick_print('##### Calclated', '')
        quick_print('Rsh', '{:.3e}'.format(self.Rsh))
        quick_print('Rs1', '{:.3e}'.format(self.Rs_1))
        quick_print('Rs2', '{:.3e}'.format(self.Rs_2))

        for key, val in self.liv.FF_vals.items():
            quick_print(key, '{:.3f}'.format(val))

        self.output_list = output_list

    def print_output_to_file(self):

        filename = self.cell_name + '_loss_analysis_summary.csv'

        output_file = open(os.path.join(self.output_dir, filename), 'w')

        for item in self.output_list:
            output_file.write(item + '\r\n')

        output_file.close()

    def plot_all(self, save_fig_bool):
        '''Plot the output of previous calculations'''
        # for reflectance

        fig_QE = plt.figure('QE', figsize=(30 / 2.54, 15 / 2.54))
        fig_QE.clf()
        # for light and dark IV
        fig_IV = plt.figure('IV', figsize=(30 / 2.54, 15 / 2.54))
        fig_IV.clf()

        ax_refl = fig_QE.add_subplot(2, 2, 1)
        ax_QE = fig_QE.add_subplot(2, 2, 2)
        ax_QE_fit = fig_QE.add_subplot(2, 2, 3)
        ax_QE_layered = fig_QE.add_subplot(2, 2, 4)

        ax_darkIV = fig_IV.add_subplot(2, 2, 1)
        ax_ideality = fig_IV.add_subplot(2, 2, 3)
        ax_lightIV = fig_IV.add_subplot(2, 2, 2)
        ax_tau = fig_IV.add_subplot(2, 2, 4)

        self.refl.plot(ax_refl)
        self.refl.plot(ax_QE)
        self.qe.plot_EQE(ax_QE)
        self.qe.plot_IQE(ax_QE)

        self.sunsVoc.plot_m(ax_ideality)
        self.sunsVoc.plot_IV(ax_lightIV)
        self.sunsVoc.plot_tau(ax_tau)
        self.liv.plot(ax_lightIV)

        self.div.plot_IV(ax_darkIV)
        self.div.plot_m(ax_ideality)

        self.qe.plot_Basore_fit(ax_QE_fit)

        dummy_ones = np.ones(len(self.refl.wl))
        ax_QE_layered.fill_between(self.refl.wl, dummy_ones * 100,
                                   100 - dummy_ones * self.refl.f_metal,  color='blue')
        ax_QE_layered.fill_between(self.refl.wl,
                                   100 - dummy_ones * self.refl.f_metal,
                                   100 - self.refl.refl_wo_escape, color='green')
        ax_QE_layered.fill_between(self.refl.wl, 100 - self.refl.refl,
                                   100 - self.refl.refl_wo_escape, color='red')
        ax_QE_layered.fill_between(self.refl.wl, 100 - self.refl.refl,
                                   self.qe.EQE, color='cyan')
        # line_EQE, = self.qe.plot_EQE(ax_QE_layered)
        # line_EQE.set_marker('x')
        # self.refl.plot_QE(ax_QE_layered)

        fig_QE.set_tight_layout(True)
        fig_IV.set_tight_layout(True)

        if save_fig_bool:
            fig_QE.savefig(os.path.join(self.output_dir,
                                        self.cell_name + '_QE.png'))
            fig_IV.savefig(os.path.join(self.output_dir,
                                        self.cell_name + '_IV.png'))

        plt.show()

    def process_all(self, save_fig_bool, output_dir, cell_name):
        '''Call all calculations'''

        if cell_name=='':
            self.cell_name = self.liv.output['Cell Name ']
        else:
            self.cell_name = cell_name

        self.output_dir = output_dir

        self.sunsVoc.process()
        self.refl.process()
        self.qe.process(self.refl.refl)
        self.Rsh = self.div.process()

        self.Rs_1 = analysis.Rs_calc_1(self.liv.output['Vmp'],
                                            self.liv.output['Jmp'],
                                            self.sunsVoc.V, self.sunsVoc.J)

        self.Rs_2 = analysis.Rs_calc_2(self.liv.output['Voc'],
                                            self.liv.output['Jsc'],
                                            self.liv.output['FF'],
                                            self.sunsVoc.output['PFF'])

        self.liv.process(self.Rsh, self.Rs_1)

        self.collect_outputs()
        self.print_output_to_file()
        self.plot_all(save_fig_bool)



if __name__ == "__main__":

    example_dir = os.path.join(os.pardir, 'example_cell')
    files = {
        'reflectance_fname': os.path.join(example_dir, 'example_reflectance.csv'),
        'EQE_fname': os.path.join(example_dir, 'example_EQE.txt'),
        'light IV_fname': os.path.join(example_dir, 'example_lightIV.lgt'),
        'suns Voc_fname': os.path.join(example_dir, 'example_sunsVoc.xlsm'),
        'dark IV_fname': os.path.join(example_dir, 'example_darkIV.drk') }

    cell1 = Cell(**files)
    cell1.process_all()
