# port "loss analysis v5.xlsx" by Ziv Hameiri to python3
# requires python3

import openpyxl
import numpy as np
import os
import re
from collections import OrderedDict
import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
import tkinter as tk
from tkinter.filedialog import askopenfilename  # remove this?
# modules for this package
import analysis     # requires correct current directory, change? xxx
from scipy import constants


class IVSuns():
    filepath = None
    filename = None

    def __init__(self, fname, cell):
        self.cell = cell

        self.output = None
        self.V = None
        self.J = None
        self.effsuns = None
        self.params = None

        self.load(fname)

    def process(self):
        '''Suns Voc calculations'''

        # Ideality factor, TODO: better method?
        self.m = 1 / self.cell.Vth * self.effsuns \
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
            wb = openpyxl.load_workbook(raw_data_file, read_only=True,
                                        data_only=True)
            ws_RawData = wb.get_sheet_by_name('RawData')
            ws_User = wb.get_sheet_by_name('User')

            last_cell = 'J' + str(ws_RawData.max_row)
            data_array = np.array([[i.value for i in j] for j in
                                   ws_RawData['E2':last_cell]])

            # TODO: I know this list() is shite, temporary workaround for my old
            # openpyxl install
            params = [i.value for i in list(ws_User['A5':'F5'])[0]]
            vals = [i.value for i in list(ws_User['A6':'F6'])[0]]
            self.params = dict(zip(params, vals))

            params = [i.value for i in list(ws_User['A8':'L8'])[0]]
            # Reduce 13 significant figures in .xlsx file to 6 (default of .format())
            # vals = [float('{:f}'.format(i.value)) for i in ws_User['A6':'F6'][0]]
            vals = [float('{:e}'.format(i.value))
                    for i in list(ws_User['A9':'L9'])[0]]
            self.output = dict(zip(params, vals))

        self.effsuns = data_array[:, 0]     # Effective Suns
        self.V = data_array[:, 1]
        self.J = data_array[:, 2]
        self.P = data_array[:, 3]
        self.Dn = data_array[:, 4]
        self.tau_eff = data_array[:, 5]


class IVLight():

    def __init__(self, fname, cell):
        self.cell = cell
        self.filepath = None
        self.filename = None
        self.output = None
        self.V = None
        self.J = None

        self.load(fname)

    def output(self):
        return

    def process(self):
        '''Light IV calculations'''

        FFo, FFs, FF = analysis.FF_ideal(self.output['Voc'],
                                         Jsc=self.output['Jsc'],
                                         Rs=self.cell.Rs_1,
                                         Rsh=self.cell.Rsh)

        self.FF_vals = {}
        self.FF_vals['FFo'] = FFo
        self.FF_vals['FFs'] = FFs
        self.FF_vals['FF'] = FF

        return self.cell

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
        # TODO: error check for nans and 1e12?

        self.output = d


class IVDark():

    def __init__(self, fname, cell):
        self.cell = cell
        self.filepath = None
        self.filename = None
        self.output = None
        self.V = None
        self.J = None
        self.m = None

        self.load(fname)

    def process(self):
        '''Dark IV calculations'''

        # Ideality factor
        self.m = 1 / self.cell.Vth * self.J \
            / (np.gradient(self.J) / np.gradient(self.V))

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
        # TODO: error check for nans and 1e12?


class Reflection():

    def __init__(self, fname):

        self.wl = None
        self.refl = None
        self.refl_wo_escape = None
        self.Jloss = None

        self.filepath = None
        self.filename = None

        self.load(fname)

    def process(self, f_metal=None, wlbounds=(900, 1000)):
        '''
        Performs several calculations including:
        - Weighted Average Reflection (WAR)
        - Light lost from front surface escape
        the results are loaded into attributes
        '''

        # exit early if data isn't loaded
        try:
            self.refl
        except AttributeError:
            print("Reflection data not loaded")
            return

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
                                   delimiter=',')
        self.wl = data_array[:, 0]
        self.refl = data_array[:, 1]


class QE():

    def __init__(self, fname):
        self.EQE = None
        self.IQE = None
        self.wl = None
        self.output = None

        self.filepath = None
        self.filename = None

        self.load(fname)

    def process(self, refl):
        '''
        Performs several calculations from QE and Reflectance data including:
        - IQE
        - Leff and SRV_rear
        the results are saved into attributes
        '''
        self.IQE = 100 * self.EQE / (100 - refl)

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

        self.output = d


class LossAnalysisHandler():

    # TODO: pass as dict instead of **args ?
    def __init__(self, **args):
        self.sample_names = {}
        self.input_errors = {}
        self.cell = Cell()
        self.reflection = Reflection(args['reflectance_fname'])
        self.qe = QE(args['EQE_fname'])
        self.sunsVoc = IVSuns(args['suns Voc_fname'], self.cell)
        self.liv = IVLight(args['light IV_fname'], self.cell)
        self.div = IVDark(args['dark IV_fname'], self.cell)

    def check_input_vals(self):
        '''
        Check the input cell parameters are consistent between measurements.
        Gives the error as a percentage.
        '''
        # TODO:
        # check whether data is loaded

        # sample names
        self.sample_names['Light IV'] = self.liv.output['Cell Name ']
        self.sample_names['Suns Voc'] = self.sunsvoc.output['Sample Name']
        self.sample_names['Dark IV'] = self.div.output['Cell Name']

        # Cell area
        # tolerance = 1e-3
        liv = self.liv.output['Cell Area (sqr cm)']
        div = self.div.output['Cell Area in sqr cm']
        delta = (div - liv) / liv
        self.input_errors['Cell Area'] = delta

        # thickness
        user_input_t = self.cell.thickness
        sunsVoc_t = self.sunsvoc.output['Wafer Thickness (cm)']
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
        delta = (div - liv) / liv
        self.input_errors['Jsc'] = delta

    def collect_outputs(self):
        '''Print input and output parameters to file or terminal'''

        output_list = []

        def quick_print(key, val):
            output_list.append('{:>30}, {:<20}'.format(key, val))

        output_list.append('\n')
        quick_print('##### Sample names', '',)

        for key, val in self.sample_names.items():
            quick_print(key, '{:.3e}'.format(val))
        for key, val in self.input_errors.items():
            quick_print(key, '{:.3e}'.format(val))

        output_list.append('\n')
        quick_print('##### Reflectance', '')
        quick_print('Reflectance filename', self.reflection.filename)
        quick_print('WAR', '{:.3e}'.format(self.reflection.WAR))
        quick_print('f_metal', '{:.3e}'.format(self.reflection.f_metal))
        for key, val in self.reflection.Jloss.items():
            quick_print(key, '{:.3e}'.format(val))

        output_list.append('\n')
        quick_print('##### QE', '')
        quick_print('EQE filename', self.qe.filename)
        for key, val in self.qe.output.items():
            quick_print(key, val)

        output_list.append('\n')
        quick_print('##### Light IV', '')
        quick_print('EQE filename', self.liv.filename)
        for key, val in self.liv.output.items():
            quick_print(key, val)

        output_list.append('\n')
        quick_print('##### Suns Voc', '')
        quick_print('Suns-Voc filename', self.sunsVoc.filename)
        for key, val in self.sunsVoc.output.items():
            quick_print(key, val)

        output_list.append('\n')
        quick_print('##### Dark IV', '')
        quick_print('EQE filename', self.div.filename)
        for key, val in self.div.output.items():
            quick_print(key, val)

        output_list.append('\n')
        quick_print('##### Calclated', '')
        quick_print('Rsh', '{:.3e}'.format(self.cell.Rsh))
        quick_print('Rs1', '{:.3e}'.format(self.cell.Rs_1))
        quick_print('Rs2', '{:.3e}'.format(self.cell.Rs_2))
        # TODO: fix this
        # for key, val in self.FF_vals.items():
        #     quick_print(key, '{:.3e}'.format(val))

        self.output_list = output_list

    def print_output_to_file(self):

        output_file = open(self.liv.output['Cell Name ']
                           + '_loss_analysis_summary.csv', 'w')

        for item in self.output_list:
            # print(item)
            output_file.write(item + '\r\n')

        output_file.close()

    def plot_all(self):
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

        self.reflection.plot(ax_refl)
        self.reflection.plot(ax_QE)
        self.qe.plot_EQE(ax_QE)
        self.qe.plot_IQE(ax_QE)

        self.sunsVoc.plot_m(ax_ideality)
        self.sunsVoc.plot_IV(ax_lightIV)
        self.sunsVoc.plot_tau(ax_tau)
        self.liv.plot(ax_lightIV)

        self.div.plot_IV(ax_darkIV)
        self.div.plot_m(ax_ideality)

        self.plot_Basore_fit(ax_QE_fit)
        line_EQE, = self.qe.plot_EQE(ax_QE_layered)
        # line_EQE.set_marker('x')
        self.reflection.plot_QE(ax_QE_layered)

        fig_QE.set_tight_layout(True)
        fig_IV.set_tight_layout(True)

        # fig_QE.savefig(self.liv.output['Cell Name ']
        #             + '_QE.png')
        # fig_IV.savefig(self.liv.output['Cell Name ']
        #             + '_IV.png')
        for i in [fig_QE, fig_IV]:
            i.show()

    def process_all(self):
        '''Call all calculations, plot and print outputs'''

        self.sunsVoc.process()
        self.reflection.process()
        self.qe.process(self.reflection.refl)
        self.cell.Rsh = self.div.process()

        self.cell.Rs_1 = analysis.Rs_calc_1(self.liv.output['Vmp'],
                                            self.liv.output['Jmp'],
                                            self.sunsVoc.V, self.sunsVoc.J)

        self.cell.Rs_2 = analysis.Rs_calc_2(self.liv.output['Voc'],
                                            self.liv.output['Jsc'],
                                            self.liv.output['FF'],
                                            self.sunsVoc.output['PFF'])

        self.cell = self.liv.process()

        vals, self.plot_Basore_fit = analysis.fit_Basore(
            self.qe.wl, self.qe.IQE)

        self.plot_all()

        self.collect_outputs()
        self.print_output_to_file()


class Cell(object):

    def __init__(self, thickness=0.019):
        self.thickness = thickness  # [cm]
        self.Rs_1 = None
        self.Rsh = None
        T = 300   # TODO: make optional input?
        self.Vth = constants.k * T / constants.e

if __name__ == "__main__":
    # pwd = os.getcwd()

    # cell1 = Cell()
    # cell1.load_sunsVoc(pwd, 'example_sunsVoc.dat', text_format=True)
    # cell1.load_lightIV(pwd, 'example_lightIV.lgt')
    # cell1.load_darkIV(pwd, 'example_darkIV.drk')
    # cell1.load_refl(pwd, 'example_reflectance.txt')
    # cell1.load_EQE(pwd, 'example_eqe.txt')
    # cell1.refl_process()
    # cell1.QE_process()
    # vals, plot_Basore = cell1.fit_Basore(cell1.QE_wl, cell1.IQE)
    # # print(cell1.fit_Isenberg(cell1.QE_wl, cell1.IQE))
    # cell1.plot()

    # example_dir = os.path.abspath(os.pardir + '/example_cell/')
    example_dir = os.path.join(os.path.pardir(os.path.dirname(__file__)), '/example_cell/')
    # pwd = os.path.join(pwd, b_dir)
    b = Cell()

    # flags
    choose_files = 0

    if choose_files:
        root = tk.Tk()
        root.withdraw()
        b.load_refl(os.path.join(example_dir, 'example_reflectance.csv'))
        b.load_EQE(os.path.join(example_dir, 'example_EQE.txt'))
        b.load_lightIV(askopenfilename(title='Light IV'))
        b.load_darkIV(askopenfilename(title='Dark IV'))
        b.load_sunsVoc(askopenfilename(title='Suns Voc'))
    else:
        b.load_refl(os.path.join(example_dir, 'example_reflectance.csv'))
        b.load_EQE(os.path.join(example_dir, 'example_EQE.txt'))
        b.load_lightIV(os.path.join(example_dir, 'example_lightIV.lgt'))
        b.load_darkIV(os.path.join(example_dir, 'example_darkIV.drk'))
        b.load_sunsVoc(os.path.join(example_dir, 'example_sunsVoc.xlsm'))

    b.process_all()

    # print(cell1.wl_to_alpha(cell1.refl_wl))
