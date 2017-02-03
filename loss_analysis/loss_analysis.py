# port "loss analysis v5.xlsx" by Ziv Hameiri to python3
# requires python3

'''
NOTES:
- not sure how to best to combine data with different wavelength resolutions,
  currently downsampling to the lowest possible resolution (?)

TODO:
- see in-text TODOs and xxx
- add pie charts or waterfall plots for loss analysis
- move some helper functions to another module?
- change load functions into pure functions returning dictionaries, then use
decorators to load attributes?
- user input for some parameters (eg. wafer thickness)
- create dependency tree for calculations (QE and IV are almost separate)
- robust to having missing data?
- should some methods be converted to in class functions? ie: wl_to_alpha
- return covariance for every fitting parameter?
- easily compare different samples

waterfall charts:
- http://tooblippe.github.io/waterfall/
- http://pbpython.com/waterfall-chart.html

Coding notes:
- Q: Why don't the methods to load data return values instead of setting class
    attributes, doesn't this make this code less reusable?
    - Ans: These methods are only run once per instance and do not have any
    feedback, so they are stable. This code should be self contained, so will
    not be reused elsewhere.
- Q: Why do the process functions return function objects?
    - Ans: To make the functions 'pure functions' and more reusable
'''

import openpyxl
import numpy as np
import os
import re
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter.filedialog import askopenfilename  # remove this?
# modules for this package
import analysis     # requires correct current directory, change? xxx
import constants

# for linear fits
def line(x, m, b):  # b = yintercept
    return m * x + b


class Cell(object):

# loading data ###########################################################

    def __init__(self, thickness = 0.019):
        # cell parameters TODO: update and check
        self.thickness = thickness  # [cm]
        self.input_errors = {}
        self.sample_names = {}
        self.Rs_1 = None
        self.Rsh = None
        # self.AM15G_wl = np.genfromtxt('AM1.5G_spectrum.dat', usecols=(0,),
        #                               skip_header=1)
        # self.AM15G_Jph = np.genfromtxt('AM1.5G_spectrum.dat', usecols=(1,),
        #                                skip_header=1)
        # self.AM15G_Jph_sum = np.sum(self.AM15G_Jph)
        # self.alpha_data = np.genfromtxt('Si_alpha_Green_2008.dat', usecols=(0,1),
                                #    skip_header=1).transpose()
        T = 300   # make optional input?
        self.Vth = constants.k * T / constants.q

    def load_refl(self, raw_data_file):
        '''Loads Reflectance data in cell attributes'''
        self.refl_filepath = raw_data_file
        self.refl_filename = os.path.basename(raw_data_file)

        data_array = np.genfromtxt(raw_data_file, usecols=(0, 1), skip_header=1,
                                   delimiter=',')
        self.refl_wl = data_array[:, 0]
        self.refl = data_array[:, 1]

    def load_EQE(self, raw_data_file):
        '''Loads EQE data into cell attributes'''
        self.EQE_filepath = raw_data_file
        self.EQE_filename = os.path.basename(raw_data_file)

        # the other columns are ignored
        data_array = np.genfromtxt(raw_data_file, usecols=(0, 1),
                                   skip_header=1, skip_footer=8)
        self.QE_wl = data_array[:, 0]
        self.EQE = data_array[:, 1]

        f = open(raw_data_file, 'r')
        d = {}
        for line in f.readlines()[-7:-1]:
            # d.update(dict(re.findall(r'([\s\S]+)\s*:\s\s([^\n]+)', line)))
            d.update(dict([line.strip('\n').split(':')]))     # alternative

        self.QE_output = d

    def load_lightIV(self, raw_data_file):
        '''Loads Light IV data in cell attributes'''
        self.lightIV_filepath = raw_data_file
        self.lightIV_filename = os.path.basename(raw_data_file)

        f = open(raw_data_file, 'r')
        d = OrderedDict()
        # rows which contain floats in lightIV data file header
        float_rows = [2]
        float_rows.extend(list(range(6,18)))
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
        self.lightIV_V = data_array[:, 0]
        self.lightIV_J = data_array[:, 1] / d['Cell Area (sqr cm)']
        # TODO: error check for nans and 1e12?

        self.lightIV_output = d

    def load_darkIV(self, raw_data_file):
        '''Loads Dark IV data in cell attributes'''
        self.darkIV_filepath = raw_data_file
        self.darkIV_filename = os.path.basename(raw_data_file)

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
        self.darkIV_output = d

        data_array = np.genfromtxt(raw_data_file, usecols=(0, 1), skip_header=11)
        self.darkIV_V = data_array[:, 0]
        self.darkIV_J = data_array[:, 1] / d['Cell Area in sqr cm']
        # TODO: error check for nans and 1e12?

    def load_sunsVoc(self, raw_data_file, text_format=False):
        '''Loads Suns Voc data in cell attributes'''
        self.sunsVoc_filepath = raw_data_file
        self.sunsVoc_filename = os.path.basename(raw_data_file)

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

            params = [i.value for i in ws_User['A5':'F5'][0]]
            vals = [i.value for i in ws_User['A6':'F6'][0]]
            self.sunsVoc_params = dict(zip(params, vals))

            params = [i.value for i in ws_User['A8':'L8'][0]]
            # Reduce 13 significant figures in .xlsx file to 6 (default of .format())
            # vals = [float('{:f}'.format(i.value)) for i in ws_User['A6':'F6'][0]]
            vals = [float('{:e}'.format(i.value)) for i in ws_User['A9':'L9'][0]]
            self.sunsVoc_output = dict(zip(params, vals))

        self.sunsVoc_effsuns = data_array[:, 0]     # Effective Suns
        self.sunsVoc_V = data_array[:, 1]
        self.sunsVoc_J = data_array[:, 2]
        self.sunsVoc_P = data_array[:, 3]
        self.sunsVoc_Dn = data_array[:, 4]
        self.sunsVoc_tau_eff = data_array[:, 5]

    def check_input_vals(self):
        '''
        Check the input cell parameters are consistent between measurements.
        Gives the error as a percentage.
        '''
        # TODO: finish
        # check whether data is loaded

        # sample names
        self.sample_names['Light IV'] = self.lightIV_output['Cell Name ']
        self.sample_names['Suns Voc'] = self.sunsVoc_params['Sample Name']
        self.sample_names['Dark IV'] = self.darkIV_output['Cell Name']

        # Cell area
        # tolerance = 1e-3
        liv = self.lightIV_output['Cell Area (sqr cm)']
        div = self.darkIV_output['Cell Area in sqr cm']
        delta = (div - liv) / liv
        self.input_errors['Cell Area'] = delta

        # Voc
        liv = self.lightIV_output['Voc']
        div = self.sunsVoc_output['Voc (V)']
        delta = (div - liv) / liv
        self.input_errors['Voc'] = delta

        # thickness
        user_input_t = self.thickness
        sunsVoc_t = self.sunsVoc_params['Wafer Thickness (cm)']
        delta = (sunsVoc_t - user_input_t) / user_input_t
        self.input_errors['Cell thickness'] = delta

# process data ###############################################################

    def refl_process(self, f_metal = None, wlbounds=(900, 1000)):
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

        self.AM15G_Jph = analysis.AM15G_resample(self.refl_wl)
        i_upper = (self.refl_wl <= 1000)
        self.WAR = (np.dot(self.refl[i_upper], self.AM15G_Jph[i_upper])
                    / np.sum(self.AM15G_Jph[i_upper]))

        if f_metal is None:
            index = (self.refl_wl >= 400) * i_upper
            refl_min = np.amin(self.refl[index])
            self.f_metal = refl_min
        else:
            self.f_metal = f_metal

        index_l = (self.refl_wl >= wlbounds[0])
        index = (self.refl_wl <= wlbounds[1]) * index_l
        popt, pcov = curve_fit(line, self.refl_wl[index], self.refl[index])

        self.refl_wo_escape = np.copy(self.refl)
        self.refl_wo_escape[index_l] = self.refl_wl[index_l] * popt[0] + popt[1]

        Jloss = {}
        Jloss['R'] = np.dot(self.refl, self.AM15G_Jph)
        Jloss['R_wo_escape'] = np.dot(self.refl_wo_escape, self.AM15G_Jph)
        self.Jloss = Jloss

        def plot_refl(ax):
            ax.fill_between(self.refl_wl, self.refl_wo_escape, self.refl,
                            facecolor = 'blue')
            ax.fill_between(self.refl_wl, 0, self.refl_wo_escape,
                            facecolor = 'green')
            ax.set_ylabel('Reflectance [%]')
            ax.grid(True)

        def plot_refl_QE(ax):
            ax.fill_between(self.refl_wl, 100 - self.refl,
                            100 - self.refl_wo_escape)
            ax.plot(self.refl_wl, 100 - self.refl_wo_escape, 'o-',
                    label='w/o escape')
            ax.legend(loc='best')
            # ax.set_ylabel('Reflectance [%]')
            # ax.grid(True)

        return (plot_refl, plot_refl_QE)

    def QE_process(self):
        '''
        Performs several calculations from QE and Reflectance data including:
        - IQE
        - Leff and SRV_rear
        the results are saved into cell attributes
        '''
        # exit early if data isn't loaded, better way?
        # try:
        #     self.refl
        # except AttributeError:
        #     print("Reflection data not loaded")
        #     return
        #
        # try:
        #     self.QE_wl
        # except AttributeError:
        #     print("QE data not loaded")
        #     return

        # TODO: deal with difference in wl sampling
        self.IQE = 100 * self.EQE / (100 - self.refl)

        def plot_EQE(ax):
            ax.plot(self.QE_wl, self.EQE, '-o', label='EQE')
            ax.set_xlabel('Wavelength [$nm$]')
            ax.set_ylabel('QE [%]')
            ax.legend(loc='best')
            ax.grid(True)

        def plot_IQE(ax):
            ax.plot(self.QE_wl, self.IQE, '-o', label='IQE')
            ax.set_xlabel('Wavelength [$nm$]')
            ax.set_ylabel('QE [%]')
            ax.legend(loc='best')
            ax.grid(True)

        return (plot_EQE, plot_IQE)

    def darkIV_process(self):
        '''Dark IV calculations'''

        # Ideality factor
        self.darkIV_m = 1 / self.Vth * self.darkIV_J \
        / (np.gradient(self.darkIV_J) / np.gradient(self.darkIV_V))

        # Shunt resistance, at 30mV
        # TODO: do linear fit with zero intercept?
        self.Rsh = 0.03 / analysis.find_nearest(0.03, self.darkIV_V, self.darkIV_J)

        def plot_darkIV(ax):
            ax.semilogy(self.darkIV_V, self.darkIV_J, '-o', label='data')
            ax.set_xlabel('Voltage [$V$]')
            ax.set_ylabel('Current Density [$A cm^{-2}$]')
            ax.grid(True)
            # ax.legend(loc='best')

        def plot_darkIV_m(ax):
            ax.plot(self.darkIV_V, self.darkIV_m, '-o', label='dark IV')
            ax.set_xlabel('Voltage [$V$]')
            ax.set_ylabel('Ideality Factor []')
            ax.grid(True)
            ax.legend(loc='best')

        return plot_darkIV, plot_darkIV_m
        # vals =  {elem: popt[i] for i, elem in enumerate(fit_params)}
        #
        # return (plot_darkIV, plot_darkIV_m)

    def sunsVoc_process(self):
        '''Suns Voc calculations'''

        # Ideality factor, TODO: better method?
        self.sunsVoc_m = 1 / self.Vth * self.sunsVoc_effsuns \
        / (np.gradient(self.sunsVoc_effsuns) / np.gradient(self.sunsVoc_V))

        def plot_sunsVoc_IV(ax):
            ax.plot(self.sunsVoc_V, self.sunsVoc_J, '-o', label='suns Voc')
            ax.set_xlabel('Voltage [$V$]')
            ax.set_ylabel('Current Density [$A cm^{-2}$]')
            ax.grid(True)
            ax.legend(loc='best')
            ax.set_ylim(ymin=0)

        def plot_sunsVoc_tau(ax):
            # TODO: trims off some noise, use better method?
            ax.loglog(self.sunsVoc_Dn[5:-5], self.sunsVoc_tau_eff[5:-5], '-o',
                    label='Suns Voc')
            ax.set_xlabel('$\Delta n$ [$cm^{-3}$]')
            ax.set_ylabel('Current Density [$A cm^{-2}$]')
            ax.grid(True)
            ax.legend(loc='best')
            # ax.set_xlim(xmin=1e11)

        def plot_sunsVoc_m(ax):
            # trims some noise at ends of array
            ax.plot(self.sunsVoc_V[10:-5], self.sunsVoc_m[10:-5], '-o', label='suns Voc')
            ax.set_xlabel('Voltage [$V$]')
            ax.set_ylabel('Ideality Factor []')
            ax.grid(True)
            ax.legend(loc='best')
            ax.set_ylim(ymin=0)

        return (plot_sunsVoc_IV, plot_sunsVoc_tau, plot_sunsVoc_m)

    def lightIV_process(self):
        '''Light IV calculations'''

        self.Rs_1 = analysis.Rs_calc_1(self.lightIV_output['Vmp'],
                                       self.lightIV_output['Jmp'],
                                       self.sunsVoc_V, self.sunsVoc_J)

        self.Rs_2 = analysis.Rs_calc_2(self.lightIV_output['Voc'],
                                       self.lightIV_output['Jsc'],
                                       self.lightIV_output['FF'],
                                       self.sunsVoc_output['PFF'])

        FFo, FFs, FF = analysis.FF_ideal(self.lightIV_output['Voc'],
                                         Jsc = self.lightIV_output['Jsc'],
                                         Rs = self.Rs_1,
                                         Rsh = self.Rsh)

        self.FF_vals = {}
        self.FF_vals['FFo'] = FFo
        self.FF_vals['FFs'] = FFs
        self.FF_vals['FF'] = FF

        def plot_lightIV(ax):
            ax.plot(self.lightIV_V, self.lightIV_J, '-o', label='light IV')
            ax.set_xlabel('Voltage [$V$]')
            ax.set_ylabel('Current Density [$A cm^{-2}$]')
            ax.grid(True)
            # ax.legend(loc='best')

        return plot_lightIV

# print and plot #############################################################

    def collect_outputs(self):
        '''Temp hack'''
        # TODO: improve this

        output_list = []
        def quick_print(key, val):
            output_list.append('{:>30}, {:>20}'.format(key, val))

        output_list.append('\n')
        quick_print('##### Sample names','')
        for key, val in self.sample_names.items():
            quick_print(key, '{:s}'.format(val))

        output_list.append('\n')
        quick_print('##### Input errors','')
        for key, val in self.input_errors.items():
            quick_print(key, '{:.3e}'.format(val))

        output_list.append('\n')
        quick_print('##### Reflectance','')
        quick_print('Reflectance filename', self.refl_filename)
        quick_print('WAR', '{:.3e}'.format(self.WAR))
        quick_print('f_metal', '{:.3e}'.format(self.f_metal))
        for key, val in self.Jloss.items():
            quick_print(key, '{:.3e}'.format(val))

        output_list.append('\n')
        quick_print('##### QE','')
        quick_print('EQE filename', self.EQE_filename)
        for key, val in self.QE_output.items():
            quick_print(key, val)

        output_list.append('\n')
        quick_print('##### Light IV','')
        quick_print('EQE filename', self.lightIV_filename)
        for key, val in self.lightIV_output.items():
            quick_print(key, val)

        output_list.append('\n')
        quick_print('##### Suns Voc','')
        quick_print('Suns-Voc filename', self.sunsVoc_filename)
        for key, val in self.sunsVoc_output.items():
            quick_print(key, val)

        output_list.append('\n')
        quick_print('##### Dark IV','')
        quick_print('EQE filename', self.darkIV_filename)
        for key, val in self.darkIV_output.items():
            quick_print(key, val)

        output_list.append('\n')
        quick_print('##### Calclated','')
        quick_print('Rsh', '{:.3e}'.format(self.Rsh))
        quick_print('Rs1', '{:.3e}'.format(self.Rs_1))
        quick_print('Rs2', '{:.3e}'.format(self.Rs_2))
        # TODO: fix this
        # for key, val in self.FF_vals.items():
        #     quick_print(key, '{:.3e}'.format(val))

        self.output_list = output_list

    def print_output_to_file(self):

        output_file = open(self.lightIV_output['Cell Name ']
                           + '_loss_analysis_summary.csv','w')

        for item in self.output_list:
            print(item)
            output_file.write(item + '\r\n')

        output_file.close()

    def plot_all(self):
        '''Plot the output of previous calculations'''
        # for reflectance
        fig_QE = plt.figure(1, figsize=(30/2.54, 15/2.54))
        fig_QE.clf()
        # for light and dark IV
        fig_IV = plt.figure(2, figsize=(30/2.54, 15/2.54))
        fig_IV.clf()

        ax_refl = fig_QE.add_subplot(2, 2, 1)
        ax_QE = fig_QE.add_subplot(2, 2, 2)
        ax_QE_fit = fig_QE.add_subplot(2, 2, 3)
        ax_QE_layered = fig_QE.add_subplot(2, 2, 4)

        ax_darkIV = fig_IV.add_subplot(2, 2, 1)
        ax_ideality = fig_IV.add_subplot(2, 2, 3)
        ax_lightIV = fig_IV.add_subplot(2, 2, 2)
        ax_tau = fig_IV.add_subplot(2, 2, 4)

        self.plot_refl(ax_refl)
        self.plot_refl(ax_QE)
        self.plot_EQE(ax_QE)
        self.plot_IQE(ax_QE)
        self.plot_Basore_fit(ax_QE_fit)
        self.plot_EQE(ax_QE_layered)
        self.plot_refl_QE(ax_QE_layered)

        self.plot_darkIV(ax_darkIV)
        self.plot_darkIV_m(ax_ideality)
        self.plot_sunsVoc_m(ax_ideality)
        self.plot_lightIV(ax_lightIV)
        self.plot_sunsVoc_IV(ax_lightIV)
        self.plot_sunsVoc_tau(ax_tau)

        fig_QE.set_tight_layout(True)
        fig_IV.set_tight_layout(True)

        # fig_QE.savefig(self.lightIV_output['Cell Name ']
        #             + '_QE.png')
        # fig_IV.savefig(self.lightIV_output['Cell Name ']
        #             + '_IV.png')
        fig_QE.show()
        fig_IV.show()

    def process_all(self):
        '''Call all calculations, plot and print outputs'''

        # most of these methods return function objects to plot default output
        self.check_input_vals()
        self.plot_sunsVoc_IV, self.plot_sunsVoc_tau, self.plot_sunsVoc_m = self.sunsVoc_process()
        self.plot_refl, self.plot_refl_QE = self.refl_process()
        self.plot_EQE, self.plot_IQE = self.QE_process()
        self.plot_lightIV = self.lightIV_process()
        self.plot_darkIV, self.plot_darkIV_m = self.darkIV_process()
        vals, self.plot_Basore_fit = analysis.fit_Basore(self.QE_wl, self.IQE)
        self.plot_all()
        self.collect_outputs()
        self.print_output_to_file()


if __name__ == "__main__":
    # pwd = '/home/ned/Dropbox/unsw/python_scripts/loss analysis/'
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

    example_dir = os.path.abspath(os.pardir + '/example_cell/')
    # pwd = os.path.join(pwd, b_dir)
    b = Cell()

    # flags
    choose_files = 0

    if choose_files:
        root = tk.Tk()
        root.withdraw()
        b.load_refl(os.path.join(example_dir, 'example_reflectance.csv'))
        b.load_EQE(os.path.join(example_dir, 'example_EQE.txt'))
        b.load_lightIV(askopenfilename(title = 'Light IV'))
        b.load_darkIV(askopenfilename(title = 'Dark IV'))
        b.load_sunsVoc(askopenfilename(title = 'Suns Voc'))
    else:
        b.load_refl(os.path.join(example_dir, 'example_reflectance.csv'))
        b.load_EQE(os.path.join(example_dir, 'example_EQE.txt'))
        b.load_lightIV(os.path.join(example_dir, 'example_lightIV.lgt'))
        b.load_darkIV(os.path.join(example_dir, 'example_darkIV.drk'))
        b.load_sunsVoc(os.path.join(example_dir, 'example_sunsVoc.xlsm'))

    b.process_all()

    # print(cell1.wl_to_alpha(cell1.refl_wl))
