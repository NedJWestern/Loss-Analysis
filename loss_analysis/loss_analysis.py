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
# from scipy.optimize import curve_fit
import tkinter as tk
from tkinter.filedialog import askopenfilename  # remove this?
# modules for this package
import analysis     # requires correct current directory, change? xxx
from scipy import constants


class IV_suns():
    def __init__(self, cell):
        self.cell = cell
        self.filepath = None
        self.filename = None
        self.output = None
        self.V = None
        self.J = None
        self.effsuns = None
        self.params = None

    def process(self):
        '''Suns Voc calculations'''

        # Ideality factor, TODO: better method?
        self.m = 1 / self.cell.Vth * self.effsuns \
        / (np.gradient(self.effsuns) / np.gradient(self.V))

    def plot_IV(self,ax):
        ax.plot(self.V, self.J, '-o', label='suns Voc')
        ax.set_xlabel('Voltage [$V$]')
        ax.set_ylabel('Current Density [$A cm^{-2}$]')
        ax.grid(True)
        ax.legend(loc='best')
        ax.set_ylim(ymin=0)

    def plot_tau(self,ax):
        # TODO: trims off some noise, use better method?
        ax.loglog(self.Dn[5:-5], self.tau_eff[5:-5], '-o',
                label='Suns Voc')
        ax.set_xlabel('$\Delta n$ [$cm^{-3}$]')
        ax.set_ylabel('Current Density [$A cm^{-2}$]')
        ax.grid(True)
        ax.legend(loc='best')
        # ax.set_xlim(xmin=1e11)

    def plot_m(self,ax):
        # trims some noise at ends of array
        ax.plot(self.V[10:-5], self.m[10:-5], '-o', label='suns Voc')
        ax.set_xlabel('Voltage [$V$]')
        ax.set_ylabel('Ideality Factor []')
        ax.grid(True)
        ax.legend(loc='best')
        ax.set_ylim(ymin=0)



    def load(self, raw_data_file, text_format=False):
        '''Loads Suns Voc data in cell attributes'''
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

            params = [i.value for i in ws_User['A5':'F5'][0]]
            vals = [i.value for i in ws_User['A6':'F6'][0]]
            self.params = dict(zip(params, vals))

            params = [i.value for i in ws_User['A8':'L8'][0]]
            # Reduce 13 significant figures in .xlsx file to 6 (default of .format())
            # vals = [float('{:f}'.format(i.value)) for i in ws_User['A6':'F6'][0]]
            vals = [float('{:e}'.format(i.value)) for i in ws_User['A9':'L9'][0]]
            self.output = dict(zip(params, vals))

        self.effsuns = data_array[:, 0]     # Effective Suns
        self.V = data_array[:, 1]
        self.J = data_array[:, 2]
        self.P = data_array[:, 3]
        self.Dn = data_array[:, 4]
        self.tau_eff = data_array[:, 5]

class IV_light():

    def __init__(self, cell):
        self.cell = cell
        self.filepath = None
        self.filename = None
        self.output = None
        self.V = None
        self.J = None

    def process(self, SunsVoc_V, SunsVoc_J, SunsVoc_PFF, Rsh):
        '''Light IV calculations'''

        self.Rs_1 = analysis.Rs_calc_1(self.output['Vmp'],
                                       self.output['Jmp'],
                                       SunsVoc_V, SunsVoc_J)

        self.Rs_2 = analysis.Rs_calc_2(self.output['Voc'],
                                       self.output['Jsc'],
                                       self.output['FF'],
                                       SunsVoc_PFF)

        FFo, FFs, FF = analysis.FF_ideal(self.output['Voc'],
                                         Jsc = self.output['Jsc'],
                                         Rs = self.Rs_1,
                                         Rsh = Rsh)

        self.FF_vals = {}
        self.FF_vals['FFo'] = FFo
        self.FF_vals['FFs'] = FFs
        self.FF_vals['FF'] = FF

    def plot(self,ax):
        ax.plot(self.V, self.J, '-o', label='light IV')
        ax.set_xlabel('Voltage [$V$]')
        ax.set_ylabel('Current Density [$A cm^{-2}$]')
        ax.grid(True)
        # ax.legend(loc='best')



    def load(self, raw_data_file):
        '''Loads Light IV data in cell attributes'''
        self.filepath = raw_data_file
        self.filename = os.path.basename(raw_data_file)

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
        self.V = data_array[:, 0]
        self.J = data_array[:, 1] / d['Cell Area (sqr cm)']
        # TODO: error check for nans and 1e12?

        self.output = d

class IV_dark():

    def __init__(self, cell):
        self.cell = cell
        self.filepath = None
        self.filename = None
        self.output = None
        self.V = None
        self.J = None
        self.m = None

    def process(self):
        '''Dark IV calculations'''

        # Ideality factor
        self.m = 1 / self.cell.Vth * self.J \
        / (np.gradient(self.J) / np.gradient(self.V))

        # Shunt resistance, at 30mV
        # TODO: do linear fit with zero intercept?
        self.Rsh = 0.03 / analysis.find_nearest(0.03, self.V, self.J)

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
        '''Loads Dark IV data in cell attributes'''
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

        data_array = np.genfromtxt(raw_data_file, usecols=(0, 1), skip_header=11)
        self.V = data_array[:, 0]
        self.J = data_array[:, 1] / d['Cell Area in sqr cm']
        # TODO: error check for nans and 1e12?

class Reflection():

    def __init__(self):
        self.wl = None
        self.refl = None
        self.refl_wo_escape = None
        self.Jloss = None

        self.filepath = None
        self.filename = None

    def process(self, f_metal = None, wlbounds=(900, 1000)):
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
        '''Loads Reflectance data in cell attributes'''
        self.filepath = raw_data_file
        self.filename = os.path.basename(raw_data_file)

        data_array = np.genfromtxt(raw_data_file, usecols=(0, 1), skip_header=1,
                                   delimiter=',')
        self.wl = data_array[:, 0]
        self.refl = data_array[:, 1]



class QE():

    def __init__(self):
        self.EQE = None
        self.IQE = None
        self.wl = None
        self.output = None

        self.filepath = None
        self.filename = None

    def process(self, refl):
        '''
        Performs several calculations from QE and Reflectance data including:
        - IQE
        - Leff and SRV_rear
        the results are saved into cell attributes
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
        '''Loads EQE data into cell attributes'''
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
            # d.update(dict(re.findall(r'([\s\S]+)\s*:\s\s([^\n]+)', line)))
            d.update(dict([line.strip('\n').split(':')]))     # alternative

        self.output = d

class loss_analysis_handeller():

    def __init__(self, **args):
        # containts t he information about the cell
        # this should get around.
        self.cell = Cell()
        self.reflection  = Reflection()
        self.qe = QE()
        self.sunvoc = IV_suns(self.cell)
        self.liv = IV_light(self.cell)
        self.div  = IV_dark(self.cell)



# print and plot #############################################################

    def collect_outputs(self):
        '''Temp hack'''
        # TODO: improve this

        output_list = []
        def quick_print(key, val):
            output_list.append('{:>30}, {:>20}'.format(key, val))

        output_list.append('\n')
        quick_print('##### Sample names',')1', '{:.3e}'.format(self.Rs_1))
        quick_print('Rs2', '{:.3e}'.format(self.Rs_2))
        # TODO: fix thi
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


        self.reflection.plot(ax_refl)
        self.reflection.plot(ax_QE)
        self.qe.plot_EQE(ax_QE)
        self.qe.plot_IQE(ax_QE)

        self.sunvoc.plot_m(ax_ideality)
        self.sunvoc.plot_IV(ax_lightIV)
        self.sunvoc.plot_tau(ax_tau)
        self.liv.plot(ax_lightIV)

        self.div.plot_IV(ax_darkIV)
        self.div.plot_m(ax_darkIV)


        # self.plot_Basore_fit(ax_QE_fit)
        # line_EQE, = self.plot_EQE(ax_QE_layered)
        # line_EQE.set_marker('v')
        # self.plot_refl_QE(ax_QE_layered)


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
        # self.check_input_vals()


        self.sunvoc.process()
        self.reflection.process()
        self.qe.process(self.reflection.refl)
        self.div.process()
        self.liv.process(self.sunvoc.V, self.sunvoc.J, self.sunvoc.output['PFF'], self.div.Rsh)

        # vals, self.plot_Basore_fit = analysis.fit_Basore(self.QE_wl, self.IQE)
        self.plot_all()

        # self.collect_outputs()
        # self.print_output_to_file()


class Cell(object):
# this seems very involved, maybe split up into component class for the different types
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
        self.Vth = constants.k * T / constants.e



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
