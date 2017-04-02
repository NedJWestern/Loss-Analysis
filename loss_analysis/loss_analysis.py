# port "loss analysis v5.xlsx" by Ziv Hameiri to python3

import openpyxl
import numpy as np
import sys
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


def waterfall(ax, y, xlabels=None):
    '''
    Create a waterfall plot.
    Assumes the first value is the starting point,
    all other values are set to negative creating a 'waterfall' downwards.
    '''
    y = abs(np.array(y))
    y[1:] = -1 * y[1:]
    x = np.arange(len(y))
    y_bot = np.append(0, y[:-1].cumsum())
    ax.bar(x, y, bottom=y_bot, align='center')
    ax.set_ylim(ymin = y_bot[-1] + y[-1])
    if xlabels is not None:
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=40, ha='right')

    return ax


class Refl(object):

    def __init__(self, fname):
        self.load(fname)

    def process(self, f_metal=None, wlbounds=(900, 1000), wljunc=600):
        '''
        Performs several calculations including:
        - Average Reflection (AR)
        - Weighted Average Reflection (WAR)
        - Light lost from front surface escape
        the results are loaded into attributes
        '''

        # xxx need upper bound for this?
        self.AR = np.trapz(self.refl / 100, x=self.wl)

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

        # defined as area between 100% and the given curve, to simplify calculations
        Jloss = OrderedDict()
        Jloss['max_limit'] = np.sum(self.AM15G_Jph)
        Jloss['metal_shading'] = np.dot(self.f_metal / 100 \
                                        * np.ones(len(self.AM15G_Jph)),
                                        self.AM15G_Jph)
        Jloss['refl_wo_escape'] = np.dot(self.refl_wo_escape / 100 \
                                         , self.AM15G_Jph) \
                                         - Jloss['metal_shading']
        Jloss['front_escape'] = np.dot(self.refl / 100, self.AM15G_Jph) \
                                    - Jloss['metal_shading'] \
                                    - Jloss['refl_wo_escape']
        # this makes qe Jloss calculations easier
        idx_junc = analysis.find_nearest(wljunc, self.wl)
        Jloss['front_escape_blue'] = np.dot(self.refl[:idx_junc] / 100,
                                            self.AM15G_Jph[:idx_junc])
        Jloss['front_escape_red'] = np.dot(self.refl[idx_junc:] / 100,
                                            self.AM15G_Jph[idx_junc:])
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

    def process(self, wl, refl, refl_wo_escape, Jloss, wljunc=600):
        '''
        Performs several calculations from QE and Reflectance data including:
        - IQE
        - Leff and SRV_rear
        - Current loss from each region of the device
        the results are saved into attributes
        '''
        self.IQE = self.EQE / (100 - refl)

        self.output_Basore_fit, self.plot_Basore_fit = analysis.fit_Basore(
            self.wl, self.IQE)

        EQE_on_eta_c = self.EQE / self.output_Basore_fit['eta_c']
        idx = analysis.find_nearest(750, wl)
        total_min = np.minimum((100 - refl_wo_escape), EQE_on_eta_c)
        self.EQE_xxx_unnamed = np.append(100 - refl_wo_escape[:idx],
                                         total_min[idx:])

        AM15G_Jph = analysis.AM15G_resample(self.wl)
        Jloss_qe = Jloss.copy()
        del Jloss_qe['front_escape_red']
        del Jloss_qe['front_escape_blue']
        idx_junc = analysis.find_nearest(wljunc, self.wl)
        Jloss_qe['parasitic_absorption'] = np.dot((100 - self.EQE_xxx_unnamed[idx_junc:]) / 100,
                                           AM15G_Jph[idx_junc:]) \
                                           - Jloss['front_escape_red']
        Jloss_qe['bulk_recomm'] = np.dot((100 - self.EQE[idx_junc:]) / 100,
                                         AM15G_Jph[idx_junc:]) \
                                  - Jloss['front_escape_red'] \
                                  - Jloss_qe['parasitic_absorption']
        Jloss_qe['blue_loss'] = np.dot((100 - self.EQE[:idx_junc]) / 100,
                                       AM15G_Jph[:idx_junc]) \
                                       - Jloss['front_escape_blue']

        self.Jloss_qe = Jloss_qe
        # print(Jloss_qe)

    def plot_EQE(self, ax):

        line_EQE = ax.plot(self.wl, self.EQE, '-o', label='EQE')
        ax.set_xlabel('Wavelength [$nm$]')
        ax.set_ylabel('QE [%]')
        ax.legend(loc='best')
        ax.grid(True)
        return line_EQE     # xxx currently not working

    def plot_IQE(self, ax):
        ax.plot(self.wl, self.IQE, '-o', label='IQE')
        ax.set_xlabel('Wavelength [$nm$]')
        ax.set_ylabel('QE [%]')
        ax.legend(loc='best')
        ax.grid(True)

    def plot_Jloss(self, ax):
        waterfall(ax, list(self.Jloss_qe.values()), list(self.Jloss_qe.keys()))

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
        '''
        Light IV calculations

        caculates the ideal fill factors:

        FF0
        FFs
        FF

        The the loss from the current
        FF_Rsh
        FF_Rsh
        FF_other

        These are all stored within two dictionaries.

        Inputs:
            Rsh: The shunt resistance
            Rs: The series resistance

        Outputs:
            None
        '''

        self.m = analysis.ideality_factor(
            self.V, -1 * (self.J - self.output['Jsc']), Vth)

        ideal_FF = OrderedDict()
        ideal_FF['FF_0'] = analysis.ideal_FF(self.output['Voc'])
        ideal_FF['FF_s'] = analysis.ideal_FF_series(self.output['Voc'],
                                                    self.output['Jsc'],
                                                    Rs)
        ideal_FF['FF_s_sh'] = analysis.ideal_FF_series_shunt(self.output['Voc'],
                                                             self.output['Jsc'],
                                                             Rs, Rsh)
        self.ideal_FF = ideal_FF

        FF_loss = OrderedDict()
        FF_loss['FF_0'] = analysis.ideal_FF(self.output['Voc'])
        FF_loss['FF_Rs'] = analysis.FF_loss_series(self.output['Voc'],
                                                        self.output['Jsc'],
                                                        self.output['Jmp'],
                                                        Rs)
        FF_loss['FF_Rsh'] = analysis.FF_loss_shunt(self.output['Voc'],
                                                        self.output['Jsc'],
                                                        self.output['Vmp'],
                                                        self.output['Jmp'],
                                                        Rs, Rsh)

        # for waterfall plot
        FF_loss['FF_other'] = (FF_loss['FF_0'] \
                                      - self.output['FF'] \
                                      - FF_loss['FF_Rs'] \
                                      - FF_loss['FF_Rsh'])

        self.FF_loss = FF_loss

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
        # ax.legend(loc='best')

    def plot_m(self, ax):
        # trims some noise at ends of array
        ax.plot(self.V[10:-5], self.m[10:-5], '-o', label='Light IV')
        ax.set_xlabel('Voltage [$V$]')
        ax.set_ylabel('Ideality Factor []')
        ax.grid(True)
        ax.legend(loc='best')
        ax.set_ylim(ymin=0)

    def plot_FF1(self, ax):
        waterfall(ax, list(self.FF_loss.values()), list(self.FF_loss.keys()))

    def load(self, raw_data_file):
        '''Loads Light IV data into attributes'''
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

        self.m = analysis.ideality_factor(self.V, self.effsuns, Vth)

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

    def plot_log_IV(self, ax):
        # trims some noise at ends of array
        # TODO: Link this to Jsc rather than this manual index

        # check for real values
        index = np.isfinite(self.J)
        # find the meaured Jsc
        Jsc_index = abs(self.V[index]) == np.min(abs(self.V[index]))

        ax.plot(self.V, -1 * (
            self.J - self.J[index][Jsc_index]), '-o', label='Suns Voc')
        ax.set_xlabel('Voltage [$V$]')
        ax.set_ylabel('Ideality Factor []')
        ax.grid(True)
        ax.legend(loc='best')

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
            # vals = [float('{:f}'.format(i.value)) for i in
            # ws_User['A6':'F6'][0]]
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
        '''
        This performs the Dark IV calculations for loss analysis

        It currently caculates:

        the idealify factor as a function of voltage

        '''

        # Ideality factor
        self.m = analysis.ideality_factor(self.V, self.J, Vth)

        # Shunt resistance, at 30mV
        # TODO: do linear fit with zero intercept?
        Rsh = 0.03 / analysis.find_nearest(0.03, self.V, self.J)

        return Rsh

    def plot_log_IV(self, ax):
        ax.semilogy(self.V, self.J, '-o', label='Dark IV')
        ax.set_xlabel('Voltage [$V$]')
        ax.set_ylabel('Current Density [$A cm^{-2}$]')
        ax.grid(True)
        ax.legend(loc='best')

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

    def __init__(self, thickness=None, **kwargs):
        self.thickness = thickness  # [cm]
        self.sample_names = {}
        self.input_errors = {}
        self.refl = Refl(kwargs['reflectance_fname'])
        self.qe = QE(kwargs['EQE_fname'])
        self.sunsVoc = IVSuns(kwargs['suns Voc_fname'])
        self.div = IVDark(kwargs['dark IV_fname'])
        self.liv = IVLight(kwargs['light IV_fname'])

        self.example_dir = os.path.join(os.pardir, 'example_cell')

        self.check_input_vals()

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
        area_liv = self.liv.output['Cell Area (sqr cm)']
        area_div = self.div.output['Cell Area in sqr cm']
        delta = (area_div - area_liv) / area_liv
        self.input_errors['Cell Area'] = delta

        # thickness
        self.thickness = self.sunsVoc.params['Wafer Thickness (cm)']
        tck_user_input = self.thickness
        tck_sunsVoc = self.sunsVoc.params['Wafer Thickness (cm)']
        delta = (tck_sunsVoc - tck_user_input) / tck_user_input
        self.input_errors['Cell thickness'] = delta

        # Voc
        Voc_liv = self.liv.output['Voc']
        Voc_div = self.sunsVoc.output['Voc (V)']
        delta = (Voc_div - Voc_liv) / Voc_liv
        self.input_errors['Voc'] = delta

        # Jsc
        Jsc_liv = self.liv.output['Jsc']
        Jsc_iqe = self.qe.output['Jsc']
        delta = (Jsc_iqe - Jsc_liv) / Jsc_liv
        self.input_errors['Jsc'] = delta

        # some checks on the data
        assert abs(self.input_errors['Cell Area']
                   ) < 0.01, "Provided sample area's disagrees: {0:.1f} cm^2 {1:.1f} cm^2".format(area_liv, area_div)
        assert abs(self.input_errors['Cell thickness']
                   ) < 0.01, "Provided sample thickness disagrees: {0:.4f} cm {1:.4f} cm".format(tck_user_input, tck_sunsVoc)
        assert abs(self.input_errors['Voc']
                   ) < 0.01, "Provided Voc disagree: {0:.0f} mV {1:.0f} mV".format(Voc_liv * 1000, Voc_div * 1000)
        assert abs(self.input_errors['Jsc']
                   ) < 0.1, "Provided Jsc disagree: {0:.0f} mA {1:.0f} mA".format(Jsc_liv * 1000, Jsc_iqe * 1000)

    def collect_outputs(self):
        '''Collects input and output parameters into self.output_list'''

        output_list = []

        def quick_print(key, val):
            output_list.append('{:>30}, {:<20}'.format(key, val))
        output_list.append('\n')

        quick_print('##### Inputs check: Percentage difference', '',)

        for key, val in self.sample_names.items():
            quick_print(key, val)
        for key, val in self.input_errors.items():
            quick_print(key, '{:.3e}%'.format(val * 100))
        output_list.append('\n')

        quick_print('##### Reflectance', '')
        quick_print('filename', self.refl.filename)
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
        quick_print('### Reflectance', '')
        quick_print('AR', '{:.3f}'.format(self.refl.AR))
        quick_print('WAR', '{:.3f}'.format(self.refl.WAR))
        quick_print('f_metal', '{:.3f}'.format(self.refl.f_metal))
        quick_print('### Parasitic resistances', '')
        quick_print('Rsh (Ohm cm2)', '{:.3e}'.format(self.Rsh))
        quick_print('Rs1 (Ohm cm2)', '{:.3e}'.format(self.Rs_1))
        quick_print('Rs2 (Ohm cm2)', '{:.3e}'.format(self.Rs_2))

        quick_print('### Current losses', '')
        for key, val in self.qe.Jloss_qe.items():
            quick_print(key + ' (mA)', '{:.3f}'.format(val))

        quick_print('### Fill Factor', '')
        for key, val in self.liv.ideal_FF.items():
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

        ax_refl = fig_QE.add_subplot(2, 2, 1)
        ax_QE = fig_QE.add_subplot(2, 2, 2)
        ax_QE_fit = fig_QE.add_subplot(2, 2, 3)
        ax_QE_layered = fig_QE.add_subplot(2, 2, 4)

        self.refl.plot(ax_refl)
        self.refl.plot(ax_QE)
        self.qe.plot_EQE(ax_QE)
        self.qe.plot_IQE(ax_QE)

        # for light and dark IV
        fig_IV = plt.figure('IV', figsize=(30 / 2.54, 15 / 2.54))
        fig_IV.clf()

        # get the plotting axes
        ax_logIV = fig_IV.add_subplot(2, 2, 1)
        ax_ideality = fig_IV.add_subplot(2, 2, 3)
        ax_lightIV = fig_IV.add_subplot(2, 2, 2)
        ax_tau = fig_IV.add_subplot(2, 2, 4)

        # plot light IV first, as is typically the noisest
        self.liv.plot_m(ax_ideality)
        self.liv.plot(ax_lightIV)

        # plot suns Voc
        self.sunsVoc.plot_m(ax_ideality)
        self.sunsVoc.plot_IV(ax_lightIV)
        self.sunsVoc.plot_tau(ax_tau)
        self.sunsVoc.plot_log_IV(ax_logIV)

        # plot dark IV as least noisest
        self.div.plot_log_IV(ax_logIV)
        self.div.plot_m(ax_ideality)

        # plot the EQE fitted data
        self.qe.plot_Basore_fit(ax_QE_fit)

        # this is doing some loss analysis filling
        dummy_ones = np.ones(len(self.refl.wl))
        ax_QE_layered.fill_between(self.refl.wl, dummy_ones * 100,
                                   100 - dummy_ones * self.refl.f_metal,  color='blue')
        ax_QE_layered.fill_between(self.refl.wl,
                                   100 - dummy_ones * self.refl.f_metal,
                                   100 - self.refl.refl_wo_escape, color='green')
        ax_QE_layered.fill_between(self.refl.wl, 100 - self.refl.refl_wo_escape,
                                   100 - self.refl.refl, color='red')
        ax_QE_layered.fill_between(self.refl.wl, 100 - self.refl.refl,
                                   self.qe.EQE_xxx_unnamed, color='cyan')
        # ax_QE_layered.plot(self.refl.wl, self.qe.EQE_xxx_unnamed)
        ax_QE_layered.fill_between(self.refl.wl, self.qe.EQE_xxx_unnamed,
                                   self.qe.EQE, color='magenta')
        # line_EQE, = self.qe.plot_EQE(ax_QE_layered)
        # line_EQE.set_marker('x')
        # self.refl.plot_QE(ax_QE_layered)

        # for loss analysis summary
        fig_LA = plt.figure('LA', figsize=(30 / 2.54, 15 / 2.54))
        fig_LA.clf()

        ax_FF = fig_LA.add_subplot(2, 2, 1)
        ax_Jloss = fig_LA.add_subplot(2, 2, 2)

        self.liv.plot_FF1(ax_FF)
        self.qe.plot_Jloss(ax_Jloss)

        fig_QE.set_tight_layout(True)
        fig_IV.set_tight_layout(True)

        if save_fig_bool:

            fig_QE.savefig(os.path.join(self.output_dir,
                                        self.cell_name + '_QE.png'))
            fig_IV.savefig(os.path.join(self.output_dir,
                                        self.cell_name + '_IV.png'))

        plt.show()

    def process_all(self, save_fig_bool, output_dir, cell_name):
        '''
        A function that calls all the processing functions.
        '''

        if cell_name == '':
            self.cell_name = self.liv.output['Cell Name ']
        else:
            self.cell_name = cell_name

        self.output_dir = output_dir

        self.sunsVoc.process()
        self.refl.process()
        self.qe.process(self.refl.wl, self.refl.refl, self.refl.refl_wo_escape,
                        self.refl.Jloss)
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
        'dark IV_fname': os.path.join(example_dir, 'example_darkIV.drk')}

    cell1 = Cell(**files)
    cell1.process_all()
