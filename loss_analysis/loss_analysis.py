# port "loss analysis v5.xlsx" by Ziv Hameiri to python3

import openpyxl
import numpy as np
import sys
import os
import re
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import rcParams
import itertools
import warnings
# modules for this package
import analysis
from scipy import constants
from data_loaders import QE, Refl, IVSuns, IVDark, IVLight

T = 300   # TODO: make optional input?
Vth = constants.k * T / constants.e


def waterfall(ax, y, xlabels=None):
    '''
    Create a waterfall plot. Assumes the first value is positive, all other
    values are negative creating a 'waterfall' downwards.
    '''
    y = np.array(y)
    x = np.arange(len(y))
    y_bot = np.append(0, y[:-1].cumsum())
    ax.bar(x, y, bottom=y_bot, align='center')

    # get rid of underscores from labels
    xlabels = [i.replace('_', ' ') for i in xlabels]

    # ax.set_ylim(ymin = y_bot[-1] + y[-1])
    if xlabels is not None:
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=40, ha='right')

    return ax


class recombination_losses():

    # required inputs
    Voc = None
    Jsc = None
    Vmp = None
    Jmp = None
    Rs = None
    Rsh = None

    # calculated values
    ideal_FF = None
    FF_loss = None

    def __init__(self, Voc, Jsc, Vmp, Jmp, Rs, Rsh, FF):
        self.Voc = Voc
        self.Jsc = Jsc
        self.Rs = Rs
        self.Rsh = Rsh
        self.Vmp = Vmp
        self.Jmp = Jmp
        self.FF = FF

    def calculate(self):

        ideal_FF = OrderedDict()
        ideal_FF['FF_0'] = analysis.ideal_FF(self.Voc)
        ideal_FF['FF_s'] = analysis.ideal_FF_series(self.Voc,
                                                    self.Jsc,
                                                    self.Rs)
        ideal_FF['FF_s_sh'] = analysis.ideal_FF_series_shunt(self.Voc, self.Jsc,
                                                             self.Rs, self.Rsh)
        self.ideal_FF = ideal_FF

        assert ideal_FF['FF_0'] > self.FF

        FF_loss = OrderedDict()

        FF_loss['Achieved'] = self.FF
        FF_loss['FF_Rs'] = - analysis.FF_loss_series(self.Voc,
                                                     self.Jsc,
                                                     self.Jmp,
                                                     self.Rs)

        FF_loss['FF_Rsh'] = - analysis.FF_loss_shunt(self.Voc,
                                                     self.Jsc,
                                                     self.Vmp,
                                                     self.Jmp,
                                                     self.Rs,
                                                     self.Rsh)

        # for waterfall plot
        FF_loss['FF_other'] = (ideal_FF['FF_0']
                               - self.FF
                               - FF_loss['FF_Rs']
                               - FF_loss['FF_Rsh'])

        self.FF_loss = FF_loss
        # print(FF_loss['FF_0'], self.FF)

    def plot_FF_loss(self, ax):
        waterfall(ax, list(self.FF_loss.values()), list(self.FF_loss.keys()))
        ax.set_ylim(bottom=0.9 * self.FF)
        ax.set_ylabel('Current (mA/cm$^{-2}$)')


class optical_losses():

    # raw data
    wl = None
    EQE = None
    refl = None
    IQE = None

    refl_front = None
    refl_metal = None
    Jph = None

    # results

    def __init__(self, wavelength, EQE, reflection, refl_front, refl_metal, **kwargs):
        self.wl = wavelength
        self.EQE = EQE
        self.refl = reflection
        self.refl_front = refl_front
        self.refl_metal = refl_metal

        # set attributes if passed
        for key in kwargs.keys():
            if hasattr(self, key):
                setattr(self, key, kwargs[key])

        if self.IQE is None:
            self.IQE = self.EQE / (100 - self.refl) * 100

    def calculate_losses(self):

        self.output_Basore_fit, self.plot_Basore_fit = analysis.fit_Basore(
            self.wl, self.IQE)

        EQE_on_eta_c = self.EQE / self.output_Basore_fit['eta_c'] * 100

        EQE_on_eta_c = self.EQE / self.output_Basore_fit['eta_c'] * 100

        total_min = np.minimum((100 - self.refl_front), EQE_on_eta_c)

        idx = analysis.find_nearest(750, self.wl)
        self.EQE_xxx_unnamed = np.append(100 - self.refl_front[:idx],
                                         total_min[idx:])

    def _EQE_breakdown(self):

        EQE_bd = OrderedDict()

        EQE_bd['metal_reflection'] = self.refl_metal * \
            np.ones(self.wl.shape[0])
        EQE_bd['front_reflection'] = (
            self.refl_front - EQE_bd['metal_reflection'])

        EQE_bd['front_escape'] = self.refl - self.refl_front

        # ensure no values less than 0
        EQE_bd['front_escape'][EQE_bd['front_escape'] < 0] = 0
        # assert np.all(EQE_bd['front_escape'] >= 0)

        val = 100 - self.EQE
        for key in EQE_bd.keys():
            val -= EQE_bd[key]
        # print(ref, val)
        EQE_bd['other'] = val
        EQE_bd['collected'] = self.EQE
        # print(val)
        return EQE_bd

    def calculate_current_loss(self):
        EQE_bd = self._EQE_breakdown()

        if self.Jph is None:
            self.Jph = analysis.AM15G_resample(self.wl)

        Jloss = OrderedDict()
        # Jloss = np.dot(self.refl_metal * np.ones(len(self.Jph)),
        # self.Jph)
        for key, value in EQE_bd.items():
            Jloss[key] = np.trapz(value * self.Jph / 100, self.wl)

        return Jloss

    def clear(self):

        self.refl_front = None
        self.refl_metal = None
        self.Jph = None

    def plot_EQE_breakdown(self, ax):
        '''
        Plots the EQE breakdown
        '''
        running_max = np.ones(self.wl.shape[0]) * 100

        clist = rcParams['axes.color_cycle']
        cgen = itertools.cycle(clist)
        eqe_bd = self._EQE_breakdown()
        for break_down, value in eqe_bd.items():

            c = next(cgen)
            ax.fill_between(self.wl, running_max,
                            running_max - value, facecolor=c)
            ax.plot(np.inf, np.inf, c=c,
                    label=break_down.replace('_', ' '))
            running_max -= value

        ax.set_ylim(0, 100)
        ax.legend(loc='best')

        # assert np.allclose(running_max, self.EQE)

    def plot_Jloss(self, ax):
        J_loss = self.calculate_current_loss()
        # print(J_loss)
        keys = [k
                for k in sorted(J_loss, key=J_loss.get, reverse=True)]
        values = [J_loss[k]
                  for k in sorted(J_loss, key=J_loss.get, reverse=True)]
        # print(s)
        waterfall(ax, values, keys)
        ax.set_ylim(bottom=0.9 * np.amax(values))
        ax.set_ylabel('Current (mA/cm$^{-2}$)')
        # print(J_loss)
        # ax.plot(self.wl, self.Jph)
        # ax.set_ylabel('Yea!')
        # print(self.wl, self.Jph)

    def plot_IQE(self, ax):
        ax.plot(self.wl, self.IQE, '.-', label='IQE')
        ax.set_xlabel('Wavelength [$nm$]')
        ax.set_ylabel('QE [%]')
        ax.legend(loc='best')
        ax.grid(True)


class Cell(object):

    # data structures
    refl = None
    qe = None
    sunsVoc = None
    div = None
    liv = None

    losses_optical = None
    losses_recombination = None


    def __init__(self, thickness=None, **kwargs):
        self.thickness = thickness  # [cm]
        self.sample_names = {}
        self.input_errors = {}

        if 'reflectance_fname' in kwargs:
            self.refl = Refl(
                kwargs['reflectance_loader'],
                kwargs['reflectance_fname'])
        if 'EQE_fname' in kwargs:
            self.qe = QE(
                kwargs['EQE_loader'],
                kwargs['EQE_fname'])
        if 'suns Voc_fname' in kwargs:
            self.sunsVoc = IVSuns(
                kwargs['suns Voc_loader'],
                kwargs['suns Voc_fname'])
        if 'dark IV_fname' in kwargs:
            self.div = IVDark(kwargs['dark IV_loader'],
                              kwargs['dark IV_fname'])
        if 'light IV_fname' in kwargs:
            self.liv = IVLight(kwargs['light IV_loader'], kwargs[
                               'light IV_fname'])

        # print(kwargs)
        # self.example_dir = os.path.join(os.pardir, 'example_cell')

        # self.check_input_vals()

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
        # Jsc_iqe = self.qe.output['Jsc']
        # delta = (Jsc_iqe - Jsc_liv) / Jsc_liv
        # self.input_errors['Jsc'] = delta

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
            quick_print(key, '{:s}%'.format(val * 100))
        for key, val in self.input_errors.items():
            quick_print(key, '{:.3e}%'.format(val * 100))

        output_list.append('\n')
        quick_print('##### Reflectance', '')
        quick_print('filename', self.refl.filename)
        quick_print('WAR', '{:.3f}'.format(self.refl.WAR))
        quick_print('f_metal', '{:.3f}'.format(self.refl.f_metal))

        output_list.append('\n')
        quick_print('##### QE', '')
        # quick_print('filename', self.qe.filename)
        # for key, val in self.qe.output.items():
        # quick_print(key, val)
        # quick_print('Basore fit Leff', '{:.3e}'.format(
        # self.qe.output_Basore_fit['Leff']))
        # quick_print('Basore fit eta_c', '{:.3f}'.format(
        # self.qe.output_Basore_fit['eta_c']))
        # for key, val in self.qe.Jloss_qe.items():
        #     quick_print(key, '{:.3f}'.format(val))

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

        for key, val in self.liv.ideal_FF.items():
            quick_print(key, '{:.3f}'.format(val))

        self.output_list = output_list

    def print_output_to_file(self):

        filename = self.cell_name + '_loss_analysis_summary.csv'

        output_file = open(os.path.join(self.output_dir, filename), 'w')

        for item in self.output_list:
            output_file.write(item + '\r\n')

        output_file.close()

    def plot_all(self, save_fig_bool=False):
        '''Plot the output of previous calculations'''

        # # for reflectance
        # fig_QE = self._plot_jsc_loss_measurements()
        # fig_IV = self._plot_FF_loss_measurements()
        #
        # # for loss analysis summary
        # fig_LA = plt.figure('LA', figsize=(30 / 2.54, 15 / 2.54))
        # fig_LA.clf()
        #
        # ax_FF = fig_LA.add_subplot(2, 2, 1)
        # ax_Jloss = fig_LA.add_subplot(2, 2, 2)
        #
        # self.liv.plot_FF1(ax_FF)
        # # self.qe.plot_Jloss(ax_Jloss)
        #
        # fig_IV.set_tight_layout(True)
        #
        # if save_fig_bool:
        #
        #     fig_QE.savefig(os.path.join(self.output_dir,
        #                                 self.cell_name + '_QE.png'))
        #     fig_IV.savefig(os.path.join(self.output_dir,
        #                                 self.cell_name + '_IV.png'))
        #
        # plt.show()
        pass

    def plot_all2(self):

        if None not in [self.liv, self.div, self.sunsVoc]:
            self._plot_FF_losses()

        if None not in [self.refl, self.qe]:
            self._plot_optical_loss_measurements()

        plt.show()

    def _plot_FF_losses(self):
        # for light and dark IV
        fig_IV = plt.figure('IV', figsize=(30 / 2.54, 15 / 2.54))
        fig_IV.clf()

        # get the plotting axes
        ax_logIV = fig_IV.add_subplot(2, 2, 1)
        ax_ideality = fig_IV.add_subplot(2, 2, 3)
        ax_lightIV = fig_IV.add_subplot(2, 2, 2)
        ax_FF = fig_IV.add_subplot(2, 2, 4)

        # plot light IV first, as is typically the noisest
        if self.liv is not None:
            self.liv.plot_mV(ax_ideality)
            self.liv.plot_JV(ax_lightIV)

        # plot suns Voc
        if self.sunsVoc is not None:
            self.sunsVoc.plot_mV(ax_ideality)
            self.sunsVoc.plot_JV(ax_lightIV)
            # self.sunsVoc.plot_tau(ax_tau)
            self.sunsVoc.plot_log_JV(ax_logIV)

        # plot dark IV as least noisest
        if self.div is not None:
            self.div.plot_log_JV(ax_logIV)
            self.div.plot_mV(ax_ideality)

        if self.div is not None:
            self.losses_recombination.plot_FF_loss(ax_FF)
        ax_lightIV.legend(loc='best')

        return fig_IV

    def _plot_optical_loss_measurements(self):
        '''
        plots all the Jsc loss stuff
        '''
        # create a figure
        fig_QE = plt.figure('QE', figsize=(30 / 2.54, 15 / 2.54))
        fig_QE.clf()

        ax_refl = fig_QE.add_subplot(2, 2, 1)
        ax_QE = fig_QE.add_subplot(2, 2, 2)
        ax_QE_fit = fig_QE.add_subplot(2, 2, 3)
        ax_QE_layered = fig_QE.add_subplot(2, 2, 4)

        # plot the stuff
        self.refl.plot(ax_refl)
        self.qe.plot_EQE(ax_QE)

        self.losses_optical.plot_IQE(ax_QE)
        self.qe.plot_EQE(ax_QE_layered)
        self.losses_optical.plot_EQE_breakdown(ax_QE_layered)
        self.losses_optical.plot_Jloss(ax_QE_fit)

        # plot the EQE fitted data ??
        # self.qe.plot_Basore_fit(ax_QE_fit)

        fig_QE.set_tight_layout(True)

        return fig_QE

    def process_all(self, save_fig_bool=None, output_dir=None, cell_name=None):
        '''
        A function that calls all the processing functions.
        '''

        # if cell_name is None:
        #     self.cell_name = self.liv.output['Cell Name ']
        # else:
        #     self.cell_name = cell_name

        # self.output_dir = output_dir

        if self.qe is not None and self.refl is not None:
            self.losses_optical = optical_losses(wavelength=self.qe.wl,
                                                 EQE=self.qe.EQE,
                                                 reflection=self.refl.reflection,
                                                 refl_front=self.refl.refl_front,
                                                 refl_metal=self.refl.refl_metal)
            self.losses_optical.calculate_losses()

        if self.liv is not None and self.div is not None:
            self.losses_recombination = recombination_losses(
                Voc=self.liv.Voc,
                Jsc=self.liv.Jsc,
                Vmp=self.liv.Vmp,
                Jmp=self.liv.Jmp,
                Rs=self.liv.Rs,
                Rsh=self.div.Rsh,
                FF=self.liv.FF
            )
            self.losses_recombination.calculate()
        # self._jsc_loss()
        # self._FF_loss()

        # self.collect_outputs()
        # self.print_output_to_file()
        # self.plot_all(save_fig_bool)

    def _jsc_loss(self):
        '''
        A function that performs the calculations required for the calculation
        of Jsc losses
        '''

        # make sure we have the required data types
        assert self.refl is not None
        # assert self.qe is not None

        # process that data
        self.refl.process()
        # self.qe.process(self.refl.wl, self.refl.refl, self.refl.refl_wo_escape,
        # self.refl.Jloss, self.refl.f_metal)

    def _FF_loss(self):

        # make sure we have the required data types
        assert self.sunsVoc is not None
        assert self.liv is not None
        assert self.div is not None

        # process the data
        self.sunsVoc.process()

        self.Rsh = self.div.process()

        self.Rs_1 = analysis.Rs_calc_1(self.liv.output['Vmp'],
                                       self.liv.output['Jmp'],
                                       self.sunsVoc.V, self.sunsVoc.J)

        self.Rs_2 = analysis.Rs_calc_2(self.liv.output['Voc'],
                                       self.liv.output['Jsc'],
                                       self.liv.output['FF'],
                                       self.sunsVoc.output['PFF'])

        self.liv.process(self.Rsh, self.Rs_1)


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
    cell1._plot_optical_loss_measurements()
    cell1._plot_FF_losses()
    # plt.show()
