import unittest
from loss_analysis import analysis as anal
import numpy.testing as npt
import numpy as np


class TestSomething(unittest.TestCase):

    def test_find_nearest(self):
        xdata = [0, 1, 2, 3]
        x_val = 1.1
        ydata = [5, 6, 7, 8]
        npt.assert_allclose(anal.find_nearest(x_val, xdata), 1)
        npt.assert_allclose(anal.find_nearest(x_val, xdata, ydata), 6)

    def test_ideality_factor(self):

        # tests if the ideality factor is being calculated properly
        # define a voltage
        V = np.arange(0.0, 0.3, 0.1)
        vt = 1

        # then define the current, and in doing so set the ideality factor
        J1 = 1e-12 * np.exp(V / vt)
        J2 = 1e-12 * np.exp(V / vt / 2)

        # check this is what you set.
        npt.assert_allclose(
            anal.ideality_factor(V, J1, vt), 1 * np.ones(V.shape[0])
        )
        npt.assert_allclose(
            anal.ideality_factor(V, J2, vt), 2 * np.ones(V.shape[0])
        )

    def test_wl_to_alpha(self):

        # check some values hard written from the file
        npt.assert_allclose(anal.wl_to_alpha(2.50E+02),	1.84E+06)
        npt.assert_allclose(anal.wl_to_alpha(1.45E+03),	1.20E-08)
        npt.assert_allclose(anal.wl_to_alpha(1.055e3),
                            np.average([16.3, 11.1]))

    def test_basore(self):
        wave = np.arange(1040, 1100, 10)
        alpha = anal.wl_to_alpha(wave)

        # theta is just a scale in the x-axis, so we can undo this by
        # prescallign the x axis
        for theta in [0, 10]:
            # checks linear fits with a gradient and a slope input
            for m, b in zip([1., 3.], [1, 5]):
                IQE = 1. / (m / alpha * np.cos(np.deg2rad(theta)) + b)

                fit_output, plot = anal.fit_Basore(wave, IQE, theta=theta)

                npt.assert_allclose(fit_output['Leff'], m)
                npt.assert_allclose(fit_output['eta_c'], 1. / b)

        # check that the bounds work
        wave = np.arange(950, 1040, 10)
        alpha = anal.wl_to_alpha(wave)
        for m, b in zip([1., 3.], [1, 5]):
            IQE = 1. / (m / alpha * np.cos(np.deg2rad(theta)) + b)

            fit_output, plot = anal.fit_Basore(
                wave, IQE, theta=theta, wlbounds=(950, 1040))

            npt.assert_allclose(fit_output['Leff'], m)
            npt.assert_allclose(fit_output['eta_c'], 1. / b)

    def test_FF_ideal(self):

        Voc = 0.64
        Jsc = 0.035
        Rs = 1
        Rsh = 1e2
        T = 300

        # this is a functionality test
        # check ideal is returned
        npt.assert_allclose(anal._ideal_FF(
            Voc, T), anal.FF_ideal(Voc, T=T))
        npt.assert_allclose(anal._ideal_FF(Voc, T),
                            anal.FF_ideal(Voc, T=T, Jsc=Jsc))
        npt.assert_allclose(anal._ideal_FF(Voc, T),
                            anal.FF_ideal(Voc, T=T,  Rs=Rs))
        # check series is returned
        npt.assert_allclose(anal._ideal_FF_series(Voc, T, Jsc, Rs),
                            anal.FF_ideal(Voc, T=T, Jsc=Jsc, Rs=Rs))
        # check shunt is returned
        npt.assert_allclose(anal._ideal_FF_shunt_series(Voc, T, Jsc, Rs, Rsh),
                            anal.FF_ideal(Voc, T=T, Jsc=Jsc, Rs=Rs, Rsh=Rsh))
