#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from enum import Enum, auto

from discrete_time import DiscreteTime

import math


class IMEXType(Enum):
    CNAB = auto()
    mCNAB = auto()
    CNLF = auto()
    SBDF2 = auto()


class IMEXTimeStepping(DiscreteTime):
    def __init__(self, start_time, end_time, imex_type, desired_start_time_step=0.0):
        super().__init__(start_time, end_time, desired_start_time_step)

        assert isinstance(imex_type, IMEXType)
        self._type = imex_type

        # set parameters
        if self._type == IMEXType.SBDF2:
            self._imex_parameters = (1.0, 0.0)
        elif self._type == IMEXType.CNAB:
            self._imex_parameters = (0.5, 0.0)
        elif self._type == IMEXType.mCNAB:
            self._imex_parameters = (0.5, 1.0/8.0)
        elif self._type == IMEXType.CNLF:
            self._imex_parameters = (0.0, 1.0)
        else:  # pragma: no cover
            raise ValueError("Type of IMEX Scheme {0} is unknown.".format(self._type))

        self._coefficients_changed = True

        # initialize time step ratio
        self._omega = -1.0

        # initialize coefficients according to first order scheme
        self._alpha = [1.0, -1.0, 0.]
        self._beta = [1.0, 0.]
        self._gamma = [1.0, 0., 0.]
        # initialize extrapolation coefficients
        self._eta = [1.0, 0.0]

    def restart(self):
        """
        Resets all member variables to the initial state.
        """
        super().restart()

        assert hasattr(self, "_type")
        # set parameters
        if self._type == IMEXType.SBDF2:
            self._imex_parameters = (1.0, 0.0)
        elif self._type == IMEXType.CNAB:
            self._imex_parameters = (0.5, 0.0)
        elif self._type == IMEXType.mCNAB:
            self._imex_parameters = (0.5, 1.0/8.0)
        elif self._type == IMEXType.CNLF:
            self._imex_parameters = (0.0, 1.0)
        else:  # pragma: no cover
            raise ValueError("Type of IMEX Scheme {0} is unknown.".format(self._type))

        self._coefficients_changed = True

        # initialize time step ratio
        self._omega = -1.0

        # initialize coefficients according to first order scheme
        self._alpha = [1.0, -1.0, 0.]
        self._beta = [1.0, 0.]
        self._gamma = [1.0, 0., 0.]
        # initialize extrapolation coefficients
        self._eta = [1.0, 0.]

    def update_coefficients(self):
        # do not change coefficients in the first step
        if (self._step_number == 0):
            return
        # compute new time step ratio
        omega = self.get_next_step_size() / self.get_previous_step_size()
        assert math.isfinite(omega)
        assert omega > 0.0

        # check if time step ratio has changed and update the IMEX coefficients
        # from the first to the second time step the coefficients change because
        # of switching from a first order to a second order scheme
        if self._omega == omega and self._step_number > 1:
            self._coefficients_changed = False
            return
        else:
            self._omega = omega

            a, b = self._imex_parameters

            self._alpha[0] = (1.0 + 2.0 * a * self._omega) / (1.0 + omega)
            self._alpha[1] = ((1.0 - 2.0 * a) * omega - 1.0)
            self._alpha[2] = (2.0 * a - 1.0) * omega * omega / (1.0 + omega)

            self._beta[0] = 1.0 + a * omega
            self._beta[1] = - a * omega

            self._gamma[0] = a + b / (2.0 * omega)
            self._gamma[1] = 1.0 - a - (1.0 + 1.0 / omega) * b / 2.0
            self._gamma[2] = b / 2.0

            # Taylor extrapolation coefficients
            self._eta[0] = 1.0 + omega
            self._eta[1] = - omega

            self._coefficients_changed = True

    def print_coefficients(self):
        print("+-" + "-+-".join(4 * (12 * "-", )) + "-+")
        print("| {:12} | {:12} | {:12} | {:12} |"
              .format("coefficient", "n + 1", "n", "n - 1"))
        print("| {:12} | {:12.2e} | {:12.2e} | {:12.2e} |"
              .format("alpha", *self._alpha))
        print("| {:12} | ".format("beta") + 12 * " " +
              " | {:12.2g} | {:12.2e} |".format(*self._beta))
        print("| {:12} | {:12.2e} | {:12.2e} | {:12.2g} |"
              .format("gamma", *self._gamma))
        print("| {:12} | ".format("eta") + 12 * " " +
              " | {:12.2g} | {:12.2e} |".format(*self._eta))

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def gamma(self):
        return self._gamma

    @property
    def eta(self):
        return self._eta

    @property
    def coefficients_changed(self):
        return self._coefficients_changed

    @property
    def n_levels(self):
        """Returns the number of solution of previous timesteps required."""
        return len(self._alpha) - 1

    @property
    def n_substeps(self):
        """Returns the number of substeps required to proceed from the current
        time level to the next time level."""
        return 1
