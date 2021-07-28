#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from discrete_time import DiscreteTime
import math


class BDFTimeStepping(DiscreteTime):
    """Time stepping using adaptive backward differentiation formulas for
    derivative of degree upto three.

    """
    def __init__(self, start_time, end_time, order=2, desired_start_time_step=0.0):
        super().__init__(start_time, end_time, desired_start_time_step)

        assert isinstance(order, int)
        assert order > 0
        if order > 2:  # pragma: no cover
            raise NotImplementedError()
        self._order = order

        self._coefficients_changed = {1: True, 2: True}

        # initialize list of time step ratios
        self._omega = [1.0, 1.0]

        # initialize coefficient dictionary
        self._alpha = dict()
        # coefficients of the first derivative
        self._alpha[1] = [0.0, ] * (self._order + 1)
        self._alpha[1][0] = 1.0   # initialize with values of
        self._alpha[1][1] = -1.0  # a first order scheme
        # coefficients of the second derivative
        self._alpha[2] = [0.0, ] * (self._order + 2)
        self._alpha[2][0] = 1.0   # initialize with values of
        self._alpha[2][1] = -2.0  # a first order scheme
        self._alpha[2][2] = 1.0

    def restart(self):
        """
        Resets all member variables to the initial state.
        """
        super().restart()

        self._coefficients_changed = {1: True, 2: True}

        # initialize time step ratio
        self._omega = [1.0, 1.0]

        # coefficients of the first derivative
        self._alpha[1] = [0.0, ] * (self._order + 1)
        self._alpha[1][0] = 1.0   # initialize with values of
        self._alpha[1][1] = -1.0  # a first order scheme
        # coefficients of the second derivative
        self._alpha[2] = [0.0, ] * (self._order + 2)
        self._alpha[2][0] = 1.0   # initialize with values of
        self._alpha[2][1] = -2.0  # a first order scheme
        self._alpha[2][2] = 1.0

    def update_coefficients(self):
        # do not change coefficients in the first step
        if (self.step_number == 0):
            return
        # compute new time step ratios
        omega = self.get_next_step_size() / self.get_previous_step_size()
        assert math.isfinite(omega)
        assert omega > 0.0
        Omega = self._omega[0]
        assert Omega > 0.0

        # check if time step ratio has changed and update the coefficients
        # from the first to the second time step the coefficients change because
        # of switching from a first order to a second order scheme
        if self._order == 1:
            if (self._omega[0] == omega and self._step_number > 1):
                for key in self._coefficients_changed.keys():
                    self._coefficients_changed[key] = False
                return
            else:
                self._omega[0] = omega
                self._omega[1] = Omega

                # coefficients of the first derivative
                self._alpha[1][0] = 1.0
                self._alpha[1][1] = -1.0
                self._coefficients_changed[1] = False
                # coefficients of the second derivative
                self._alpha[2][0] = 2.0 * omega / (1.0 + omega)
                self._alpha[2][1] = -2.0 * omega
                self._alpha[2][2] = 2.0 * omega * omega / (1.0 + omega)
                self._coefficients_changed[2] = True
        elif self._order == 2:
            if (self._omega[0] == omega and self._omega[1] == Omega and
                    self._step_number > 1):
                for key in self._coefficients_changed.keys():
                    self._coefficients_changed[key] = False
                return
            elif (self._omega[0] == omega and self._omega[1] != Omega and
                  self._step_number > 1):
                self._omega[1] = Omega

                # coefficients of the first derivative
                self._coefficients_changed[1] = False
                # coefficients of the second derivative
                num = 2.0 * omega * (1.0 + (2.0 + 3.0 * omega) * Omega)
                denom = (1.0 + omega) * (1.0 + Omega + omega * Omega)
                self._alpha[2][0] = num / denom
                num = -2.0 * omega * (1.0 + 2.0 * (1.0 + omega) * Omega)
                denom = 1.0 + Omega
                self._alpha[2][1] = num / denom
                num = 2.0 * omega**2 * (1.0 + Omega + 2.0 * omega * Omega)
                denom = 1.0 + omega
                self._alpha[2][2] = num / denom
                num = -2.0 * omega**2 * (1.0 + 2.0 * omega) * Omega**3
                denom = (1.0 + Omega) * (1.0 + Omega + omega * Omega)
                self._alpha[2][3] = num / denom
                self._coefficients_changed[2] = True

            else:
                self._omega[0] = omega
                self._omega[1] = Omega

                # coefficients of the first derivative
                self._alpha[1][0] = (1.0 + 2.0 * omega) / (1.0 + omega)
                self._alpha[1][1] = -(1.0 + omega)
                self._alpha[1][2] = omega * omega / (1.0 + omega)
                self._coefficients_changed[1] = True
                # coefficients of the second derivative
                num = 2.0 * omega * (1.0 + (2.0 + 3.0 * omega) * Omega)
                denom = (1.0 + omega) * (1.0 + Omega + omega * Omega)
                self._alpha[2][0] = num / denom
                num = -2.0 * omega * (1.0 + 2.0 * (1.0 + omega) * Omega)
                denom = 1.0 + Omega
                self._alpha[2][1] = num / denom
                num = 2.0 * omega**2 * (1.0 + Omega + 2.0 * omega * Omega)
                denom = 1.0 + omega
                self._alpha[2][2] = num / denom
                num = -2.0 * omega**2 * (1.0 + 2.0 * omega) * Omega**3
                denom = (1.0 + Omega) * (1.0 + Omega + omega * Omega)
                self._alpha[2][3] = num / denom
                self._coefficients_changed[2] = True

    def print_coefficients(self):
        """Print the coefficients of the different derivatives to STDOUT."""
        levels = ("n + 1", "n", "n - 1", "n - 2")
        n_levels = 2 + self._order
        n_columns = n_levels + 1
        print("+-" + "-+-".join(n_columns * (12 * "-", )) + "-+")
        print(("| {:12} | " + " | ".join(n_levels * ("{:12}", )))
              .format("derivative", *levels[:n_levels]) + " |")
        for d, coeffs in self._alpha.items():
            if d == 1:
                d_str = "1st"
            elif d == 2:
                d_str = "2nd"
            if len(coeffs) == n_levels:
                line = "| {:12} | " + " | ".join(n_levels * ("{:12.2e}", ))
                line = line.format(d_str, *coeffs)
                line += " |"
                print(line)
            else:
                line = "| {:12} | " + " | ".join(len(coeffs) * ("{:12.2e}", ))
                line = line.format(d_str, *coeffs)
                line += " | "
                line += " | ".join((n_levels - len(coeffs)) * (12 * " ", ))
                line += " |"
                print(line)
        print("+-" + "-+-".join(n_columns * (12 * "-", )) + "-+")

    def coefficients(self, derivative):
        assert derivative in (1, 2)
        return tuple(self._alpha[derivative])

    def coefficients_changed(self, derivative):
        assert derivative in (1, 2)
        return self._coefficients_changed[derivative]

    def n_levels(self, derivative=1):
        """Returns the number of solution of previous timesteps required."""
        assert derivative in (1, 2)
        return len(self._alpha[derivative]) - 1

    @property
    def n_substeps(self):
        """Returns the number of substeps required to proceed from the current
        time level to the next time level."""
        return 1
