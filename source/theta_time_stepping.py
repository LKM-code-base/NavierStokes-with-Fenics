#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from enum import Enum, auto
from discrete_time import DiscreteTime
import math


class ThetaTimeSteppingType(Enum):
    ForwardEuler = auto()
    BackwardEuler = auto()
    CrankNicolson = auto()
    FractionalStep01 = auto()
    FractionalStep02 = auto()


class GeneralThetaTimeStepping(DiscreteTime):
    """Generalized fractional step theta-scheme according to V. John (2016),
    Tables 7.1 and 7.2 on pg. 397.
    """
    _theta = 1.0 - math.sqrt(2.0) / 2.0
    _zeta = 1.0 - 2.0 * _theta
    _tau = _zeta / (1.0 - _theta)
    _eta = 1.0 - _tau

    def __init__(self, start_time, end_time, theta_type, desired_start_time_step=0.0):
        super().__init__(start_time, end_time, desired_start_time_step)

        assert isinstance(theta_type, ThetaTimeSteppingType)
        self._type = theta_type

        # set parameters
        if self._type == ThetaTimeSteppingType.ForwardEuler:
            self._n_steps = 1
            self._Theta = [(0.0, 1.0, 1.0, 0.0)]

        elif self._type == ThetaTimeSteppingType.BackwardEuler:
            self._n_steps = 1
            self._Theta = [(1.0, 0.0, 0.0, 1.0)]

        elif self._type == ThetaTimeSteppingType.CrankNicolson:
            self._n_steps = 1
            self._Theta = [(0.5, 0.5, 0.5, 0.5)]

        elif self._type == ThetaTimeSteppingType.FractionalStep01:
            self._n_steps = 3
            theta = self._theta
            zeta = self._zeta
            tau = self._tau
            eta = self._eta
            self._Theta = [(tau * theta, eta * theta, eta * theta, tau * theta),
                           (eta * zeta, tau * zeta, tau * zeta, eta * zeta),
                           (tau * theta, eta * theta, eta * theta, tau * theta)]

        elif self._type == ThetaTimeSteppingType.FractionalStep02:
            self._n_steps = 3
            theta = self._theta
            zeta = self._zeta
            tau = self._tau
            eta = self._eta
            self._Theta = [(tau * theta, eta * theta, theta, 0.0),
                           (eta * zeta, tau * zeta, 0.0, zeta),
                           (tau * theta, eta * theta, theta, 0.0)]

        self._intermediate_timesteps = [0.0] * self._n_steps
        self._intermediate_times = [[0.0] * self._n_steps for i in range(2)]

    def restart(self):
        """
        Resets all member variables to the initial state.
        """
        super().restart()

        self._intermediate_timesteps = [0.0] * self._n_steps
        self._intermediate_times = [[0.0] * self._n_steps for i in range(2)]

    def update_coefficients(self):
        # compute intermediate time steps
        next_step_size = self.get_next_step_size()
        assert math.isfinite(next_step_size)
        if self._type == ThetaTimeSteppingType.FractionalStep01 or\
                self._type == ThetaTimeSteppingType.FractionalStep02:
            self._intermediate_timesteps[0] = self._theta * next_step_size
            self._intermediate_timesteps[1] = self._zeta * next_step_size
            self._intermediate_timesteps[2] = self._theta * next_step_size
        else:
            self._intermediate_timesteps[0] = next_step_size
        # compute intermediate times
        current_time = self.current_time
        next_time = self.next_time
        if self._type == ThetaTimeSteppingType.FractionalStep01 or\
                self._type == ThetaTimeSteppingType.FractionalStep02:
            self._intermediate_times[0][0] = current_time
            self._intermediate_times[0][1] = current_time + self._theta * next_step_size
            self._intermediate_times[0][2] = next_time - self._theta * next_step_size
            self._intermediate_times[1][0] = current_time + self._theta * next_step_size
            self._intermediate_times[1][1] = next_time - self._theta * next_step_size
            self._intermediate_times[1][2] = next_time
        else:
            self._intermediate_times[0][0] = current_time
            self._intermediate_times[1][0] = next_time

    @property
    def theta(self):
        return self._Theta

    @property
    def intermediate_timesteps(self):
        return self._intermediate_timesteps

    @property
    def intermediate_times(self):
        return self._intermediate_times

    @property
    def n_levels(self):
        """Returns the number of solution of previous timesteps required."""
        return 1

    @property
    def n_steps(self):
        """Returns the number of substeps required to proceed from the current
        time level to the next time level."""
        return self._n_steps
