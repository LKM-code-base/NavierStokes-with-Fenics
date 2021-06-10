#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from theta_time_stepping import GeneralThetaTimeStepping, ThetaTimeSteppingType
import math


def compare_lists(a, b):
    assert a == b, "The list {0} is not equal to the list {1}".format(a, b)


step_sizes = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0]

_theta = 1.0 - math.sqrt(2.0) / 2.0
_zeta = 1.0 - 2.0 * _theta
_tau = _zeta / (1.0 - _theta)
_eta = 1.0 - _tau


def time_loop(time_stepping, theta, intermediate_times, intermediate_timesteps):
    # simple time loop
    while not time_stepping.is_at_end():
        # extract step number and step size
        step_number = time_stepping.step_number
        step_size = step_sizes[step_number]
        # set next step size
        time_stepping.set_desired_next_step_size(step_size)
        # update coefficients
        time_stepping.update_coefficients()
        # print info
        print(time_stepping)
        # check correctness of coefficients
        compare_lists(time_stepping.theta, theta[step_number])
        compare_lists(time_stepping.intermediate_times, intermediate_times[step_number])
        compare_lists(time_stepping.intermediate_timesteps, intermediate_timesteps[step_number])
        # advance time
        time_stepping.advance_time()
    print(time_stepping)
    assert time_stepping.is_at_end()
    # restart
    time_stepping.restart()
    while not time_stepping.is_at_end():
        # extract step number and step size
        step_number = time_stepping.step_number
        step_size = step_sizes[step_number]
        # set next step size
        time_stepping.set_desired_next_step_size(step_size)
        # update coefficients
        time_stepping.update_coefficients()
        # print info
        print(time_stepping)
        # check correctness of coefficients
        compare_lists(time_stepping.theta, theta[step_number])
        compare_lists(time_stepping.intermediate_times, intermediate_times[step_number])
        compare_lists(time_stepping.intermediate_timesteps, intermediate_timesteps[step_number])
        # advance time
        time_stepping.advance_time()
    print(time_stepping)
    assert time_stepping.is_at_end()


def test_forward_euler():
    time_stepping = GeneralThetaTimeStepping(0.0, 9.0, ThetaTimeSteppingType.ForwardEuler)
    theta = [[(0.0, 1.0, 1.0, 0.0)]] * len(step_sizes)
    intermediate_timesteps = []
    for i in range(len(step_sizes)):
        intermediate_timesteps.append([step_sizes[i]])
    intermediate_times = []
    for i in range(len(step_sizes)):
        if i > 0:
            intermediate_times.append(
                    [[intermediate_times[i-1][1][0]],
                     [intermediate_times[i-1][1][0] + step_sizes[i]]])
        else:
            intermediate_times.append([[0.0], [step_sizes[i]]])

    time_loop(time_stepping, theta, intermediate_times, intermediate_timesteps)


def test_backward_euler():
    time_stepping = GeneralThetaTimeStepping(0.0, 9.0, ThetaTimeSteppingType.BackwardEuler)
    theta = [[(1.0, 0.0, 0.0, 1.0)]] * len(step_sizes)
    intermediate_timesteps = []
    for i in range(len(step_sizes)):
        intermediate_timesteps.append([step_sizes[i]])
    intermediate_times = []
    for i in range(len(step_sizes)):
        if i > 0:
            intermediate_times.append(
                    [[intermediate_times[i-1][1][0]],
                     [intermediate_times[i-1][1][0] + step_sizes[i]]])
        else:
            intermediate_times.append([[0.0], [step_sizes[i]]])

    time_loop(time_stepping, theta, intermediate_times, intermediate_timesteps)


def test_crank_nicolson():
    time_stepping = GeneralThetaTimeStepping(0.0, 9.0, ThetaTimeSteppingType.CrankNicolson)
    theta = [[(0.5, 0.5, 0.5, 0.5)]] * len(step_sizes)
    intermediate_timesteps = []
    for i in range(len(step_sizes)):
        intermediate_timesteps.append([step_sizes[i]])
    intermediate_times = []
    for i in range(len(step_sizes)):
        if i > 0:
            intermediate_times.append(
                    [[intermediate_times[i-1][1][0]],
                     [intermediate_times[i-1][1][0] + step_sizes[i]]])
        else:
            intermediate_times.append([[0.0], [step_sizes[i]]])

    time_loop(time_stepping, theta, intermediate_times, intermediate_timesteps)


def test_fractional_step01():
    time_stepping = GeneralThetaTimeStepping(0.0, 9.0, ThetaTimeSteppingType.FractionalStep01)
    theta = [[(_tau * _theta, _eta * _theta, _eta * _theta, _tau * _theta),
              (_eta * _zeta, _tau * _zeta, _tau * _zeta, _eta * _zeta),
              (_tau * _theta, _eta * _theta, _eta * _theta, _tau * _theta)]] * len(step_sizes)
    intermediate_timesteps = []
    for i in range(len(step_sizes)):
        k = step_sizes[i]
        intermediate_timesteps.append([_theta * k, _zeta * k, _theta * k])
    intermediate_times = []
    for i in range(len(step_sizes)):
        k = step_sizes[i]
        if i > 0:
            previous_time = intermediate_times[i-1][1][2]
            next_time = intermediate_times[i-1][1][2] + k
        else:
            previous_time = 0.0
            next_time = k
        intermediate_times.append(
                [[previous_time, previous_time + _theta * k, next_time - _theta * k],
                 [previous_time + _theta * k, next_time - _theta * k, next_time]])

    time_loop(time_stepping, theta, intermediate_times, intermediate_timesteps)


def test_fractional_step02():
    time_stepping = GeneralThetaTimeStepping(0.0, 9.0, ThetaTimeSteppingType.FractionalStep02)
    theta = [[(_tau * _theta, _eta * _theta, _theta, 0.0),
              (_eta * _zeta, _tau * _zeta, 0.0, _zeta),
              (_tau * _theta, _eta * _theta, _theta, 0.0)]] * len(step_sizes)

    intermediate_timesteps = []
    for i in range(len(step_sizes)):
        k = step_sizes[i]
        intermediate_timesteps.append([_theta * k, _zeta * k, _theta * k])
    intermediate_times = []
    for i in range(len(step_sizes)):
        k = step_sizes[i]
        if i > 0:
            previous_time = intermediate_times[i-1][1][2]
            next_time = intermediate_times[i-1][1][2] + k
        else:
            previous_time = 0.0
            next_time = k
        intermediate_times.append(
                [[previous_time, previous_time + _theta * k, next_time - _theta * k],
                 [previous_time + _theta * k, next_time - _theta * k, next_time]])

    time_loop(time_stepping, theta, intermediate_times, intermediate_timesteps)


if __name__ == "__main__":
    test_forward_euler()
    test_backward_euler()
    test_crank_nicolson()
    test_fractional_step01()
    test_fractional_step02()
