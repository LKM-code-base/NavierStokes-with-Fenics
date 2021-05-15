#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from imex_time_stepping import IMEXTimeStepping, IMEXType


def compare_lists(a, b):
    assert a == b, "The list {0} is not equal to the list {1}".format(a, b)


step_sizes = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0]

eta = [[1.0, 0.0],
       [2.0, -1.0],
       [3.0, -2.0],
       [2.0, -1.0],
       [1.5, -0.5],
       [2.0, -1.0],
       [2.0, -1.0]]
assert len(eta) == len(step_sizes)

coefficients_changed = [True, True, True, True, True, True, False]
assert len(coefficients_changed) == len(step_sizes)


def time_loop(time_stepping, alpha, beta, gamma):
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
        time_stepping.print_coefficients()
        # check correctness of coefficients
        compare_lists(time_stepping.alpha, alpha[step_number])
        compare_lists(time_stepping.beta, beta[step_number])
        compare_lists(time_stepping.gamma, gamma[step_number])
        compare_lists(time_stepping.eta, eta[step_number])
        assert (time_stepping.coefficients_changed == coefficients_changed[step_number])
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
        time_stepping.print_coefficients()
        # check correctness of coefficients
        compare_lists(time_stepping.alpha, alpha[step_number])
        compare_lists(time_stepping.beta, beta[step_number])
        compare_lists(time_stepping.gamma, gamma[step_number])
        compare_lists(time_stepping.eta, eta[step_number])
        assert (time_stepping.coefficients_changed == coefficients_changed[step_number])
        # advance time
        time_stepping.advance_time()
    print(time_stepping)
    assert time_stepping.is_at_end()


def test_SBDF2():
    time_stepping = IMEXTimeStepping(0.0, 9.0, IMEXType.SBDF2)
    alpha = [[1.0, -1.0, 0.0],
             [1.5, -2.0, 0.5],
             [5.0/3.0, -3.0, 4.0/3.0],
             [1.5, -2.0, 0.5],
             [4.0/3.0, -1.5, 1.0/6.0],
             [1.5, -2.0, 0.5],
             [1.5, -2.0, 0.5]]
    beta = [[1.0, 0.0],
            [2.0, -1.0],
            [3.0, -2.0],
            [2.0, -1.0],
            [1.5, -0.5],
            [2.0, -1.0],
            [2.0, -1.0]]
    gamma = [[1.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [1.0, 0.0, 0.0]]

    assert len(alpha) == len(step_sizes)
    assert len(beta) == len(step_sizes)
    assert len(gamma) == len(step_sizes)

    time_loop(time_stepping, alpha, beta, gamma)


def test_CNAB():
    time_stepping = IMEXTimeStepping(0.0, 9.0, IMEXType.CNAB)
    alpha = [[1.0, -1.0, 0.0],
             [1.0, -1.0, 0.0],
             [1.0, -1.0, 0.0],
             [1.0, -1.0, 0.0],
             [1.0, -1.0, 0.0],
             [1.0, -1.0, 0.0],
             [1.0, -1.0, 0.0]]
    beta = [[1.0, 0.0],
            [1.5, -0.5],
            [2.0, -1.0],
            [1.5, -0.5],
            [1.25, -0.25],
            [1.5, -0.5],
            [1.5, -0.5]]
    gamma = [[1.0, 0.0, 0.0],
             [0.5, 0.5, 0.0],
             [0.5, 0.5, 0.0],
             [0.5, 0.5, 0.0],
             [0.5, 0.5, 0.0],
             [0.5, 0.5, 0.0],
             [0.5, 0.5, 0.0]]

    assert len(alpha) == len(step_sizes)
    assert len(beta) == len(step_sizes)
    assert len(gamma) == len(step_sizes)

    time_loop(time_stepping, alpha, beta, gamma)


def test_mCNAB():
    time_stepping = IMEXTimeStepping(0.0, 9.0, IMEXType.mCNAB)
    alpha = [[1.0, -1.0, 0.0],
             [1.0, -1.0, 0.0],
             [1.0, -1.0, 0.0],
             [1.0, -1.0, 0.0],
             [1.0, -1.0, 0.0],
             [1.0, -1.0, 0.0],
             [1.0, -1.0, 0.0]]
    beta = [[1.0, 0.0],
            [1.5, -0.5],
            [2.0, -1.0],
            [1.5, -0.5],
            [1.25, -0.25],
            [1.5, -0.5],
            [1.5, -0.5]]
    gamma = [[1.0, 0.0, 0.0],
             [9.0/16.0, 6.0/16.0, 1.0/16.0],
             [17.0/32.0, 13.0/32.0, 1.0/16.0],
             [9.0/16.0, 6.0/16.0, 1.0/16.0],
             [5.0/8.0, 2.5/8.0, 1.0/16.0],
             [9.0/16.0, 6.0/16.0, 1.0/16.0],
             [9.0/16.0, 6.0/16.0, 1.0/16.0]]

    assert len(alpha) == len(step_sizes)
    assert len(beta) == len(step_sizes)
    assert len(gamma) == len(step_sizes)

    time_loop(time_stepping, alpha, beta, gamma)


def test_CNLF():
    time_stepping = IMEXTimeStepping(0.0, 9.0, IMEXType.CNLF)
    alpha = [[1.0, -1.0, 0.0],
             [1.0/2.0, 0.0, -1.0/2.0],
             [1.0/3.0, 1.0, -4.0/3.0],
             [1.0/2.0, 0.0, -1.0/2.0],
             [2.0/3.0, -0.5, -1.0/6.0],
             [1.0/2.0, 0.0, -1.0/2.0],
             [1.0/2.0, 0.0, -1.0/2.0]]
    beta = [[1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0]]
    gamma = [[1.0, 0.0, 0.0],
             [1.0/2.0, 0.0, 1.0/2.0],
             [1.0/4.0, 1.0/4.0, 1.0/2.0],
             [1.0/2.0, 0.0, 1.0/2.0],
             [1.0, -1.0/2.0, 1.0/2.0],
             [1.0/2.0, 0.0, 1.0/2.0],
             [1.0/2.0, 0.0, 1.0/2.0]]

    assert len(alpha) == len(step_sizes)
    assert len(beta) == len(step_sizes)
    assert len(gamma) == len(step_sizes)

    time_loop(time_stepping, alpha, beta, gamma)


if __name__ == "__main__":
    test_SBDF2()
    test_CNAB()
    test_mCNAB()
    test_CNLF()
