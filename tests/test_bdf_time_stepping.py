#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from bdf_time_stepping import BDFTimeStepping


def compare_lists(a, b):
    assert isinstance(a, (tuple, list))
    assert isinstance(b, (tuple, list))
    if isinstance(b, list) and isinstance(a, tuple):
        b = tuple(b)
    if isinstance(a, list) and isinstance(b, tuple):
        a = tuple(a)
    assert a == b, "The list {0} is not equal to the list {1}".format(a, b)


step_sizes = [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0]
step_sizes_changed = [True, True, True, False, True, False, False]
assert len(step_sizes_changed) == len(step_sizes)


def time_loop(time_stepping, alpha, coefficients_changed):
    # simple time loop
    while not time_stepping.is_at_end():
        # extract step number and step size
        step_number = time_stepping.step_number
        step_size = step_sizes[step_number]
        # set next step size
        time_stepping.set_desired_next_step_size(step_size)
        print(time_stepping)
        # update coefficients
        time_stepping.update_coefficients()
        # check correctness of coefficients
        compare_lists(time_stepping.coefficients(1), alpha[1][step_number])
        compare_lists(time_stepping.coefficients(2), alpha[2][step_number])
        assert (time_stepping.coefficients_changed(1) == coefficients_changed[1][step_number])
        assert (time_stepping.coefficients_changed(2) == coefficients_changed[2][step_number])
        # print info
        time_stepping.print_coefficients()
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
        print(time_stepping)
        # update coefficients
        time_stepping.update_coefficients()
        # check correctness of coefficients
        compare_lists(time_stepping.coefficients(1), alpha[1][step_number])
        compare_lists(time_stepping.coefficients(2), alpha[2][step_number])
        assert (time_stepping.coefficients_changed(1) == coefficients_changed[1][step_number])
        assert (time_stepping.coefficients_changed(2) == coefficients_changed[2][step_number])
        # print info
        time_stepping.print_coefficients()
        # advance time
        time_stepping.advance_time()
    print(time_stepping)
    assert time_stepping.is_at_end()


def test_first_order():
    coefficients_changed = {1: [True, False, False, False, False, False, False],
                            2: [True, True, True, True, True, True, False]}

    time_stepping = BDFTimeStepping(0.0, 9.0, order=1)
    alpha = dict()
    alpha[1] = [[1.0, -1.0],
                [1.0, -1.0],
                [1.0, -1.0],
                [1.0, -1.0],
                [1.0, -1.0],
                [1.0, -1.0],
                [1.0, -1.0]]
    assert len(alpha[1]) == len(step_sizes)
    alpha[2] = [[1.0, -2.0, 1.0],
                [1.0, -2.0, 1.0],
                [4.0/3.0, -4.0, 8.0/3.0],
                [1.0, -2.0, 1.0],
                [2.0/3.0, -1.0, 1.0/3.0],
                [1.0, -2.0, 1.0],
                [1.0, -2.0, 1.0]]
    assert len(alpha[2]) == len(step_sizes)
    time_loop(time_stepping, alpha, coefficients_changed)


def test_second_order():
    coefficients_changed = {1: [True, True, True, True, True, True, False],
                            2: [True, True, True, True, True, True, True]}

    time_stepping = BDFTimeStepping(0.0, 9.0, order=2)
    alpha = dict()
    alpha[1] = [[1.0, -1.0, 0.0],
                [3.0/2.0, -2.0, 1.0/2.0],
                [5.0/3.0, -3.0, 4.0/3.0],
                [3.0/2.0, -2.0, 1.0/2.0],
                [4.0/3.0, -3.0/2.0, 1.0/6.0],
                [3.0/2.0, -2.0, 1.0/2.0],
                [3.0/2.0, -2.0, 1.0/2.0]]
    assert len(alpha[1]) == len(step_sizes)
    alpha[2] = [[1.0, -2.0, 1.0, 0.0],
                [2.0, -5.0, 4.0, -1.0],
                [3.0, -14.0, 16.0, -5.0],
                [11.0/5.0, -6.0, 7.0, -16.0/5.0],
                [6.0/5.0, -2.0, 1.0, -1.0/5.0],
                [7.0/4.0, -4.0, 5.0/2.0, -1.0/4.0],
                [2.0, -5.0, 4.0, -1.0]]
    assert len(alpha[2]) == len(step_sizes)
    time_loop(time_stepping, alpha, coefficients_changed)


if __name__ == "__main__":
    test_first_order()
    test_second_order()
