#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def calculate_next_time(current_time, step_size, end_time):
    """
    Returns the next time which is computed based on the step size and the
    current time. The value of the step size might be adjusted such that the
    next time is exactly equal to the end time and to avoid very small time
    steps.
    """
    # input check
    assert isinstance(current_time, float)
    assert isinstance(step_size, float)
    assert isinstance(end_time, float)
    assert step_size >= 0.
    assert end_time >= current_time
    # compute next time
    next_time = current_time + step_size
    # compute tolerance
    relative_tolerance = 0.05
    time_tolerance = relative_tolerance * step_size
    # adjust next time if necessary
    if next_time > end_time - time_tolerance:
        next_time = end_time
    return next_time


class DiscreteTime:
    def __init__(self, start_time, end_time, desired_start_time_step=0.0):
        """
        Default constructor.
        """
        # input check
        assert isinstance(start_time, float)
        assert isinstance(end_time, float)
        assert isinstance(desired_start_time_step, float)
        assert start_time < end_time
        assert desired_start_time_step >= 0.0

        # initialize members variables
        self._start_time = start_time
        self._end_time = end_time
        self._current_time = self._start_time
        self._next_time = calculate_next_time(start_time,
                                              desired_start_time_step,
                                              end_time)
        self._previous_time = start_time
        self._start_step_size = self._next_time - self._start_time
        self._step_number = 0

    def __str__(self):
        """
        Returns a string containing information on the current status of the
        time discretization.
        """
        string = "step number {0:8d}, ".format(self._step_number)
        string += "current time {0:10.2e}, ".format(self._current_time)
        string += "next step size {0:10.2e}".format(self.get_next_step_size())
        return string

    @property
    def current_time(self):
        """
        Returns the current time.
        """
        return self._current_time

    @property
    def next_time(self):
        """
        Returns the next time.
        """
        return self._next_time

    @property
    def previous_time(self):
        """
        Returns the previous time.
        """
        return self._previous_time

    @property
    def start_time(self):
        """
        Returns the start time.
        """
        return self._start_time

    @property
    def end_time(self):
        """
        Returns the end time.
        """
        return self._end_time

    def is_at_start(self):
        """
        Returns a boolean indicating whether the method is at the start.
        """
        return self._step_number == 0

    def is_at_end(self):
        """
        Returns a boolean indicating whether has reached the end time.
        """
        return self._current_time == self._end_time

    def get_next_step_size(self):
        """
        Returns the next step size.
        """
        return self._next_time - self._current_time

    def get_previous_step_size(self):
        """
        Returns the previous step size.
        """
        return self._current_time - self._previous_time

    @property
    def step_number(self):
        """
        Returns the step number.
        """
        return self._step_number

    def set_desired_next_step_size(self, next_step_size):
        """
        Sets the next step size to the desired value.
        """
        assert isinstance(next_step_size, float)
        assert next_step_size > 0.0
        self._next_time = calculate_next_time(self._current_time,
                                              next_step_size,
                                              self._end_time)

    def advance_time(self):
        """
        Advance member variables to the next time level.
        """
        # sanity check
        assert self._next_time > self._current_time
        # get step size
        step_size = self.get_next_step_size()
        # advance time variables
        self._previous_time = self._current_time
        self._current_time = self._next_time
        # increment step number
        self._step_number += 1
        # compute next time level
        self._next_time = calculate_next_time(self._current_time, step_size,
                                              self._end_time)

    def restart(self):
        """
        Resets all member variables to the initial state.
        """
        # reset member variables
        self._previous_time = self._start_time
        self._current_time = self._start_time
        self._next_time = calculate_next_time(self._current_time,
                                              self._start_step_size,
                                              self._end_time)
        self._step_number = 0

    def set_end_time(self, new_end_time):
        """
        Modifies the value of the end time.
        """
        # input check
        assert isinstance(new_end_time, float)
        assert new_end_time > self._start_time
        assert new_end_time > self._current_time
        # set new end time
        self._end_time = new_end_time
        # set desired step size
        if (self._step_number == 0):
            step_size = self._start_step_size
        else:
            step_size = self.get_previous_step_size()
        # calculate next time level
        self._next_time = calculate_next_time(self._current_time, step_size,
                                              self._end_time)
