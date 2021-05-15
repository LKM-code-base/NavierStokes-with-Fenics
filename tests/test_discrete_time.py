#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from discrete_time import DiscreteTime
import numpy as np


def test_discrete_time():
    time_stepping = DiscreteTime(0.0, 5.0)

    print("start time {0:6.2f}".format(time_stepping.start_time))
    print("next time  {0:6.2f}".format(time_stepping.next_time))
    print("end time   {0:6.2f}".format(time_stepping.end_time))

    if time_stepping.is_at_start():
        while not time_stepping.is_at_end():
            print(time_stepping)
            time_stepping.set_desired_next_step_size(np.random.rand())
            time_stepping.advance_time()
        print(time_stepping)
        assert time_stepping.is_at_end()

        time_stepping.restart()
        while not time_stepping.is_at_end():
            print(time_stepping)
            time_stepping.set_desired_next_step_size(np.random.rand())
            time_stepping.advance_time()
        print(time_stepping)
        assert time_stepping.is_at_end()

        print("current time  {0:6.2f}".format(time_stepping.current_time))
        print("previous time {0:6.2f}".format(time_stepping.previous_time))

        print("next step size     {0:6.2f}"
              .format(time_stepping.get_next_step_size()))
        print("previous step size {0:6.2f}"
              .format(time_stepping.get_previous_step_size()))

        time_stepping.set_end_time(10.0)
        while not time_stepping.is_at_end():
            print(time_stepping)
            time_stepping.set_desired_next_step_size(np.random.rand())
            time_stepping.advance_time()
        print(time_stepping)
        assert time_stepping.is_at_end()


if __name__ == "__main__":
    test_discrete_time()
