#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from grid_generator import hyper_cube
from grid_generator import open_hyper_cube
from grid_generator import spherical_shell


def test_hyper_cube():
    # two-dimensional case
    _, _ = hyper_cube(2, 8)
    # three-dimensional case
    _, _ = hyper_cube(3, 8)


def test_open_hyper_cube():
    # two-dimensional case
    openings = (("left", (0.0, 0.5), 0.1),
                ("right", (1.0, 0.7), 0.1),
                ("bottom", (0.7, 0.0), 0.05),
                ("top", (0.5, 1.0), 0.8))
    _, _ = open_hyper_cube(2, 8, openings)
    # three-dimensional case
    openings = (("left", (0.0, 0.5, 0.5), (0.1, 0.2)),
                ("right", (1.0, 0.7, 0.3), (0.1, 0.1)),
                ("bottom", (0.7, 0.0, 0.7), (0.05, 0.2)),
                ("top", (0.5, 1.0, 0.2), (0.8, 0.8)),
                ("back", (0.7, 0.3, 0.0), (0.05, 0.1)),
                ("front", (0.5, 0.25, 1.0), (0.2, 0.3))
                )
    _, _ = open_hyper_cube(3, 8, openings)


def test_spherical_shell():
    # two-dimensional case
    _, _ = spherical_shell(2, (0.3, 1.0), 2)
    # three-dimensional case
    _, _ = spherical_shell(3, (0.3, 1.0), 2)


if __name__ == "__main__":
    test_hyper_cube()
    test_open_hyper_cube()
    test_spherical_shell()
#    test_converging_diverging_pipe()
