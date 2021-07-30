#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
import numpy as np
from grid_generator import hyper_cube
from ns_solver_base import SolverBase

# mesh
n_points = 5
mesh, boundary_markers = hyper_cube(2, n_points)
# constructor
solver = SolverBase(mesh, boundary_markers)
# setup function space
solver._setup_function_spaces()
Wh = solver._Wh
WhSub = solver._get_subspaces()
# solutions
solution = dlfn.Function(Wh)
solution_dict = dict()
solution_dict["velocity"] = dlfn.Function(WhSub["velocity"])
solution_dict["pressure"] = dlfn.Function(WhSub["pressure"])


def test_forward_assignment():
    # forward assignment
    dlfn.project(dlfn.Constant((1.0, 2.0, 3.0)), Wh, function=solution)
    solver._assign_function(solution_dict, solution)
    assert np.allclose(solution_dict["velocity"](0.1, 0.1), np.array([1.0, 2.0]))
    assert np.allclose(solution_dict["pressure"](0.1, 0.1), np.array([3.0]))
    # forward assignment split
    dlfn.project(dlfn.Constant((10.0, 20.0, 30.0)), Wh, function=solution)
    velocity, pressure = solution.split()
    solver._assign_function({"velocity": solution_dict["velocity"]}, velocity)
    solver._assign_function({"pressure": solution_dict["pressure"]}, pressure)
    assert np.allclose(solution_dict["velocity"](0.1, 0.1), np.array([10.0, 20.0]))
    assert np.allclose(solution_dict["pressure"](0.1, 0.1), np.array([30.0]))
    # forward assignment split / no dict
    dlfn.project(dlfn.Constant((100.0, 200.0, 300.0)), Wh,
                 function=solution)
    solver._assign_function(solution_dict["velocity"], velocity)
    solver._assign_function(solution_dict["pressure"], pressure)
    assert np.allclose(solution_dict["velocity"](0.1, 0.1), np.array([100.0, 200.0]))
    assert np.allclose(solution_dict["pressure"](0.1, 0.1), np.array([300.0]))


def test_backward_assignment():
    # backward assignment
    dlfn.project(dlfn.Constant((-1.0, -2.0)), WhSub["velocity"],
                 function=solution_dict["velocity"])
    dlfn.project(dlfn.Constant(-3.0), WhSub["pressure"],
                 function=solution_dict["pressure"])
    solver._assign_function(solution, solution_dict)
    assert np.allclose(solution(0.1, 0.1), np.array([-1.0, -2.0, -3.0]))
    # backward assignment split
    dlfn.project(dlfn.Constant((-10.0, -20.0)), WhSub["velocity"],
                 function=solution_dict["velocity"])
    dlfn.project(dlfn.Constant(-30.0), WhSub["pressure"],
                 function=solution_dict["pressure"])
    velocity, pressure = solution.split()
    solver._assign_function(velocity, {"velocity": solution_dict["velocity"]})
    assert np.allclose(solution(0.1, 0.1), np.array([-10.0, -20.0, -3.0]))
    solver._assign_function(pressure, {"pressure": solution_dict["pressure"]})
    assert np.allclose(solution(0.1, 0.1), np.array([-10.0, -20.0, -30.0]))
    # backward assignment split / no dict
    dlfn.project(dlfn.Constant((-100.0, -200.0)), WhSub["velocity"],
                 function=solution_dict["velocity"])
    dlfn.project(dlfn.Constant(-300.0), WhSub["pressure"],
                 function=solution_dict["pressure"])
    solver._assign_function(velocity, solution_dict["velocity"])
    assert np.allclose(solution(0.1, 0.1), np.array([-100.0, -200.0, -30.0]))
    solver._assign_function(pressure, solution_dict["pressure"])
    assert np.allclose(solution(0.1, 0.1), np.array([-100.0, -200.0, -300.0]))


if __name__ == "__main__":
    test_forward_assignment()
    test_backward_assignment()
