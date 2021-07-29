#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from auxiliary_classes import EquationCoefficientHandler
from grid_generator import hyper_cube, HyperCubeBoundaryMarkers
from ns_problem import StationaryProblem
from ns_solver_base import VelocityBCType
dlfn.set_log_level(40)


class CavityProblem(StationaryProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir)
        self._n_points = n_points
        self._problem_name = "Cavity"

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_boundary_conditions(self):
        # velocity boundary conditions
        no_slip = VelocityBCType.no_slip
        constant = VelocityBCType.constant
        BoundaryMarkers = HyperCubeBoundaryMarkers
        self._bcs = ((no_slip, BoundaryMarkers.left.value, None),
                     (no_slip, BoundaryMarkers.right.value, None),
                     (no_slip, BoundaryMarkers.bottom.value, None),
                     (constant, BoundaryMarkers.top.value, (1.0, 0.0)))

    def set_equation_coefficients(self):
        self._coefficient_handler = EquationCoefficientHandler(Re=10.0)


if __name__ == "__main__":
    cavity_flow = CavityProblem(25)
    cavity_flow.solve_problem()
