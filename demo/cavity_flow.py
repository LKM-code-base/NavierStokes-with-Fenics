#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from grid_generator import hyper_cube, HyperCubeBoundaryMarkers
from navier_stokes_problem import StationaryNavierStokesProblem
from navier_stokes_solver_base import VelocityBCType
dlfn.set_log_level(40)


class CavityProblem(StationaryNavierStokesProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir)

        self._n_points = n_points
        self._problem_name = "Cavity"

        self.set_parameters(Re=10.0)

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


if __name__ == "__main__":
    cavity_flow = CavityProblem(25)
    cavity_flow.solve_problem()
