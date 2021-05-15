#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
dlfn.set_log_level(40)

from navier_stokes_problem import StationaryNavierStokesProblem, VelocityBCType
from grid_generator import hyper_cube, open_hyper_cube, HyperCubeBoundaryMarkers


class CavityProblem(StationaryNavierStokesProblem):
    def __init__(self, n_points, main_dir = None):
        super().__init__(main_dir)

        self._n_points = n_points
        self._problem_name  = "Cavity"

        self.set_parameters(Re = 10.0)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_boundary_conditions(self):
        # velocity boundary conditions
        velocity_bcs = (
                (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.left.value, None),
                (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.right.value, None),
                (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.bottom.value, None),
                (VelocityBCType.constant, HyperCubeBoundaryMarkers.top.value, (1.0, 0.0)))
        self._bcs = {"velocity": velocity_bcs}


class GravityDrivenFlowProblem(StationaryNavierStokesProblem):
    def __init__(self, n_points, main_dir = None):
        super().__init__(main_dir)

        self._n_points = n_points
        self._problem_name  = "OpenCube"

        self.set_parameters(Re=25.0,  Fr=10.0)

    def setup_mesh(self):
        # create mesh
        openings = (("bottom", (0.2, 0.0), 0.1),
                    ("left", (0.0, 0.5), 0.1),
                    ("right", (1.0, 0.7), 0.1),
                    ("bottom", (0.7, 0.0), 0.05),
                    ("top", (0.5, 1.0), 0.8))
        self._mesh, self._boundary_markers = open_hyper_cube(2, self._n_points, openings)
        self.write_boundary_markers()

    def set_boundary_conditions(self):
        # velocity boundary conditions
        velocity_bcs = (
                (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.left.value, None),
                (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.right.value, None),
                (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.bottom.value, None))
        self._bcs = {"velocity": velocity_bcs}

    def set_body_force(self):
        self._body_force = dlfn.Constant((0.0, -1.0))


def test_cavity():
    cavity_flow = CavityProblem(25)
    cavity_flow.solve_problem()


def test_gravity_driven_flow():
    gravity_flow = GravityDrivenFlowProblem(25)
    gravity_flow.solve_problem()


if __name__ == "__main__":
    test_cavity()
    test_gravity_driven_flow()
