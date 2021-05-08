#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dolfin as dlfn

from navier_stokes_problem import StationaryNavierStokesProblem

from navier_stokes_solver import VelocityBCType

from grid_generator import open_hyper_cube, HyperCubeBoundaryMarkers

dlfn.set_log_level(20)


class GravityDrivenFlowProblem(StationaryNavierStokesProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir)

        self._n_points = n_points
        self._problem_name = "OpenCube"

        self.set_parameters(Re=25.0,  Fr=10.0)

    def setup_mesh(self):
        # create mesh
        openings = (("bottom", (0.2, 0.0), 0.1),
                    ("left", (0.0, 0.5), 0.1),
                    ("right", (1.0, 0.7), 0.1),
                    ("bottom", (0.7, 0.0), 0.05),
                    ("top", (0.5, 1.0), 0.8))
        self._mesh, self._boundary_markers = open_hyper_cube(2, self._n_points,
                                                             openings)
        self.write_boundary_markers()

    def set_boundary_conditions(self):
        # velocity boundary conditions
        no_slip = VelocityBCType.no_slip
        BoundaryMarkers = HyperCubeBoundaryMarkers
        velocity_bcs = (
                (no_slip, BoundaryMarkers.left.value, None),
                (no_slip, BoundaryMarkers.right.value, None),
                (no_slip, BoundaryMarkers.bottom.value, None))
        self._bcs = {"velocity": velocity_bcs}

    def postprocess_solution(self):
        pressure = self._get_pressure()
        velocity = self._get_velocity()

        strings = tuple("x[{0:d}]".format(i) for i in range(self._space_dim))
        position_vector = dlfn.Expression(strings, degree=1)

        potential_energy = dlfn.dot(self._body_force, position_vector)

        Phi = dlfn.Constant(0.5) * dlfn.dot(velocity, velocity)
        Phi += pressure + potential_energy / dlfn.Constant(self._Fr)**2

        Vh = dlfn.FunctionSpace(self._mesh, "CG", 1)
        phi = dlfn.project(Phi, Vh)
        phi.rename("Bernoulli potential", "")

    def set_body_force(self):
        self._body_force = dlfn.Constant((0.0, -1.0))


if __name__ == "__main__":
    gravity_flow = GravityDrivenFlowProblem(25)
    gravity_flow.solve_problem()
