#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from navier_stokes_problem import InstationaryNavierStokesProblem
from navier_stokes_problem import VelocityBCType
from grid_generator import hyper_rectangle
from grid_generator import open_hyper_cube
from grid_generator import HyperCubeBoundaryMarkers
from grid_generator import HyperRectangleBoundaryMarkers
from imex_time_stepping import IMEXType

dlfn.set_log_level(30)


class ChannelFlowProblem(InstationaryNavierStokesProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir, start_time=0.0, end_time=1.0,
                         desired_start_time_step=0.01, n_max_steps=100)

        self._n_points = n_points
        self._problem_name = "ChannelFlow"

        self._imex_type = IMEXType.SBDF2
        self.set_parameters(Re=10.0)

        self._output_frequency = 10
        self._postprocessing_frequency = 10

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_rectangle((0.0, 0.0), (10.0, 1.0), (100, 10))
        self.write_boundary_markers()

    def set_initial_conditions(self):
        self._initial_conditions = dict()
        self._initial_conditions["velocity"] = (0.0, 0.0)

    def set_boundary_conditions(self):
        # velocity boundary conditions
        inlet_velocity = dlfn.Expression(("6.0*x[1]*(1.0-x[1]) * (1.0 + 0.5 * sin(M_PI * t))", "0.0"),
                                         degree=2, t=0.0)
        self._bcs = ((VelocityBCType.function, HyperRectangleBoundaryMarkers.left.value, inlet_velocity),
                     (VelocityBCType.no_slip, HyperRectangleBoundaryMarkers.bottom.value, None),
                     (VelocityBCType.no_slip, HyperRectangleBoundaryMarkers.top.value, None))

    def postprocess_solution(self):
        # add pressure gradient to the field output
        self._add_to_field_output(self._compute_pressure_gradient())
        # add vorticity to the field output
        self._add_to_field_output(self._compute_vorticity())


class GravityDrivenFlowProblem(InstationaryNavierStokesProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir, start_time=0.0, end_time=1.0,
                         desired_start_time_step=0.01, n_max_steps=100)

        self._n_points = n_points
        self._problem_name = "OpenCubeTransient"

        self._imex_type = IMEXType.SBDF2
        self.set_parameters(Re=100.0, Fr=1.0)

        self._output_frequency = 10
        self._postprocessing_frequency = 10

    def setup_mesh(self):
        # create mesh
        openings = (("bottom", (0.4, 0.0), 0.4),
                    ("left", (0.0, 0.5), 0.1),
                    ("right", (1.0, 0.7), 0.1),
                    ("bottom", (0.7, 0.0), 0.05),
                    ("top", (0.5, 1.0), 0.8))
        self._mesh, self._boundary_markers = open_hyper_cube(2, self._n_points, openings)
        self.write_boundary_markers()

    def set_initial_conditions(self):
        self._initial_conditions = dict()

        self._initial_conditions["velocity"] = (0.0, 0.0)

    def set_boundary_conditions(self):
        # velocity boundary conditions
        self._bcs = ((VelocityBCType.no_slip, HyperCubeBoundaryMarkers.left.value, None),
                     (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.right.value, None),
                     (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.bottom.value, None),
                     (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.top.value, None))

    def postprocess_solution(self):
        # add pressure gradient to the field output
        self._add_to_field_output(self._compute_pressure_gradient())
        # add vorticity to the field output
        self._add_to_field_output(self._compute_vorticity())

    def set_body_force(self):
        self._body_force = dlfn.Constant((0.0, -1.0))


def test_channel_flow():
    channel_flow = ChannelFlowProblem(50)
    channel_flow.solve_problem()


def test_transient_gravity_driven_flow():
    gravity_flow = GravityDrivenFlowProblem(50)
    gravity_flow.solve_problem()


if __name__ == "__main__":
    test_channel_flow()
    test_transient_gravity_driven_flow()
