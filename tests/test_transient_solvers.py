#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from auxiliary_classes import EquationCoefficientHandler
from ns_problem import InstationaryProblem
from ns_solver_base import VelocityBCType
from ns_solver_base import PressureBCType
from ns_bdf_solver import ImplicitBDFSolver
from grid_generator import hyper_cube
from grid_generator import hyper_rectangle
from grid_generator import open_hyper_cube
from grid_generator import HyperRectangleBoundaryMarkers
from grid_generator import HyperCubeBoundaryMarkers


dlfn.set_log_level(30)


class PeriodicDomain(dlfn.SubDomain):
    def inside(self, x, on_boundary):
        """Return True if `x` is located on the master edge and False
        else.
        """
        inside = False
        if (dlfn.near(x[0], 0.0) and on_boundary):
            inside = True
        elif (dlfn.near(x[1], 0.0) and on_boundary):
            inside = True
        return inside

    def map(self, x_slave, x_master):
        """Map the coordinates of the support points (nodes) of the degrees
        of freedom of the slave to the coordinates of the corresponding
        master edge.
        """
        # points at the right edge
        if dlfn.near(x_slave[0], 1.0):
            x_master[0] = x_slave[0] - 1.0
            x_master[1] = x_slave[1]
        # points at the top edge
        elif dlfn.near(x_slave[1], 1.0):
            x_master[0] = x_slave[0]
            x_master[1] = x_slave[1] - 1.0
        else:
            # map other outside of the domain
            x_master[0] = -10.0
            x_master[1] = -10.0


class ChannelFlowProblem(InstationaryProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir, start_time=0.0, end_time=1.0,
                         desired_start_time_step=0.01, n_max_steps=10)
        self._n_points = n_points
        self._problem_name = "ChannelFlow"
        self._output_frequency = 10
        self._postprocessing_frequency = 10
        self.set_solver_class(ImplicitBDFSolver)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_rectangle((0.0, 0.0), (10.0, 1.0),
                                                             (10 * self._n_points, self._n_points))

    def set_equation_coefficients(self):
        self._coefficient_handler = EquationCoefficientHandler(Re=10.0)

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


class GravityDrivenFlowProblem(InstationaryProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir, start_time=0.0, end_time=1.0,
                         desired_start_time_step=0.01, n_max_steps=10)
        self._n_points = n_points
        self._problem_name = "OpenCubeTransient"
        self._output_frequency = 10
        self._postprocessing_frequency = 10
        self.set_solver_class(ImplicitBDFSolver)

    def setup_mesh(self):
        # create mesh
        openings = (("bottom", (0.4, 0.0), 0.4),
                    ("left", (0.0, 0.5), 0.1),
                    ("right", (1.0, 0.7), 0.1),
                    ("bottom", (0.7, 0.0), 0.05),
                    ("top", (0.5, 1.0), 0.8))
        self._mesh, self._boundary_markers = open_hyper_cube(2, self._n_points, openings)

    def set_equation_coefficients(self):
        self._coefficient_handler = EquationCoefficientHandler(Re=100.0, Fr=1.0)

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


class TaylorGreenVortex(InstationaryProblem):
    _gamma = gamma = 2.0 * dlfn.pi

    def __init__(self, main_dir=None):
        super().__init__(main_dir, start_time=0.0, end_time=1.0,
                         desired_start_time_step=0.1, n_max_steps=10)
        self._problem_name = "TaylorGreenVortex"
        self._n_points = 16
        self._output_frequency = 0
        self._postprocessing_frequency = 0
        self.set_solver_class(ImplicitBDFSolver)

    def setup_mesh(self):
        assert self._n_points is not None
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_equation_coefficients(self):
        self._coefficient_handler = EquationCoefficientHandler(Re=100.0)

    def set_initial_conditions(self):
        self._initial_conditions = dict()
        self._initial_conditions["velocity"] = \
            dlfn.Expression(("cos(gamma * x[0]) * sin(gamma * x[1])",
                             "-sin(gamma * x[0]) * cos(gamma * x[1])"),
                            gamma=self._gamma, degree=3)
        self._initial_conditions["pressure"] = \
            dlfn.Expression("-1.0/4.0 * (cos(2.0 * gamma * x[0]) + cos(2.0 * gamma * x[1]))",
                            gamma=self._gamma, degree=3)

    def set_boundary_conditions(self):
        # pressure mean value constraint
        self._bcs = ((PressureBCType.mean_value, None, 0.0), )

    def set_periodic_boundary_conditions(self):
        """Set periodic boundary conditions in x- and y-direction."""
        self._periodic_bcs = PeriodicDomain()
        self._periodic_boundary_ids = (HyperCubeBoundaryMarkers.left.value,
                                       HyperCubeBoundaryMarkers.right.value,
                                       HyperCubeBoundaryMarkers.top.value,
                                       HyperCubeBoundaryMarkers.bottom.value)


def test_channel_flow():
    channel_flow = ChannelFlowProblem(5)
    channel_flow.solve_problem()


def test_transient_gravity_driven_flow():
    gravity_flow = GravityDrivenFlowProblem(32)
    gravity_flow.solve_problem()


def test_taylor_green_vortex():
    taylor_green = TaylorGreenVortex()
    taylor_green.solve_problem()


if __name__ == "__main__":
    test_channel_flow()
    test_transient_gravity_driven_flow()
    test_taylor_green_vortex()
