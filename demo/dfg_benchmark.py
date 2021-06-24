#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from grid_generator import channel_with_cylinder
from ns_bdf_solver import ImplicitBDFSolver
from ns_problem import InstationaryProblem
from ns_solver_base import VelocityBCType

dlfn.set_log_level(40)


class DFGBenchmark2D2(InstationaryProblem):
    def __init__(self, main_dir=None):
        super().__init__(main_dir, start_time=0.0, end_time=1.0,
                         desired_start_time_step=0.001, n_max_steps=50)

        self._problem_name = "DFGBenchmark2D2"

        self.set_parameters(Re=100.0)

        self._output_frequency = 10
        self._postprocessing_frequency = 10

        self.set_solver_class(ImplicitBDFSolver)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers, self._boundary_marker_map = channel_with_cylinder()

    def set_initial_conditions(self):
        self._initial_conditions = dict()
        self._initial_conditions["velocity"] = (0.0, 0.0)

    def set_boundary_conditions(self):
        # velocity boundary conditions
        inlet_velocity = dlfn.Expression(("6.0*x[1]/h*(1.0-x[1]/h)", "0.0"), h=4.1,
                                         degree=2)

        self._bcs = ((VelocityBCType.function, self._boundary_marker_map["inlet"], inlet_velocity),
                     (VelocityBCType.no_slip, self._boundary_marker_map["cylinder"], None),
                     (VelocityBCType.no_slip, self._boundary_marker_map["upper wall"], None),
                     (VelocityBCType.no_slip, self._boundary_marker_map["lower wall"], None))

    def postprocess_solution(self):
        # add pressure gradient to the field output
        self._add_to_field_output(self._compute_pressure_gradient())
        # add vorticity to the field output
        self._add_to_field_output(self._compute_vorticity())


if __name__ == "__main__":
    dfg_benchmark = DFGBenchmark2D2()
    dfg_benchmark.solve_problem()
