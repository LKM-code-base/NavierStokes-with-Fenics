#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from auxiliary_classes import EquationCoefficientHandler
from grid_generator import backward_facing_step
from ns_problem import StationaryProblem
from ns_solver_base import VelocityBCType

dlfn.set_log_level(20)


class BackwardFacingStepProblem(StationaryProblem):
    def __init__(self, main_dir=None):
        super().__init__(main_dir)
        self._problem_name = "BackwardFacingStep"

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers, self._boundary_marker_map = backward_facing_step()

    def set_boundary_conditions(self):
        # velocity boundary conditions
        inlet_velocity = dlfn.Expression(("6.0*(x[1] - y0)/h*(1.0-(x[1] - y0)/h)", "0.0"),
                                         h=0.5, y0=0.5, degree=2)
        self._bcs = ((VelocityBCType.function, self._boundary_marker_map["inlet"], inlet_velocity),
                     (VelocityBCType.no_slip, self._boundary_marker_map["walls"], None))

    def set_equation_coefficients(self):
        self._coefficient_handler = EquationCoefficientHandler(Re=50.0)

    def postprocess_solution(self):
        # add pressure gradient to the field output
        self._add_to_field_output(self._compute_pressure_gradient())
        # add vorticity to the field output
        self._add_to_field_output(self._compute_vorticity())


if __name__ == "__main__":
    gravity_flow = BackwardFacingStepProblem()
    gravity_flow.solve_problem()
