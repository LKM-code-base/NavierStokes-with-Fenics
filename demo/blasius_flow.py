#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from auxiliary_classes import EquationCoefficientHandler

from grid_generator import blasius_plate
from ns_problem import StationaryProblem
from ns_solver_base import VelocityBCType

dlfn.set_log_level(20)


class BlasiusFlowProblem(StationaryProblem):
    def __init__(self, main_dir=None):
        super().__init__(main_dir)
        self._problem_name = "BlasiusFlow"

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers, self._boundary_marker_map = blasius_plate()

    def set_boundary_conditions(self):
        # velocity boundary conditions
        inlet_velocity = dlfn.Expression(("1.0", "0.0"),
                                         h=0.5, y0=0.5, degree=2)
        self._bcs = ((VelocityBCType.function, self._boundary_marker_map["inlet"], inlet_velocity),
                     (VelocityBCType.no_normal_flux, self._boundary_marker_map["bottom"], None),
                     (VelocityBCType.no_normal_flux, self._boundary_marker_map["top"], None))

    def set_equation_coefficients(self):
        self._coefficient_handler = EquationCoefficientHandler(Re=200.0)

    def set_internal_constraints(self):
        self._internal_constraints = ((VelocityBCType.no_slip, self._boundary_marker_map["plate"], None), )

    def postprocess_solution(self):
        # add pressure gradient to the field output
        self._add_to_field_output(self._compute_pressure_gradient())
        # add vorticity to the field output
        self._add_to_field_output(self._compute_vorticity())


if __name__ == "__main__":
    blasius_flow = BlasiusFlowProblem()
    blasius_flow.solve_problem()
