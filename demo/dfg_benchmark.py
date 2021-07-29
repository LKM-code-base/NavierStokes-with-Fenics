#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from auxiliary_classes import EquationCoefficientHandler
from grid_generator import channel_with_cylinder
from ns_bdf_solver import ImplicitBDFSolver
from ns_problem import InstationaryProblem
from ns_solver_base import VelocityBCType
import numpy as np

dlfn.set_log_level(40)


class DFGBenchmark2D2(InstationaryProblem):
    def __init__(self, main_dir=None):
        super().__init__(main_dir, start_time=0.0, end_time=80.0,
                         desired_start_time_step=0.005, n_max_steps=16000)
        self._problem_name = "DFGBenchmark2D2"
        self._output_frequency = 50
        self._postprocessing_frequency = 50
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

    def set_equation_coefficients(self):
        self._coefficient_handler = EquationCoefficientHandler(Re=100.0)

    def postprocess_solution(self):
        # get pressure, velocity
        pressure = self._get_pressure()
        velocity = self._get_velocity()
        # get cylinder boundary
        cylinder_id = self._boundary_marker_map["cylinder"]
        dA_cyl = dlfn.ds(domain=self._mesh,
                         subdomain_data=self._boundary_markers,
                         subdomain_id=cylinder_id)
        # get normal vector
        n = dlfn.FacetNormal(self._mesh)
        # symmetric gradient
        d = dlfn.Constant(0.5) * (dlfn.grad(velocity) + dlfn.grad(velocity).T)
        # traction vector
        traction = - pressure * n + 1 / self._Re * dlfn.dot(d, n)
        # integrate to get forces
        drag_force = dlfn.assemble(-traction[0] * dA_cyl)
        lift_force = dlfn.assemble(-traction[1] * dA_cyl)
        # calculate coefficients
        drag_coeff = 2.0 * drag_force
        lift_coeff = 2.0 * lift_force
        print(drag_coeff, lift_coeff)
        Coefficients.append([drag_coeff, lift_coeff])


if __name__ == "__main__":
    dfg_benchmark = DFGBenchmark2D2()
    Coefficients = []
    dfg_benchmark.solve_problem()

    Coefficients = np.asarray(Coefficients)
    np.savetxt("results/Coefficients.txt", Coefficients)
