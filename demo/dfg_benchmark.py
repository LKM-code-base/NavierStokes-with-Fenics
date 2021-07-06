#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from grid_generator import channel_with_cylinder
from ns_bdf_solver import ImplicitBDFSolver
from ns_problem import InstationaryProblem
from ns_solver_base import VelocityBCType
import csv

dlfn.set_log_level(40)


class DFGBenchmark2D2(InstationaryProblem):
    def __init__(self, main_dir=None):
        super().__init__(main_dir, start_time=0.0, end_time=60.0,
                         desired_start_time_step=0.005, n_max_steps=12000)

        self._problem_name = "DFGBenchmark2D2"

        self.set_parameters(Re=100.0)

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

    def postprocess_solution(self):
        #get pressure, velocity and kinematic visosity
        vel_mean = 1
        L = 1
        p = self._get_pressure()
        vel = self._get_velocity()
        nu = vel_mean * L / self._Re
        #get cylinder boundary
        ds_Cyl = dlfn.ds(domain=self._mesh, subdomain_data=self._boundary_markers, subdomain_id=104)
        #get normals
        n = dlfn.FacetNormal(self._mesh)
        #stress term
        force = - p * n + nu * dlfn.dot(dlfn.grad(vel), n)
        #integrate to get forces
        F_D = dlfn.assemble(-force[0] * ds_Cyl)
        F_L = dlfn.assemble(-force[1] * ds_Cyl)
        #calc coefficients
        C_D = 2 / (vel_mean ** 2 * L) * F_D
        C_L = 2 / (vel_mean ** 2 * L) * F_L
        print(C_D, C_L)
        coefficients = [C_D, C_L]
        Coefficients.append(coefficients)        


if __name__ == "__main__":
    dfg_benchmark = DFGBenchmark2D2()
    Coefficients = []
    dfg_benchmark.solve_problem()
    
    f = open('results/Coefficients.csv', 'w')
    writer = csv.writer(f)
    for row in Coefficients:
        writer.writerow(row)
    f.close()    
