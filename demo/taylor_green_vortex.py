#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from grid_generator import hyper_cube
from grid_generator import HyperCubeBoundaryMarkers
from ns_bdf_solver import ImplicitBDFSolver
from ns_problem import InstationaryProblem

dlfn.set_log_level(40)

Re = 100.0
gamma = 2.0 * dlfn.pi

class TaylorGreenVortex(InstationaryProblem):
    def __init__(self, main_dir=None):
        super().__init__(main_dir, start_time=0.0, end_time=1.0,
                         desired_start_time_step=0.001, n_max_steps=50)

        self._problem_name = "TaylorGreenVortex"

        self.set_parameters(Re=Re)

        self._n_points = 50
        self._output_frequency = 10
        self._postprocessing_frequency = 10

        self.set_solver_class(ImplicitBDFSolver)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_initial_conditions(self):
        self._initial_conditions = dict()
        self._initial_conditions["velocity"] = \
            dlfn.Expression(("cos(gamma * x[0]) * sin(gamma * x[1])",
                             "-sin(gamma * x[0]) * cos(gamma * x[1])"), gamma=gamma,
                            degree=3)
        self._initial_conditions["pressure"] = \
            dlfn.Expression("-1.0/4.0 * (cos(2.0 * gamma * x[0]) + cos(2.0 * gamma * x[1]))",
                            gamma=gamma, degree=3)

    def set_periodic_boundary_conditions(self):
        """Set periodic boundary conditions in x- and y-direction."""
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

        self._periodic_bcs = PeriodicDomain()
        self._periodic_boundary_ids = (HyperCubeBoundaryMarkers.left.value,
                                       HyperCubeBoundaryMarkers.right.value)

    def postprocess_solution(self):
        # add pressure gradient to the field output
        self._add_to_field_output(self._compute_pressure_gradient())
        # add vorticity to the field output
        self._add_to_field_output(self._compute_vorticity())


if __name__ == "__main__":
    taylor_green = TaylorGreenVortex()
    taylor_green.solve_problem()
