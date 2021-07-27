#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from grid_generator import hyper_cube
from grid_generator import HyperCubeBoundaryMarkers
from math import ceil
import numpy as np
from ns_bdf_solver import ImplicitBDFSolver
from ns_solver_base import PressureBCType
from ns_problem import InstationaryProblem

dlfn.set_log_level(30)

Re = 100.0
gamma = 2.0 * dlfn.pi

probes = {}
for key in ("velocity", "pressure"):
    probes[key] = np.load(f"probe_{key}.npy")


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


class TaylorGreenVortex(InstationaryProblem):
    def __init__(self, time_step, main_dir=None):
        end_time = 0.05
        n_max_steps = ceil(end_time / time_step)
        super().__init__(main_dir, start_time=0.0, end_time=end_time,
                         desired_start_time_step=time_step, n_max_steps=n_max_steps)
        self._problem_name = "TaylorGreenVortex"
        self.set_parameters(Re=Re)
        self._n_points = None
        self._output_frequency = 0
        self._postprocessing_frequency = 1
        self.set_solver_class(ImplicitBDFSolver)

    @property
    def n_points(self):
        return self._n_points

    @n_points.setter
    def n_points(self, n):
        assert isinstance(n, int)
        assert n > 0
        self._n_points = n

    def setup_mesh(self):
        assert self._n_points is not None
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_initial_conditions(self):
        self._initial_conditions = dict()
        self._initial_conditions["velocity"] = \
            dlfn.Expression(("cos(gamma * x[0]) * sin(gamma * x[1])",
                             "-sin(gamma * x[0]) * cos(gamma * x[1])"),
                            gamma=gamma, degree=3)
        self._initial_conditions["pressure"] = \
            dlfn.Expression("-1.0/4.0 * (cos(2.0 * gamma * x[0]) + cos(2.0 * gamma * x[1]))",
                            gamma=gamma, degree=3)

    def set_boundary_conditions(self):
        # pressure mean value constraint
        self._bcs = ((PressureBCType.mean_value , None, 0.0), )

    def set_periodic_boundary_conditions(self):
        """Set periodic boundary conditions in x- and y-direction."""
        self._periodic_bcs = PeriodicDomain()
        self._periodic_boundary_ids = (HyperCubeBoundaryMarkers.left.value,
                                       HyperCubeBoundaryMarkers.right.value,
                                       HyperCubeBoundaryMarkers.top.value,
                                       HyperCubeBoundaryMarkers.bottom.value)

#    def compute_error(self):
#        # current time
#        assert self._time_stepping.is_at_end()
#        assert self._time_stepping.current_time == self._time_stepping.end_time
#        current_time = self._time_stepping.current_time
#        
#        # get velocity and pressure
#        velocity = self._get_velocity()
#        pressure = self._get_pressure()
#        
#        # exact solutions
#        exact_solution = dict()
#        exact_solution["velocity"] = \
#            dlfn.Expression(("exp(-2.0 * gamma * gamma / Re * t) * cos(gamma * x[0]) * sin(gamma * x[1])",
#                             "-exp(-2.0 * gamma * gamma / Re * t) * sin(gamma * x[0]) * cos(gamma * x[1])"),
#                            gamma=gamma, Re=Re, t=current_time, degree=5)
#        exact_solution["pressure"] = \
#        dlfn.Expression("-1.0/4.0 * exp(-4.0 * gamma * gamma / Re * t) * (cos(2.0 * gamma * x[0]) + cos(2.0 * gamma * x[1]))",
#                        gamma=gamma, Re=Re, t=current_time, degree=5)
#        errors["velocity"].append(dlfn.errornorm(exact_solution["velocity"], velocity))
#        errors["pressure"].append(dlfn.errornorm(exact_solution["pressure"], pressure))

    def preprocess_solution(self):
        # get velocity and pressure
        velocity = self._get_velocity()
        pressure = self._get_pressure()
        
        print("velocity(x = 0.1, y = 0.1) = ", velocity(0.1, 0.1))
        print("velocity reference value at (x = 0.1, y = 0.1) = ",
              probes["velocity"][0])
        print("velocity difference at (x = 0.1, y = 0.1) = ",
              np.linalg.norm(velocity(0.1, 0.1) - probes["velocity"][0]))
        print("pressure(x = 0.1, y = 0.1) = ", pressure(0.1, 0.1))
        print("pressure reference value at (x = 0.1, y = 0.1) = ",
              probes["pressure"][0])
        print("pressure difference at (x = 0.1, y = 0.1) = ",
              pressure(0.1, 0.1) - probes["pressure"][0])

    
    def postprocess_solution(self):
        # get velocity and pressure
        velocity = self._get_velocity()
        pressure = self._get_pressure()
        
        step_number = self._time_stepping.step_number
        print("velocity(x = 0.1, y = 0.1) = ", velocity(0.1, 0.1))
        print("velocity difference at (x = 0.1, y = 0.1) = ",
              np.linalg.norm(velocity(0.1, 0.1) - probes["velocity"][step_number+1]))
        print("pressure(x = 0.1, y = 0.1) = ", pressure(0.1, 0.1))
        print("pressure difference at (x = 0.1, y = 0.1) = ",
              pressure(0.1, 0.1) - probes["pressure"][step_number+1])

if __name__ == "__main__":
    errors = {"pressure": [], "velocity": []}
    initial_time_step = 1e-2
    time_step_reduction_factor = 0.5
    n_levels = 1
    for i in range(n_levels):
        time_step = initial_time_step * time_step_reduction_factor**i
        taylor_green = TaylorGreenVortex(time_step )
        taylor_green.n_points = 64
        taylor_green.solve_problem()
#        taylor_green.compute_error()
        print(errors)
