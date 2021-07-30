#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from bdf_time_stepping import BDFTimeStepping
from dolfin import div, dot, grad
from ns_solver_base import InstationarySolverBase


class IPCSSolver(InstationarySolverBase):
    _required_objects = ("_diffusion_solver", "_projection_solver",
                         "_velocity_correction_solver")

    def __init__(self, mesh, boundary_markers, form_convective_term, time_stepping, tol=1e-10, max_iter=50):
        # input check
        assert isinstance(time_stepping, BDFTimeStepping)
        super().__init__(mesh, boundary_markers, form_convective_term,
                         time_stepping, tol, max_iter)

    def _acceleration_term(self, velocity_solutions, w):
        # input check
        assert isinstance(velocity_solutions, (list, tuple))
        assert all(isinstance(x, self._form_function_types) for x in velocity_solutions)
        assert isinstance(w, self._form_function_types)
        # step size
        k = self._next_step_size
        # time stepping coefficients
        alpha = self._alpha
        assert len(alpha) == len(velocity_solutions)
        # compute accelerations
        accelerations = []
        for i in range(len(alpha)):
            accelerations.append(alpha[i] * dlfn.dot(velocity_solutions[i], w))
        return sum(accelerations) / k

    def _advance_solution(self):
        """Advance solution objects in time."""
        assert hasattr(self, "_velocities")
        assert hasattr(self, "_intermediate_velocity")
        assert hasattr(self, "_pressure")
        assert hasattr(self, "_old_pressure")
        for i in range(len(self._velocities), 1, -1):
            self._velocities[i-1].assign(self._velocities[i-2])
        self._old_pressure.assign(self._pressure)

    def _setup_boundary_conditions(self):
        # split boundary conditions
        self._dirichlet_bcs = dict()
        self._dirichlet_bcs["velocity"] = []
        self._dirichlet_bcs["pressure"] = []
        # velocity part
        if hasattr(self, "_velocity_bcs"):
            velocity_space = self._get_subspace("velocity")
            self._setup_velocity_boundary_conditions(self._dirichlet_bcs["velocity"],
                                                     self._velocity_bcs,
                                                     velocity_space)
        # pressure part
        if hasattr(self, "_pressure_bcs"):
            pressure_space = self._get_subspace("pressure")
            self._setup_pressure_boundary_conditions(self._dirichlet_bcs["pressure"],
                                                     self._pressure_bcs,
                                                     pressure_space)
        if len(self._dirichlet_bcs["velocity"]) == 0 and \
                len(self._dirichlet_bcs["pressure"]) == 0:  # pragma: no cover
            assert hasattr(self, "_constrained_domain")

    def _setup_function_spaces(self):
        """
        Class method setting up function spaces.
        """
        # create joint function space
        if not hasattr(self, "_Wh"):
            super()._setup_function_spaces()
        # create subspaces
        WhSub = self._get_subspaces()
        # create separate solutions
        self._velocities = []
        for i in range(self._time_stepping.n_levels() + 1):
            name = i * "old" + (i > 0) * "_" + "velocity"
            self._velocities.append(dlfn.Function(WhSub["velocity"], name=name))
        self._intermediate_velocity = dlfn.Function(WhSub["velocity"], name="intermediate_velocity")
        self._pressure = dlfn.Function(WhSub["pressure"], name="pressure")
        self._old_pressure = dlfn.Function(WhSub["pressure"], name="old_pressure")

    def _setup_problem(self):
        """Method setting up solvers object of the instationary problem."""
        assert hasattr(self, "_mesh")
        assert hasattr(self, "_boundary_markers")
        if not all(hasattr(self, attr) for attr in ("_Wh", "_solutions",
                                                    "_intermediate_velocity",
                                                    "_velocities",
                                                    "_pressure", "_old_pressure")):  # pragma: no cover
            self._setup_function_spaces()

        if not all(hasattr(self, attr) for attr in ("_next_step_size",
                                                    "_alpha")):
            self._update_time_stepping_coefficients()

        self._setup_boundary_conditions()
        # setup the diffusion step
        self._setup_diffusion_step()
        # setup the projection step
        self._setup_projection_step()
        # setup the correction step
        self._setup_correction_step()

    def _setup_diffusion_step(self):
        """Method setting up solver object of the diffusion step."""
        assert hasattr(self, "_intermediate_velocity")
        assert hasattr(self, "_old_pressure")
        assert hasattr(self, "_velocities")
        # creating test and trial functions
        Vh = self._get_subspace("velocity")
        w = dlfn.TestFunction(Vh)
        # volume element
        dV = dlfn.Measure("dx", domain=self._mesh)
        # sought-for velocity
        velocity = self._intermediate_velocity
        # velocities used in acceleration term
        velocities = []
        velocities.append(velocity)
        for i in range(1, self._time_stepping.n_levels() + 1):
            velocities.append(self._velocities[i])
        # volume element
        dV = dlfn.Measure("dx", domain=self._mesh)
        # momentum equation
        self._F = (self._acceleration_term(velocities, w)
                   + self._convective_term(velocity, w)
                   - self._divergence_term(w, self._old_pressure)
                   + self._viscous_term(velocity, w)
                   ) * dV
        # add boundary tractions
        self._F = self._add_boundary_tractions(self._F, w)
        # add body force term
        self._F = self._add_body_forces(self._F, w)
        # linearization using Newton's method
        self._J_newton = dlfn.derivative(self._F, velocity)
        # setup problem with Newton linearization
        self._diffusion_problem = dlfn.NonlinearVariationalProblem(self._F,
                                                                   velocity,
                                                                   self._dirichlet_bcs["velocity"],
                                                                   self._J_newton)
        # setup non-linear solver
        self._diffusion_solver = dlfn.NonlinearVariationalSolver(self._diffusion_problem)
        self._diffusion_solver.parameters["newton_solver"]["absolute_tolerance"] = self._tol
        self._diffusion_solver.parameters["newton_solver"]["maximum_iterations"] = self._maxiter
        self._diffusion_solver.parameters["newton_solver"]["relative_tolerance"] = 1.0e1 * self._tol
        self._diffusion_solver.parameters["newton_solver"]["error_on_nonconvergence"] = True

    def _setup_projection_step(self):
        """Method setting up solver object of the projection step."""
        assert hasattr(self, "_intermediate_velocity")
        assert hasattr(self, "_old_pressure")
        # creating test and trial functions
        Vh = self._get_subspace("pressure")
        p = dlfn.TrialFunction(Vh)
        q = dlfn.TestFunction(Vh)
        # volume element
        dV = dlfn.Measure("dx", domain=self._mesh)
        # pressure projection equation
        self._pressure_correction_lhs = dot(grad(p), grad(q)) * dV
        self._pressure_correction_rhs = (dot(grad(self._old_pressure), grad(q))
                                         - self._alpha[0] / self._next_step_size *
                                         dot(div(self._intermediate_velocity), q)
                                         ) * dV
        # setup linear problem
        self._projection_problem = dlfn.LinearVariationalProblem(self._pressure_correction_lhs,
                                                                 self._pressure_correction_rhs,
                                                                 self._pressure,
                                                                 self._dirichlet_bcs["pressure"])
        # setup linear solver
        self._projection_solver = dlfn.LinearVariationalSolver(self._projection_problem)

    def _setup_correction_step(self):
        """Method setting up solver object of the correction step."""
        # volume element
        dV = dlfn.Measure("dx", domain=self._mesh)
        # velocity correction step
        # creating test and trial functions
        Vh = self._get_subspace("velocity")
        v = dlfn.TrialFunction(Vh)
        w = dlfn.TestFunction(Vh)
        # velocity correction equation
        self._velocity_correction_lhs = dot(v, w) * dV
        self._velocity_correction_rhs = (dot(self._intermediate_velocity, w)
                                         - (self._next_step_size / self._alpha[0]) *
                                         dot(grad(self._pressure-self._old_pressure), w)
                                         ) * dV
        # setup linear problem
        self._velocity_correction_problem = \
            dlfn.LinearVariationalProblem(self._velocity_correction_lhs,
                                          self._velocity_correction_rhs,
                                          self._velocities[0],
                                          self._dirichlet_bcs["velocity"])
        # setup linear solver
        self._velocity_correction_solver = \
            dlfn.LinearVariationalSolver(self._velocity_correction_problem)

    def _solve_time_step(self):
        """Solves the nonlinear problem for one time step."""
        dlfn.info("Solving diffusion step...")
        dlfn.info("Starting Newton iteration...")
        self._diffusion_solver.solve()

        dlfn.info("Solving projection step...")
        self._projection_solver.solve()

        dlfn.info("Solving velocity correction step...")
        self._velocity_correction_solver.solve()

    def _update_time_stepping_coefficients(self):
        """Update time stepping coefficients ``_alpha`` and ``_next_step_size``."""
        # update time steps
        next_step_size = self._time_stepping.get_next_step_size()
        if not hasattr(self, "_next_step_size"):
            self._next_step_size = dlfn.Constant(next_step_size)
        else:
            self._next_step_size.assign(next_step_size)

        # update coefficients
        alpha = self._time_stepping.coefficients(derivative=1)
        assert len(alpha) == 3
        if not hasattr(self, "_alpha"):
            self._alpha = [dlfn.Constant(alpha[0]), dlfn.Constant(alpha[1]),
                           dlfn.Constant(alpha[2])]
        else:
            for i in range(3):
                self._alpha[i].assign(alpha[i])

    def set_initial_conditions(self, initial_conditions):
        super().set_initial_conditions(initial_conditions)
        assert all(hasattr(self, x) for x in ("_velocities", "_intermediate_velocity",
                                              "_pressure", "_old_pressure"))
        velocity, pressure = self._solutions[0].split()
        old_velocity, old_pressure = self._solutions[1].split()
        self._assign_function(self._pressure, pressure)
        self._assign_function(self._old_pressure, old_pressure)
        self._assign_function(self._velocities[0], velocity)
        self._assign_function(self._velocities[1], old_velocity)

    @property
    def solution(self):
        WhSub = self._get_subspaces()
        assert self._velocities[0] in WhSub["velocity"]
        assert self._pressure in WhSub["pressure"]
        self._assign_function(self._solutions[0], {"velocity": self._velocities[0],
                                                   "pressure": self._pressure})
        return self._solutions[0]
