#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from bdf_time_stepping import BDFTimeStepping
import dolfin as dlfn
from ns_solver_base import SolverBase, InstationarySolverBase


class IPCSSolver(InstationarySolverBase):

    def __init__(self, mesh, boundary_markers, form_convective_term, time_stepping, tol=1e-10, max_iter=50):

        # input check
        assert isinstance(time_stepping, BDFTimeStepping)

        super().__init__(mesh, boundary_markers, form_convective_term,
                         time_stepping, tol, max_iter)

    def _setup_function_spaces(self):
        """
        Class method setting up function spaces.
        """
        # create joint function space
        SolverBase._setup_function_spaces(self)

        # create subspaces
        self._Vh = dict()
        for key, index in self._field_association.items():
            space = dlfn.FunctionSpace(self._Wh.mesh(),
                                       self._Wh.sub(index).ufl_element())
            self._Vh[key] = space

        # create joint solution
        self._joint_solution = dlfn.Function(self._Wh)

        # create separate solutions
        self._velocities = [dlfn.Function(self._Vh["velocity"]) for _ in range(self._time_stepping.n_levels() + 1)]
        self._intermediate_velocity = dlfn.Function(self._Vh["velocity"])
        self._pressure = dlfn.Function(self._Vh["pressure"])
        self._phi = dlfn.Function(self._Vh["pressure"])

    def _setup_problem(self):
        """Method setting up solvers object of the instationary problem.
        """
        assert hasattr(self, "_mesh")
        assert hasattr(self, "_boundary_markers")

        if not all(hasattr(self, attr) for attr in ("_Wh", "_Vh"
                                                    "_joint_solution",
                                                    "_velocities",
                                                    "_pressure", "_phi")):  # pragma: no cover
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
        assert hasattr(self, "_Vh")
        assert hasattr(self, "_intermediate_velocity")
        assert hasattr(self, "_pressure")
        assert hasattr(self, "_velocities")

        # creating test and trial functions
        Vh = self._Vh["velocity"]
        w = dlfn.TestFunction(Vh)
        # volume element
        dV = dlfn.Measure("dx", domain=self._mesh)

        # dimensionless parameters
        assert hasattr(self, "_Re")
        Re = self._Re

        # sought-for velocity
        velocity = self._intermediate_velocity

        # velocities used in acceleration term
        velocities = []
        velocities[0] = velocity
        for i in range(1, self._time_stepping.n_levels() + 1):
            velocities.append(self._velocities[i])

        # momentum equation
        self._F = (
                    self._acceleration_term(velocities, w)
                    + self._convective_term(velocity, w)
                    - self._divergence_term(w, self._pressure)
                    + self._viscous_term(velocity, w) / Re
                    ) * dV

        # add boundary tractions
        self._F = self._add_boundary_tractions(self._F, w)

        # add body force term
        if hasattr(self, "_body_force"):
            assert hasattr(self, "_Fr"), "Froude number is not specified."
            self._F -= dlfn.dot(self._body_force, w) / self._Fr**2 * dV

        # linearization using Newton's method
        self._J_newton = dlfn.derivative(self._F, velocity)

        # setup problem with Newton linearization
        self._diffusion_problem = dlfn.NonlinearVariationalProblem(self._F,
                                                                   velocity,
                                                                   self._dirichlet_bcs,
                                                                   self._J_newton)
        # setup non-linear solver
        self._diffusion_solver = dlfn.NonlinearVariationalSolver(self._problem)
        self._diffusion_solver.parameters["newton_solver"]["absolute_tolerance"] = self._tol
        self._diffusion_solver.parameters["newton_solver"]["maximum_iterations"] = self._maxiter
        self._diffusion_solver.parameters["newton_solver"]["relative_tolerance"] = 1.0e1 * self._tol
        self._diffusion_solver.parameters["newton_solver"]["error_on_nonconvergence"] = True

    def _setup_projection_step(self):
        """Method setting up solver object of the projection step."""
        assert hasattr(self, "_Vh")
        assert hasattr(self, "_intermediate_velocity")
        assert hasattr(self, "_pressure")

        # creating test and trial functions
        Vh = self._Vh["pressure"]
        p = dlfn.TrialFunction(Vh)
        q = dlfn.TestFunction(Vh)

        # volume element
        dV = dlfn.Measure("dx", domain=self._mesh)

        # pressure projection equation
        self._projection_lhs = dlfn.dot(dlfn.grad(p), dlfn.grad(q)) * dV
        self._projection_rhs = dlfn.div(self._intermediate_velocity) * q * dV

        # setup linear problem
        self._projection_problem = dlfn.LinearVariationalProblem(self._projection_lhs,
                                                                 self._projection_rhs,
                                                                 self._phi,
                                                                 self._dirichlet_bcs_phi)
        # setup linear solver
        self._projection_solver = dlfn.LinearVariationalSolver(self._projection_problem)

    def _setup_correction_step(self):
        """Method setting up solver object of the correction step."""

        # volume element
        dV = dlfn.Measure("dx", domain=self._mesh)

        # velocity correction step
        # creating test and trial functions
        Vh = self._Vh["velocity"]
        v = dlfn.TrialFunction(Vh)
        w = dlfn.TestFunction(Vh)

        # velocity correction equation
        self._velocity_correction_lhs = dlfn.dot(velocities[0], v) * dV
        self._velocity_correction_rhs = (dlfn.dot(self._intermediate_velocity, v) - (self._next_step_size / self._alpha[0]) * dlfn.dot(dlfn.grad(self._phi)), v) * dV

        # setup linear problem
        self._velocity_correction_problem = \
            dlfn.LinearVariationalProblem(self._velocity_correction_lhs,
                                          self._velocity_correction_rhs,
                                          self._velocities[0],
                                          self._dirichlet_bcs_velocity)
        # setup linear solver
        self._velocity_correction_solver = \
            dlfn.LinearVariationalSolver(self._velocity_correction_problem)

        # pressure correction step
        # creating test and trial functions
        Vh = self._Vh["pressure"]
        p = dlfn.TrialFunction(Vh)
        q = dlfn.TestFunction(Vh)

        # pressure correction equation
        self._pressure_correction_lhs = dlfn.dot(dlfn.grad(self._phi), dlfn.grad(q)) * dV
        self._pressure_correction_rhs = (-self._alpha[0]/self._next_step_size) * dlfn.dot(dlfn.div(self._intermediate_velocity), q) * dV
        # setup linear problem
        self._pressure_correction_problem = \
            dlfn.LinearVariationalProblem(self._pressure_correction_lhs,
                                          self._pressure_correction_rhs,
                                          self._pressure,
                                          self._dirichlet_bcs_pressure)
        # setup linear solver
        self._pressure_correction_solver = \
            dlfn.LinearVariationalSolver(self._pressure_correction_problem)

    def _solve_time_step(self):
        """Solves the nonlinear problem for one time step."""
        dlfn.info("Solving diffusion step...")
        dlfn.info("Starting Newton iteration...")
        self._diffusion_solver.solve()

        dlfn.info("Solving projection step...")
        self._projection_solver.solve()

        dlfn.info("Solving velocity correction step...")
        self._velocity_correction_solver.solve()

        dlfn.info("Solving pressure correction step...")
        self._pressure_correction_solver.solve()

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
