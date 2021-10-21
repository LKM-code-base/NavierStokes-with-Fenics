#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from bdf_time_stepping import BDFTimeStepping
import dolfin as dlfn
from ns_solver_base import InstationarySolverBase


class ImplicitBDFSolver(InstationarySolverBase):
    _required_objects = ("_solver")

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
            accelerations.append(alpha[i] * velocity_solutions[i])

        return dlfn.dot(sum(accelerations), w) / k

    def _setup_problem(self):
        """Method setting up non-linear solver object of the instationary
        problem.
        """
        assert hasattr(self, "_mesh")
        assert hasattr(self, "_boundary_markers")

        if not all(hasattr(self, attr) for attr in ("_Wh",
                                                    "_solutions")):  # pragma: no cover
            self._setup_function_spaces()

        if not all(hasattr(self, attr) for attr in ("_next_step_size",
                                                    "_alpha")):
            self._update_time_stepping_coefficients()

        self._setup_boundary_conditions()

        # creating test and trial functions
        (w, q) = dlfn.TestFunctions(self._Wh)

        # split solutions
        velocity, pressure = dlfn.split(self._solutions[0])
        velocity_solutions = []
        velocity_solutions.append(velocity)
        index = self._field_association["velocity"]
        for i in range(1, len(self._solutions)):
            velocity_solutions.append(dlfn.split(self._solutions[i])[index])

        # volume element
        dV = dlfn.Measure("dx", domain=self._mesh)
        # weak forms
        # mass balance
        F_mass = -self._divergence_term(velocity, q) * dV
        # momentum balance
        F_momentum = (self._acceleration_term(velocity_solutions, w)
                      + self._convective_term(velocity, w)
                      - self._divergence_term(w, pressure)
                      + self._viscous_term(velocity, w)
                      ) * dV
        # add boundary tractions
        F_momentum = self._add_boundary_tractions(F_momentum, w)
        # add body force term
        F_momentum = self._add_body_forces(F_momentum, w)
        # add Coriolis acceleration
        F_momentum = self._add_coriolis_acceleration(F_momentum, velocity, w)
        # add Euler acceleration
        F_momentum = self._add_euler_acceleration(F_momentum, w)

        # joint weak form
        self._F = F_mass + F_momentum

        # linearization using Newton's method
        self._J_newton = dlfn.derivative(self._F, self._solutions[0])

        # setup problem with Newton linearization
        self._problem = dlfn.NonlinearVariationalProblem(self._F,
                                                         self._solutions[0],
                                                         self._dirichlet_bcs,
                                                         self._J_newton)
        # setup non-linear solver
        self._solver = dlfn.NonlinearVariationalSolver(self._problem)
        self._solver.parameters["newton_solver"]["absolute_tolerance"] = self._tol
        self._solver.parameters["newton_solver"]["maximum_iterations"] = self._maxiter
        self._solver.parameters["newton_solver"]["relative_tolerance"] = 1.0e1 * self._tol
        self._solver.parameters["newton_solver"]["error_on_nonconvergence"] = True

    def _solve_time_step(self):
        """Solves the nonlinear problem for one time step."""
        # solve problem
        dlfn.info("Starting Newton iteration...")
        self._solver.solve()

    def _update_time_stepping_coefficients(self):
        """Update time stepping coefficients ``_alpha`` and ``_next_step_size``."""
        # update time steps
        next_step_size = self._time_stepping.get_next_step_size()
        if not hasattr(self, "_next_step_size"):
            self._next_step_size = dlfn.Constant(next_step_size, name="dt")
        else:
            self._next_step_size.assign(next_step_size)

        # update coefficients
        alpha = self._time_stepping.coefficients(derivative=1)
        assert len(alpha) == 3
        if not hasattr(self, "_alpha"):
            self._alpha = [dlfn.Constant(alpha[0], name="alpha00"),
                           dlfn.Constant(alpha[1], name="alpha01"),
                           dlfn.Constant(alpha[2], name="alpha02")]
        else:
            for i in range(3):
                self._alpha[i].assign(alpha[i])
