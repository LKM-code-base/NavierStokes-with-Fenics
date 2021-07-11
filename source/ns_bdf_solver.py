#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from bdf_time_stepping import BDFTimeStepping
import dolfin as dlfn
from ns_solver_base import InstationarySolverBase


class ImplicitBDFSolver(InstationarySolverBase):

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
        (v, p) = dlfn.TrialFunctions(self._Wh)
        (w, q) = dlfn.TestFunctions(self._Wh)

        # split solutions
        _, pressure = dlfn.split(self._solutions[0])
        velocity_solutions = []
        for solution in self._solutions:
            velocity_solutions.append(dlfn.split(solution)[0])
        velocity = velocity_solutions[0]

        # volume element
        dV = dlfn.Measure("dx", domain=self._mesh)

        # dimensionless parameters
        assert hasattr(self, "_Re")
        Re = self._Re

        # weak forms
        # mass balance
        F_mass = -self._divergence_term(velocity, q) * dV

        # momentum balance
        F_momentum = (
                        self._acceleration_term(velocity_solutions, w)
                        + self._convective_term(velocity, w)
                        - self._divergence_term(w, pressure)
                        + self._viscous_term(velocity, w) / Re
                        ) * dV

        # add boundary tractions
        F_momentum = self._add_boundary_tractions(F_momentum, w)

        # add body force term
        if hasattr(self, "_body_force"):
            assert hasattr(self, "_Fr"), "Froude number is not specified."
            F_momentum -= dlfn.dot(self._body_force, w) / self._Fr**2 * dV
            
        # add coriolis force term
        if hasattr(self, "_Omega"):
            assert hasattr(self, "_Ro"), "Rossby number is not specified."
            F_momentum += self._coriolis_term(velocity, w) / self._Ro * dV
            
        # add euler force term
        if hasattr(self, "_Alpha"):
            assert hasattr(self, "_Ro"), "Rossby number is not specified."
            x = dlfn.SpatialCoordinate(self._mesh)
            F_momentum += self._euler_term(x, w) / self._Ro * dV

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
