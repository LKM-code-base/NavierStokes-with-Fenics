#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from auxiliary_classes import CustomNonlinearProblem
from bdf_time_stepping import BDFTimeStepping
import dolfin as dlfn
from dolfin import dot
import math
from ns_solver_base import InstationaryNavierStokesSolverBase


class ImplicitBDFNavierStokesSolver(InstationaryNavierStokesSolverBase):
    
    def __init__(self, mesh, boundary_markers, form_convective_term, time_stepping, tol=1e-10, max_iter=50):

        super().__init__(mesh, boundary_markers, form_convective_term, 
                         time_stepping, tol, max_iter)
        assert isinstance(time_stepping, BDFTimeStepping)

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
            accelerations.append(alpha[i] * dot(velocity_solutions[i], w))

        return sum(accelerations) / k

    def _setup_problem(self):
        """
        Method setting up solver objects of the instationary problem.
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
        self._add_boundary_tractions(F_momentum, w)
        # add body force term
        if hasattr(self, "_body_force"):
            assert hasattr(self, "_Fr"), "Froude number is not specified."
            F_momentum -= dot(self._body_force, w) / self._Fr**2 * dV
        
        # joint weak form
        self._F = F_mass + F_momentum

        # linearization using Picard's method
        J_picard_mass = -self._divergence_term(v, q) * dV
        J_picard_momentum = (
                self._alpha[0] * dot(v, w) / self._next_step_size
                + self._picard_linerization_convective_term(velocity, v, w)
                - self._divergence_term(w, p) 
                + self._viscous_term(v, w) / Re) * dV
        self._J_picard = J_picard_mass + J_picard_momentum

        # linearization using Newton's method
        self._J_newton = dlfn.derivative(self._F, self._solutions[0])

        # setup non-linear solver
        linear_solver = dlfn.PETScLUSolver()
        comm = dlfn.MPI.comm_world
        factory = dlfn.PETScFactory.instance()
        self._nonlinear_solver = dlfn.NewtonSolver(comm, linear_solver, factory)
        self._nonlinear_solver.parameters["absolute_tolerance"] = self._tol
        self._nonlinear_solver.parameters["maximum_iterations"] = self._maxiter

        # setup problem with Picard linearization
        self._picard_problem = CustomNonlinearProblem(self._F,
                                                      self._dirichlet_bcs,
                                                      self._J_picard)

        # setup problem with Newton linearization
        self._problem = CustomNonlinearProblem(self._F,
                                               self._dirichlet_bcs,
                                               self._J_newton)
    def _solve_time_step(self):  # pragma: no cover
        """
        Method solving one time step of the non-linear saddle point problem.
        """
        solution = self._solutions[0]
        # compute initial residual
        residual_vector = dlfn.Vector(solution.vector())
        self._picard_problem.F(residual_vector, solution.vector())
        residual = residual_vector.norm("l2")

        # correct initial tolerance if necessary
        # determine order of magnitude
        order = math.floor(math.log10(residual))
        # specify corrected tolerance
        tol_picard = (residual / 10.0**order) * 10.0**(order - 1.0)

        # Picard iteration
        dlfn.info("Starting Picard iteration...")
        self._nonlinear_solver.parameters["maximum_iterations"] = 10
        self._nonlinear_solver.parameters["absolute_tolerance"] = tol_picard
        self._nonlinear_solver.parameters["relative_tolerance"] = 1.0e-1
        self._nonlinear_solver.parameters["error_on_nonconvergence"] = False
        self._nonlinear_solver.solve(self._picard_problem, solution.vector())

        # solve problem
        dlfn.info("Starting Newton iteration...")
        self._nonlinear_solver.parameters["absolute_tolerance"] = self._tol
        self._nonlinear_solver.parameters["maximum_iterations"] = self._maxiter
        self._nonlinear_solver.parameters["relative_tolerance"] = 1.0e1 * self._tol
        self._nonlinear_solver.parameters["error_on_nonconvergence"] = True
        self._nonlinear_solver.solve(self._problem, solution.vector())


    def _update_time_stepping_coefficients(self):
        """Update time stepping coefficients ``_alpha`` and ``_next_step_size``."""
        # update time steps
        next_step_size = self._time_stepping.get_next_step_size()
        if not hasattr(self, "_next_step_size"):
            self._next_step_size = dlfn.Constant(next_step_size)
        else:
            self._next_step_size.assign(next_step_size)
        # update coefficients
        alpha = self._time_stepping.coefficients(1)
        assert len(alpha)
        if not hasattr(self, "_alpha"):
            self._alpha = [dlfn.Constant(alpha[0]), dlfn.Constant(alpha[1]),
                           dlfn.Constant(alpha[2])]
        else:
            for i in range(3):
                self._alpha[i].assign(alpha[i])
