#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
import ufl
from theta_time_stepping import GeneralThetaTimeStepping
from dolfin import dot
from ns_solver_base import InstationarySolverBase, TractionBCType


class ImplicitThetaSolver(InstationarySolverBase):
    def __init__(self, mesh, boundary_markers, form_convective_term, time_stepping, tol=1e-10, max_iter=50):

        super().__init__(mesh, boundary_markers, form_convective_term,
                         time_stepping, tol, max_iter)
        assert isinstance(time_stepping, GeneralThetaTimeStepping)

    def _acceleration_term(self, velocity_solutions, w):
        # input check
        assert isinstance(velocity_solutions, (list, tuple))
        assert len(velocity_solutions) == 2
        assert all(isinstance(x, self._form_function_types) for x in velocity_solutions)
        assert isinstance(w, self._form_function_types)
        # step size
        k = self._next_step_size
        # compute accelerations
        return dot(velocity_solutions[0] - velocity_solutions[1], w) / k

    def _advance_solution(self):
        """Advance solution objects in time."""
        assert hasattr(self, "_solutions")
        for i in range(len(self._solutions) - 1):
            self._solutions[i+1].assign(self._solutions[i])

    def _add_boundary_tractions(self, F, w):
        """Method adding boundary traction terms to the weak form"""
        # input check
        assert isinstance(F, ufl.form.Form)
        assert isinstance(w, self._form_trial_function_types)

        dA = dlfn.Measure("ds", domain=self._mesh, subdomain_data=self._boundary_markers)

        if hasattr(self, "_traction_bcs") and not hasattr(self, "_current_traction_bcs"):
            self._current_traction_bcs = []
            for bc in self._traction_bcs:
                # unpack values
                if len(bc) == 3:
                    bc_type, bndry_id, traction = bc
                    current_traction = type(traction)(traction)
                    self.modify_time(current_traction, self._current_time)
                    self._current_traction_bcs.append((bc_type, bndry_id, current_traction))
                elif len(bc) == 4:
                    bc_type, bndry_id, component_index, traction = bc
                    current_traction = type(traction)(traction)
                    self.modify_time(current_traction, self._current_time)
                    self._current_traction_bcs.append((bc_type, bndry_id, component_index, current_traction))
                else:  # pragma: no cover
                    raise RuntimeError()

        if hasattr(self, "_traction_bcs"):
            for bc in self._traction_bcs:
                # unpack values
                if len(bc) == 3:
                    bc_type, bndry_id, traction = bc
                elif len(bc) == 4:
                    bc_type, bndry_id, component_index, traction = bc
                else:  # pragma: no cover
                    raise RuntimeError()

                if bc_type is TractionBCType.constant:
                    assert isinstance(traction, (tuple, list))
                    const_function = dlfn.Constant(traction)
                    F += self._theta[2] * dot(const_function, w) * dA(bndry_id)

                elif bc_type is TractionBCType.constant_component:
                    assert isinstance(traction, float)
                    const_function = dlfn.Constant(traction)
                    F += self._theta[2] * const_function * w[component_index] * dA(bndry_id)

                elif bc_type is TractionBCType.function:
                    assert isinstance(traction, dlfn.Expression)
                    F += self._theta[2] * dot(traction, w) * dA(bndry_id)

                elif bc_type is TractionBCType.function_component:
                    assert isinstance(traction, dlfn.Expression)
                    F += self._theta[2] * traction * w[component_index] * dA(bndry_id)

            for bc in self._current_traction_bcs:
                # unpack values
                if len(bc) == 3:
                    bc_type, bndry_id, traction = bc
                elif len(bc) == 4:
                    bc_type, bndry_id, component_index, traction = bc
                else:  # pragma: no cover
                    raise RuntimeError()

                if bc_type is TractionBCType.constant:
                    assert isinstance(traction, (tuple, list))
                    const_function = dlfn.Constant(traction)
                    F += self._theta[2] * dot(const_function, w) * dA(bndry_id)

                elif bc_type is TractionBCType.constant_component:
                    assert isinstance(traction, float)
                    const_function = dlfn.Constant(traction)
                    F += self._theta[2] * const_function * w[component_index] * dA(bndry_id)

                elif bc_type is TractionBCType.function:
                    assert isinstance(traction, dlfn.Expression)
                    F += self._theta[2] * dot(traction, w) * dA(bndry_id)

                elif bc_type is TractionBCType.function_component:
                    assert isinstance(traction, dlfn.Expression)
                    F += self._theta[2] * traction * w[component_index] * dA(bndry_id)

        return F

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
        velocity, pressure = dlfn.split(self._solutions[0])
        previous_velocity, _ = dlfn.split(self._solutions[1])
        velocity_solutions = []
        velocity_solutions.append(velocity)
        velocity_solutions.append(previous_velocity)

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
                        + self._theta[0] * (self._convective_term(velocity, w) +
                                            self._viscous_term(velocity, w) / Re)
                        + self._theta[1] * (self._convective_term(previous_velocity, w) +
                                            self._viscous_term(previous_velocity, w) / Re)
                        - self._intermediate_step_size / self._next_step_size * self._divergence_term(w, pressure)
                        ) * dV
        # add boundary tractions
        self._add_boundary_tractions(F_momentum, w)

        # add body force term
        if hasattr(self, "_body_force"):
            assert hasattr(self, "_Fr"), "Froude number is not specified."
            if not hasattr(self, "_current_body_force"):
                # copy object of the body force at previous time level by
                # calling a copy constructor
                self._current_body_force = type(self._body_force)(self._body_force)
                self.modify_time(self._current_body_force, self._current_time)

            F_momentum -= self._theta[2] * dot(self._body_force, w) / self._Fr**2 * dV
            F_momentum -= self._theta[3] * dot(self._current_body_force, w) / self._Fr**2 * dV

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

    def _solve_time_step(self):  # pragma: no cover
        """
        Method solving one time step of the non-linear saddle point problem.
        """
        # solve problem
        dlfn.info("Starting Newton iteration...")
        self._solver.solve()

    def _update_theta(self, step):
        theta = self._time_stepping.theta[step]
        assert len(theta) == 4
        if not hasattr(self, "_theta"):
            self._theta = [dlfn.Constant(theta[0]), dlfn.Constant(theta[1]),
                           dlfn.Constant(theta[2]), dlfn.Constant(theta[3])]
        else:
            for i in range(3):
                self._theta[i].assign(theta[i])

    def _update_intermediate_times(self, step):
        intermediate_times = self._time_stepping.intermediate_times[step]
        current_time = intermediate_times[0]
        next_time = intermediate_times[1]

    def _update_intermediate_timesteps(self, step):
        intermediate_timestep = self._time_stepping.intermediate_timesteps[step]
        if not hasattr(self, "_intermediate_step_size"):
            self._intermediate_step_size = dlfn.Constant(intermediate_timestep)
        else:
            self._intermediate_step_size.assign(intermediate_timestep)

    def _update_time_stepping_coefficients(self):
        """Update time stepping coefficients ``_alpha`` and ``_next_step_size``."""
        # update time steps
        next_step_size = self._time_stepping.get_next_step_size()
        if not hasattr(self, "_next_step_size"):
            self._next_step_size = dlfn.Constant(next_step_size)
        else:
            self._next_step_size.assign(next_step_size)
