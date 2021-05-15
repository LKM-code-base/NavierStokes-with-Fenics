#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from enum import Enum, auto

import numpy as np

import math

import dolfin as dlfn
from dolfin import grad, div, dot, inner

from auxiliary_modules import CustomNonlinearProblem


class VelocityBCType(Enum):
    no_slip = auto()
    no_normal_flux = auto()
    no_tangential_flux = auto()
    constant = auto()
    function = auto()


class PressureBCType(Enum):
    constant = auto()
    function = auto()
    mean_value = auto()
    none = auto()


class TractionBCType(Enum):
    constant = auto()
    function = auto()
    free = auto()


class SpatialDiscretizationConvectiveTerm(Enum):
    standard = auto()


class StationaryNavierStokesSolver():
    """
    Class to simulate stationary fluid flow of an incompressible fluid using
    P2-P1 finite elements. The system is solved hybrid Picard-Newton iteration.

    Parameters
    ----------
    """
    # class variables
    _sub_space_association = {0: "velocity", 1: "pressure"}
    _field_association = {value: key for key, value in _sub_space_association.items()}
    _apply_body_force = False
    _body_force_specified = False
    _apply_boundary_traction = False
    _null_scalar = dlfn.Constant(0.)

    def __init__(self, mesh, boundary_markers, tol=1e-10, maxiter=50,
                 tol_picard=1e-2, maxiter_picard=10):

        # input check
        assert isinstance(mesh, dlfn.Mesh)
        assert isinstance(boundary_markers, (dlfn.cpp.mesh.MeshFunctionSizet,
                                             dlfn.cpp.mesh.MeshFunctionInt))
        assert all(isinstance(i, int) and i > 0 for i in (maxiter, maxiter_picard))
        assert all(isinstance(i, float) and i > 0.0 for i in (tol_picard, tol_picard))

        # set mesh variables
        self._mesh = mesh
        self._boundary_markers = boundary_markers
        self._space_dim = self._mesh.geometry().dim()
        assert self._boundary_markers.dim() == self._space_dim - 1
        self._n_cells = self._mesh.num_cells()

        # dimension-dependent variables
        self._null_vector = dlfn.Constant((0., ) * self._space_dim)

        # set numerical tolerances
        self._tol_picard = tol_picard
        self._maxiter_picard = maxiter_picard
        self._tol = tol
        self._maxiter = maxiter

        # set discretization parameters
        # polynomial degree
        self._p_deg = 1

        # quadrature degree
        q_deg = self._p_deg + 2
        dlfn.parameters["form_compiler"]["quadrature_degree"] = q_deg

    def _setup_function_spaces(self):
        """
        Class method setting up function spaces.
        """
        assert hasattr(self, "_mesh")
        cell = self._mesh.ufl_cell()

        # element formulation
        elemV = dlfn.VectorElement("CG", cell, self._p_deg + 1)
        elemP = dlfn.FiniteElement("CG", cell, self._p_deg)

        # element
        mixedElement = dlfn.MixedElement([elemV, elemP])

        # mixed function space
        self._Wh = dlfn.FunctionSpace(self._mesh, mixedElement)
        self._n_dofs = self._Wh.dim()

        assert hasattr(self, "_n_cells")
        dlfn.info("Number of cells {0}, number of DoFs: {1}".format(self._n_cells, self._n_dofs))

    def _setup_boundary_conditions(self):
        assert hasattr(self, "_bcs")
        assert hasattr(self, "_Wh")
        assert hasattr(self, "_boundary_markers")

        self._dirichlet_bcs = []
        # velocity part
        velocity_space = self._Wh.sub(self._field_association["velocity"])
        velocity_bcs = self._bcs["velocity"]
        for bc_type, bc_bndry_id, bc_value in velocity_bcs:

            if bc_type is VelocityBCType.no_slip:
                bc_object = dlfn.DirichletBC(velocity_space, self._null_vector,
                                             self._boundary_markers, bc_bndry_id)
                self._dirichlet_bcs.append(bc_object)

            elif bc_type is VelocityBCType.constant:
                const_function = dlfn.Constant(bc_value)
                bc_object = dlfn.DirichletBC(velocity_space, const_function,
                                             self._boundary_markers, bc_bndry_id)
                self._dirichlet_bcs.append(bc_object)

            elif bc_type is VelocityBCType.function:
                bc_object = dlfn.DirichletBC(velocity_space, bc_value,
                                             self._boundary_markers, bc_bndry_id)
                self._dirichlet_bcs.append(bc_object)

            # TODO: requires testing
            elif bc_type is VelocityBCType.no_normal_flux:
                # extract normal vector
                bndry_normal = boundary_normal(self._mesh, self._boundary_markers, bc_bndry_id)
                # find direction of normal vector
                bndry_normal = np.array(bndry_normal)
                projections = np.abs(np.identity(self._space_dim).dot(bndry_normal))
                normal_direction = projections.argmax()
                # check that direction is either e_x, e_y or e_z
                assert np.abs(projections[normal_direction] - 1.0) < 1.0e3 * dlfn.DOLFIN_EPS
                assert all([np.abs(projections[d]) for d in range(self._space_dim) if d != normal_direction])
                # construct boundary condition on subspace
                bc_object = dlfn.DirichletBC(velocity_space.sub(normal_direction), self._null_scalar,
                                             self._boundary_markers, bc_bndry_id)
                self._dirichlet_bcs.append(bc_object)

            # TODO: requires testing
            elif bc_type is VelocityBCType.no_tangential_flux:
                # extract normal vector
                bndry_normal = boundary_normal(self._mesh, self._boundary_markers, bc_bndry_id)
                # find direction of normal vector
                bndry_normal = np.array(bndry_normal)
                projections = np.abs(np.identity(self._space_dim).dot(bndry_normal))
                normal_direction = projections.argmax()
                # check that direction is either e_x, e_y or e_z
                assert np.abs(projections[normal_direction] - 1.0) < 1.0e3 * dlfn.DOLFIN_EPS
                assert all([np.abs(projections[d]) for d in range(self._space_dim) if d != normal_direction])
                # construct boundary conditions on subspace
                for d in range(self._space_dim):
                    if d != normal_direction:
                        bc_object = dlfn.DirichletBC(velocity_space.sub(d), self._null_scalar,
                                                     self._boundary_markers, bc_bndry_id)
                        self._dirichlet_bcs.append(bc_object)

            else:
                raise NotImplementedError()
        # pressure part
        if "pressure" in self._bcs:
            pressure_space = self._Wh.sub(self._field_association["pressure"])
            pressure_bcs = self._bcs["pressure"]
            for bc_type, bc_bndry_id, bc_value in pressure_bcs:
                if bc_type is PressureBCType.constant:
                    const_function = dlfn.Constant(bc_value)
                    bc_object = dlfn.DirichletBC(pressure_space, const_function,
                                                 self._boundary_markers, bc_bndry_id)
                    self._dirichlet_bcs.append(bc_object)
                elif bc_type is PressureBCType.function:
                    bc_object = dlfn.DirichletBC(pressure_space, bc_value,
                                                 self._boundary_markers, bc_bndry_id)
                    self._dirichlet_bcs.append(bc_object)
                elif PressureBCType.none:
                    continue
                else:
                    raise NotImplementedError()

        # traction boundary conditions
        if "traction" in self._bcs:
            self._traction_bcs = dict()
            traction_bcs = self._bcs["traction"]
            for bc_type, bc_bndry_id, bc_value in traction_bcs:
                if bc_type is not TractionBCType.free:
                    # make sure that there is no velocity boundary condition on
                    # the current boundary
                    for _, velocity_bndry_id, _ in velocity_bcs:
                        assert velocity_bndry_id != bc_bndry_id, \
                            ValueError("Unconsistent boundary conditions on boundry with "
                                       "boundary id: {0}.".format(bc_bndry_id))
                    if bc_type is TractionBCType.constant:
                        const_function = dlfn.Constant(bc_value)
                        self._traction_bs[bc_bndry_id] = const_function
                    elif bc_type is TractionBCType.function:
                        self._traction_bs[bc_bndry_id] = bc_value
                    else:
                        raise NotImplementedError()

    def _setup_problem(self):
        """
        Method setting up non-linear solver objects of the stationary problem.
        """
        assert hasattr(self, "_mesh")
        assert hasattr(self, "_boundary_markers")

        self._setup_function_spaces()
        self._setup_boundary_conditions()

        # creating test and trial functions
        (v, p) = dlfn.TrialFunctions(self._Wh)
        (w, q) = dlfn.TestFunctions(self._Wh)

        # solution
        self._solution = dlfn.Function(self._Wh)
        sol_v, sol_p = dlfn.split(self._solution)

        # volume element
        dV = dlfn.Measure("dx", domain=self._mesh)
        dA = dlfn.Measure("ds", domain=self._mesh, subdomain_data=self._boundary_markers)

        # dimensionless parameters
        assert hasattr(self, "_Re")
        Re = self._Re

        # viscous operator
        if hasattr(self, "_traction_bs"):
            def a(phi, psi): return inner(grad(phi), grad(psi))
        else:
            def a(phi, psi):
                return dlfn.Constant(0.5) * inner(grad(phi) + grad(phi).T,
                                                  grad(psi) + grad(psi).T)
        # divergence operator
        def b(phi, psi): return inner(div(phi), psi)
        # non-linear convection operator
        def c(phi, chi, psi): return dot(dot(grad(chi), phi), psi)

        # weak forms
        # mass balance
        F_mass = -b(sol_v, q) * dV
        # momentum balance
        F_momentum = (c(sol_v, sol_v, w) - b(w, sol_p) + (1. / Re) * a(sol_v, w)) * dV
        # add body force term
        if self._apply_body_force is True:
            assert hasattr(self, "_Fr"), "Froude number is not specified."
            assert hasattr(self, "_body_force"), "Body force is not specified."
            F_momentum -= dot(self._body_force, w) * dV
        # add boundary tractions
        if hasattr(self, "traction_bcs"):
            for bndry_id, traction in self._traction_bcs.items():
                F_momentum += dot(traction, w) * dA(bndry_id)

        self._F = F_mass + F_momentum

        # linearization using Picard's method
        J_picard_mass = -b(v, q) * dV
        J_picard_momentum = (c(sol_v, v, w) - b(w, p) + (1. / Re) * a(v, w)) * dV
        self._J_picard = J_picard_mass + J_picard_momentum

        # linearization using Newton's method
        self._J_newton = dlfn.derivative(self._F, self._solution)

        # setup non-linear solver
        linear_solver = dlfn.PETScLUSolver()
        comm = dlfn.MPI.comm_world
        factory = dlfn.PETScFactory.instance()
        self._nonlinear_solver = dlfn.NewtonSolver(comm, linear_solver, factory)

        # setup problem with Picard linearization
        self._picard_problem = CustomNonlinearProblem(self._F,
                                                      self._dirichlet_bcs,
                                                      self._J_picard)

        # setup problem with Newton linearization
        self._newton_problem = CustomNonlinearProblem(self._F,
                                                      self._dirichlet_bcs,
                                                      self._J_newton)

    def set_boundary_conditions(self, bcs):
        """
        Set the boundary conditions of the problem.
        """
        assert isinstance(bcs, dict)

        # create a set containing contrained boundaries
        bndry_ids = set()

        # check if structure of dictionary is correct
        for key, bc_group in bcs.items():
            if key == "velocity":
                group_type = VelocityBCType
                none_type = VelocityBCType.no_normal_flux
            elif key == "pressure":
                group_type = PressureBCType
                none_type = PressureBCType.none
            elif key == "traction":
                group_type = TractionBCType
                none_type = TractionBCType.free
            else:
                raise ValueError("The field key <{0}> is unknown ".format(key))

            const_type = group_type.constant
            assert isinstance(bc_group, (tuple, list))

            # check group of boundary conditions
            for bc in bc_group:
                assert isinstance(bc, tuple)
                assert len(bc) == 3

                bc_type, bc_bndry_id, bc_value = bc

                # check if type of boundary condition is known
                assert bc_type in group_type

                # check if type of boundary id is correct
                assert isinstance(bc_bndry_id, int) and bc_bndry_id > 0
                bndry_ids.add(bc_bndry_id)

                # check if value type of bc is correct
                # a) check None for none_type
                if bc_type is none_type:
                    assert bc_value is None
                # b) check None for trivial ones
                elif bc_type in (VelocityBCType.no_slip, VelocityBCType.no_tangential_flux):
                    assert bc_value is None
                # c) check dimensions for constants
                elif bc_type is const_type:
                    # vector fields
                    if group_type in (VelocityBCType, TractionBCType):
                        assert isinstance(bc_value, (list, tuple))
                        assert len(bc_value) == self._space_dim
                        assert all(isinstance(x, float) for x in bc_value)
                    # scalar fields
                    else:
                        assert isinstance(bc_value, float)
                # d) check dimensions for functions
                else:
                    isinstance(bc_value, dlfn.Expression)
                    if group_type in (VelocityBCType, TractionBCType):
                        # check rank
                        assert len(bc_value.ufl_shape) == 1
                        # check dimension
                        assert bc_value.ufl_shape[0] == self._space_dim

        # check that all passed boundary ids occur in the facet markers
        bndry_ids_found = dict(zip(bndry_ids, (False, ) * len(bndry_ids)))
        for facet in dlfn.facets(self._mesh):
            if facet.exterior():
                if self._boundary_markers[facet] in bndry_ids:
                    bndry_ids_found[self._boundary_markers[facet]] = True
                    if all(bndry_ids_found.values()):
                        break
        if not all(bndry_ids_found):
            missing = [key for key, value in bndry_ids_found.items() if value is False]
            message = "Boundary id" + ("s " if len(missing) > 1 else " ")
            message += ", ".join(map(str, missing))
            message += "were not found in the facet markers of the mesh."
            raise ValueError(message)

        # boundary conditions accepted
        self._bcs = bcs

    def set_dimensionless_numbers(self, Re=1.0, Fr=None):
        """
        Updates the parameters of the model by creating or modifying class
        objects.

        Parameters
        ----------
        Re : float
            Kinetic Reynolds numbers.
        Fr : float
            Froude number.
        """
        assert isinstance(Re, float) and Re > 0.0
        if not hasattr(self, "_Re"):
            self._Re = dlfn.Constant(Re)
        else:
            self._Re.assign(Re)

        if Fr is not None:
            assert isinstance(Fr, float) and Fr > 0.0
            self._apply_body_force = True

            if self._body_force_specified is False:
                dlfn.info("Attention: The body force is not specified "
                          "although the Froude number is.")

            if not hasattr(self, "_Fr"):
                self._Fr = dlfn.Constant(Fr)
            else:
                self._Fr.assign(Fr)

    @property
    def sub_space_association(self):
        return self._sub_space_association

    @property
    def field_association(self):
        return self._field_association

    @property
    def solution(self):
        return self._solution

    def set_body_force(self, body_force):
        """
        Specifies the body force.

        Parameters
        ----------
        body_force : dolfin.Expression, dolfin. Constant
            The body force.
        """
        assert isinstance(body_force, (dlfn.Expression, dlfn.Constant))
        assert body_force.ufl_shape[0] == self._space_dim
        self._body_force = body_force
        self._body_force_specified = True

        if self._body_force_specified is False:
            dlfn.info("Attention: The Froude number is not specified "
                      "although the body force is.")

    def solve(self):
        """
        Solves the nonlinear problem.
        """
        # setup problem
        if not all(hasattr(self, attr) for attr in ("_nonlinear_solver",
                                                    "_picard_problem",
                                                    "_newton_problem",
                                                    "_solution")):
            self._setup_problem()

        # compute initial residual
        residual_vector = dlfn.Vector(self._solution.vector())
        self._picard_problem.F(residual_vector, self._solution.vector())
        residual = residual_vector.norm("l2")

        # correct initial tolerance if necessary
        if residual < self._tol_picard:
            # determine order of magnitude
            order = math.floor(math.log10(residual))
            # specify corrected tolerance
            self._tol_picard = (residual / 10.0**order - 1.0) * 10.0**order

        # Picard iteration
        dlfn.info("Starting Picard iteration...")
        self._nonlinear_solver.parameters["maximum_iterations"] = self._maxiter_picard
        self._nonlinear_solver.parameters["absolute_tolerance"] = self._tol_picard
        self._nonlinear_solver.solve(self._picard_problem, self._solution.vector())

        # Newton's method
        dlfn.info("Starting Newton iteration...")
        self._nonlinear_solver.parameters["absolute_tolerance"] = self._tol
        self._nonlinear_solver.parameters["maximum_iterations"] = self._maxiter
        self._nonlinear_solver.parameters["error_on_nonconvergence"] = False
        self._nonlinear_solver.solve(self._newton_problem, self._solution.vector())

        # check residual
        self._newton_problem.F(residual_vector, self._solution.vector())
        residual = residual_vector.norm("l2")
        assert residual <= self._tol, "Newton iteration did not converge."
