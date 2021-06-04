#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from enum import Enum, auto

import numpy as np

import math

import dolfin as dlfn
from dolfin import grad, div, dot, inner

from auxiliary_classes import CustomNonlinearProblem
from auxiliary_methods import boundary_normal
from auxiliary_methods import extract_all_boundary_markers
from imex_time_stepping import IMEXTimeStepping


class VelocityBCType(Enum):
    no_slip = auto()
    no_normal_flux = auto()
    no_tangential_flux = auto()
    constant = auto()
    constant_component = auto()
    function = auto()
    function_component = auto()


class PressureBCType(Enum):
    constant = auto()
    function = auto()
    mean_value = auto()
    none = auto()


class TractionBCType(Enum):
    constant = auto()
    constant_component = auto()
    function = auto()
    function_component = auto()
    free = auto()


class SpatialDiscretizationConvectiveTerm(Enum):
    standard = auto()


class NavierStokesSolverBase:
    """
    Class to simulate stationary fluid flow of an incompressible fluid using
    P2-P1 finite elements. The system is solved hybrid Picard-Newton iteration.

    Parameters
    ----------
    """
    # class variables
    _null_scalar = dlfn.Constant(0.)

    def __init__(self, mesh, boundary_markers):

        # input check
        assert isinstance(mesh, dlfn.Mesh)
        assert isinstance(boundary_markers, (dlfn.cpp.mesh.MeshFunctionSizet,
                                             dlfn.cpp.mesh.MeshFunctionInt))
        # set mesh variables
        self._mesh = mesh
        self._boundary_markers = boundary_markers
        self._space_dim = self._mesh.geometry().dim()
        assert self._boundary_markers.dim() == self._space_dim - 1
        self._n_cells = self._mesh.num_cells()

        # dimension-dependent variables
        self._null_vector = dlfn.Constant((0., ) * self._space_dim)

        # set discretization parameters

        # polynomial degree
        self._p_deg = 1

        # quadrature degree
        q_deg = self._p_deg + 2
        dlfn.parameters["form_compiler"]["quadrature_degree"] = q_deg

    def _check_boundary_condition_format(self, bc):
        """
        Check the general format of an arbitrary boundary condition.
        """
        assert hasattr(self, "_mesh")
        assert hasattr(self, "_boundary_markers")
        # boundary ids specified in the MeshFunction
        all_bndry_ids = extract_all_boundary_markers(self._mesh, self._boundary_markers)
        # 0. input check
        assert isinstance(bc, (list, tuple))
        assert len(bc) >= 2
        # 1. check bc type
        assert isinstance(bc[0], (VelocityBCType, PressureBCType, TractionBCType))
        if isinstance(bc[0], PressureBCType):
            rank = 0
        else:
            rank = 1
        # 2. check boundary id
        assert isinstance(bc[1], int)
        assert bc[1] in all_bndry_ids, "Boundary id {0} ".format(bc[1]) +\
                                       "was not found in the boundary markers."
        # 3. check value type
        # distinguish between scalar and vector field
        if rank == 0:
            # scalar field (tensor of rank zero)
            assert isinstance(bc[2], (dlfn.Expression, float)) or bc[2] is None
            if isinstance(bc[2], dlfn.Expression):
                # check rank of expression
                assert bc[2].value_rank() == 0

        elif rank == 1:
            # vector field (tensor of rank one)
            # distinguish between full or component-wise boundary conditions
            if len(bc) == 3:
                # full boundary condition
                assert isinstance(bc[2], (dlfn.Expression, tuple, list)) or bc[2] is None
                if isinstance(bc[2], dlfn.Expression):
                    # check rank of expression
                    assert bc[2].value_rank() == 1
                elif isinstance(bc[2], (tuple, list)):
                    # size of tuple or list
                    assert len(bc[2]) == self._space_dim
                    # type of the entries
                    assert all(isinstance(x, float) for x in bc[2])

            elif len(bc) == 4:
                # component-wise boundary condition
                # component index specified
                assert isinstance(bc[2], int)
                assert bc[2] < self._space_dim
                # value specified
                assert isinstance(bc[3], (dlfn.Expression, float)) or bc[3] is None
                if isinstance(bc[3], dlfn.Expression):
                    # check rank of expression
                    assert bc[3].value_rank() == 0

    def _setup_function_spaces(self):
        """Class method setting up function spaces."""
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
        assert hasattr(self, "_Wh")
        assert hasattr(self, "_boundary_markers")
        assert hasattr(self, "_velocity_bcs")
        # empty dirichlet bcs
        self._dirichlet_bcs = []

        # velocity part
        velocity_space = self._Wh.sub(self._field_association["velocity"])
        for bc in self._velocity_bcs:
            # unpack values
            if len(bc) == 3:
                bc_type, bndry_id, value = bc
            elif len(bc) == 4:
                bc_type, bndry_id, component_index, value = bc
            else:  # pragma: no cover
                raise RuntimeError()
            # create dolfin.DirichletBC object
            if bc_type is VelocityBCType.no_slip:
                bc_object = dlfn.DirichletBC(velocity_space, self._null_vector,
                                             self._boundary_markers, bndry_id)
                self._dirichlet_bcs.append(bc_object)

            elif bc_type is VelocityBCType.no_normal_flux:
                # compute normal vector of boundary
                bndry_normal = boundary_normal(self._mesh, self._boundary_markers, bndry_id)
                # find associated component
                bndry_normal = np.array(bndry_normal)
                normal_component_index = np.abs(bndry_normal).argmax()
                # check that direction is either e_x, e_y or e_z
                assert abs(bndry_normal[component_index] - 1.0) < 5.0e-15
                assert all([abs(bndry_normal[d]) < 5.0e-15 for d in range(self._space_dim) if d != normal_component_index])
                # construct boundary condition on subspace
                bc_object = dlfn.DirichletBC(velocity_space.sub(normal_component_index),
                                             self._null_scalar, self._boundary_markers,
                                             bndry_id)
                self._dirichlet_bcs.append(bc_object)

            elif bc_type is VelocityBCType.no_tangential_flux:
                # compute normal vector of boundary
                bndry_normal = boundary_normal(self._mesh, self._boundary_markers, bndry_id)
                # find associated component
                bndry_normal = np.array(bndry_normal)
                normal_component_index = np.abs(bndry_normal).argmax()
                # check that direction is either e_x, e_y or e_z
                assert abs(bndry_normal[normal_component_index] - 1.0) < 5.0e-15
                assert all([abs(bndry_normal[d]) < 5.0e-15 for d in range(self._space_dim) if d != normal_component_index])
                # compute tangential components
                tangential_component_indices = (d for d in range(self._space_dim) if d != normal_component_index)
                # construct boundary condition on subspace
                for component_index in tangential_component_indices:
                    bc_object = dlfn.DirichletBC(velocity_space.sub(component_index),
                                                 self._null_scalar, self._boundary_markers,
                                                 bndry_id)
                    self._dirichlet_bcs.append(bc_object)

            elif bc_type is VelocityBCType.constant:
                assert isinstance(value, (tuple, list))
                const_function = dlfn.Constant(value)
                bc_object = dlfn.DirichletBC(velocity_space, const_function,
                                             self._boundary_markers, bndry_id)
                self._dirichlet_bcs.append(bc_object)

            elif bc_type is VelocityBCType.constant_component:
                assert isinstance(value, float)
                const_function = dlfn.Constant(value)
                bc_object = dlfn.DirichletBC(velocity_space.sub(component_index),
                                             const_function,
                                             self._boundary_markers, bndry_id)
                self._dirichlet_bcs.append(bc_object)

            elif bc_type is VelocityBCType.function:
                assert isinstance(value, dlfn.Expression)
                bc_object = dlfn.DirichletBC(velocity_space, value,
                                             self._boundary_markers, bndry_id)
                self._dirichlet_bcs.append(bc_object)

            elif bc_type is VelocityBCType.function_component:
                assert isinstance(value, dlfn.Expression)
                bc_object = dlfn.DirichletBC(velocity_space.sub(component_index),
                                             value,
                                             self._boundary_markers, bndry_id)
                self._dirichlet_bcs.append(bc_object)

            else:  # pragma: no cover
                raise RuntimeError()

        # velocity part
        pressure_space = self._Wh.sub(self._field_association["pressure"])
        if hasattr(self, "_pressure_bcs"):
            for bc in self._pressure_bcs:
                # unpack values
                if len(bc) == 3:
                    bc_type, bndry_id, value = bc
                else:  # pragma: no cover
                    raise RuntimeError()
                # create dolfin.DirichletBC object
                if bc_type is VelocityBCType.constant:
                    assert isinstance(value, (tuple, list))
                    const_function = dlfn.Constant(value)
                    bc_object = dlfn.DirichletBC(pressure_space, const_function,
                                                 self._boundary_markers, bndry_id)
                    self._dirichlet_bcs.append(bc_object)

                elif bc_type is VelocityBCType.function:
                    assert isinstance(value, dlfn.Expression)
                    bc_object = dlfn.DirichletBC(pressure_space, value,
                                                 self._boundary_markers, bndry_id)
                    self._dirichlet_bcs.append(bc_object)

                elif bc_type is VelocityBCType.none:
                    continue

                else:  # pragma: no cover
                    raise RuntimeError()
        # HINT: traction boundary conditions are covered in _setup_problem

    @property
    def field_association(self):
        assert hasattr(self, "_field_association")
        return self._field_association

    def set_body_force(self, body_force):
        """
        Specifies the body force.

        Parameters
        ----------
        body_force : dolfin.Expression, dolfin. Constant
            The body force.
        """
        assert isinstance(body_force, (dlfn.Expression, dlfn.Constant))
        if isinstance(body_force, dlfn.Expression):
            assert body_force.value_rank() == 1
        else:
            assert len(body_force.ufl_shape) == 1
            assert body_force.ufl_shape[0] == self._space_dim
        self._body_force = body_force

    def set_boundary_conditions(self, bcs):
        """
        Set the boundary conditions of the problem.
        The boundary conditions are specified as a list of tuples where each
        tuple represents a separate boundary condition. This means that, for
        example,
            bcs = [(Type, boundary_id, value),
                   (Type, boundary_id, component, value)]
        The first entry of each tuple specifies the type of the boundary
        condition. The second entry specifies the boundary identifier where the
        boundary should be applied. If full vector field is constrained through
        the boundary condition, the third entry specifies the value. If only a
        single component is constrained, the third entry specifies the
        component index and the third entry specifies the value.
        """
        assert isinstance(bcs, (list, tuple))
        # check format
        for bc in bcs:
            self._check_boundary_condition_format(bc)

        # extract velocity/traction bcs and related boundary ids
        velocity_bcs = []
        velocity_bc_ids = set()
        traction_bcs = []
        traction_bc_ids = set()
        pressure_bcs = []
        pressure_bc_ids = set()
        for bc in bcs:
            if isinstance(bc[0], VelocityBCType):
                velocity_bcs.append(bc)
                velocity_bc_ids.add(bc[1])
            elif isinstance(bc[0], TractionBCType):
                traction_bcs.append(bc)
                traction_bc_ids.add(bc[1])
            elif isinstance(bc[0], PressureBCType):
                pressure_bcs.append(bc)
                pressure_bc_ids.add(bc[1])
        # check that at least one velocity bc is specified
        assert len(velocity_bcs) > 0

        # check that there is no conflict between velocity and traction bcs
        if len(traction_bcs) > 0:
            # compute boundary ids with simultaneous bcs
            joint_bndry_ids = velocity_bc_ids.intersection(traction_bc_ids)
            # make sure that bcs are only applied component-wise
            allowedVelocityBCTypes = (VelocityBCType.no_normal_flux,
                                      VelocityBCType.no_tangential_flux,
                                      VelocityBCType.constant_component,
                                      VelocityBCType.function_component)
            allowedTractionBCTypes = (TractionBCType.constant_component,
                                      TractionBCType.function_component)
            for bndry_id in joint_bndry_ids:
                # extract component of velocity bc
                vel_bc_component = None
                for bc in velocity_bcs:
                    if bc[1] == bndry_id:
                        assert bc[0] in allowedVelocityBCTypes
                        vel_bc_component = bc[2]
                        break
                # extract component of traction bc
                traction_bc_component = None
                for bc in traction_bcs:
                    if bc[1] == bndry_id:
                        assert bc[0] in allowedTractionBCTypes
                        traction_bc_component = bc[2]
                        break
                # compare components
                assert traction_bc_component != vel_bc_component
        # boundary conditions accepted
        self._velocity_bcs = velocity_bcs
        if len(traction_bcs) > 0:
            self._traction_bcs = traction_bcs
        if len(pressure_bcs) > 0:
            self._pressure_bcs = pressure_bcs

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
            if not hasattr(self, "_Fr"):
                self._Fr = dlfn.Constant(Fr)
            else:
                self._Fr.assign(Fr)

    @property
    def sub_space_association(self):
        assert hasattr(self, "_sub_space_association")
        return self._sub_space_association

    @property
    def solution(self):
        return self._solution

    def solve(self):
        """
        Purely virtual method for specifying setting up the problem.
        """
        raise NotImplementedError("You are calling a purely virtual method.")


class StationaryNavierStokesSolver(NavierStokesSolverBase):
    """
    Class to simulate stationary fluid flow of an incompressible fluid using
    P2-P1 finite elements. The system is solved hybrid Picard-Newton iteration.

    Parameters
    ----------
    """
    # class variables
    _sub_space_association = {0: "velocity", 1: "pressure"}
    _field_association = {value: key for key, value in _sub_space_association.items()}
    _apply_boundary_traction = False

    def __init__(self, mesh, boundary_markers, tol=1e-10, maxiter=50,
                 tol_picard=1e-2, maxiter_picard=10):

        super().__init__(mesh, boundary_markers)

        # input check
        assert all(isinstance(i, int) and i > 0 for i in (maxiter, maxiter_picard))
        assert all(isinstance(i, float) and i > 0.0 for i in (tol, tol_picard))

        # set numerical tolerances
        self._tol_picard = tol_picard
        self._maxiter_picard = maxiter_picard
        self._tol = tol
        self._maxiter = maxiter

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
        if hasattr(self, "_body_force"):
            assert hasattr(self, "_Fr"), "Froude number is not specified."
            F_momentum -= dot(self._body_force, w) / self._Fr**2 * dV

        # add boundary tractions
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
                    F_momentum += dot(const_function, w) * dA(bndry_id)

                elif bc_type is TractionBCType.constant_component:
                    assert isinstance(traction, float)
                    const_function = dlfn.Constant(traction)
                    F_momentum += const_function * w[component_index] * dA(bndry_id)

                elif bc_type is TractionBCType.function:
                    assert isinstance(traction, dlfn.Expression)
                    F_momentum += dot(traction, w) * dA(bndry_id)

                elif bc_type is TractionBCType.function_component:
                    assert isinstance(traction, dlfn.Expression)
                    F_momentum += traction * w[component_index] * dA(bndry_id)

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

    def solve(self):
        """Solves the nonlinear problem."""
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


class InstationaryNavierStokesSolver(NavierStokesSolverBase):
    _sub_space_association = {0: "velocity", 1: "pressure"}
    _field_association = {value: key for key, value in _sub_space_association.items()}

    def __init__(self, mesh, boundary_markers, time_stepping, tol, max_iter):

        super().__init__(mesh, boundary_markers)

        # input check
        assert isinstance(max_iter, int)
        assert max_iter > 0
        assert isinstance(tol, float)
        assert tol > 0.0
        assert isinstance(time_stepping, (IMEXTimeStepping, ))

        # time stepping scheme
        self._time_stepping = time_stepping

        # set numerical tolerances
        self._tol = tol
        self._maxiter = max_iter

    def _advance_solution(self):
        """Advance solution objects in time."""
        assert hasattr(self, "_old_old_solution")
        assert hasattr(self, "_old_solution")
        assert hasattr(self, "_solution")
        self._old_old_solution.assign(self._old_solution)
        self._old_solution.assign(self._solution)

    def _setup_function_spaces(self):
        """Class method setting up function spaces."""
        super()._setup_function_spaces()
        # create solution
        self._solution = dlfn.Function(self._Wh)
        # create old solutions
        self._old_solution = dlfn.Function(self._Wh)
        self._old_old_solution = dlfn.Function(self._Wh)

    def _setup_problem(self):
        """Method setting up non-linear solver object of the instationary
        problem.
        """
        assert hasattr(self, "_mesh")
        assert hasattr(self, "_boundary_markers")

        if not all(hasattr(self, attr) for attr in ("_Wh",
                                                    "_solution",
                                                    "_old_solution",
                                                    "_old_old_solution")):  # pragma: no cover
            self._setup_function_spaces()

        if not all(hasattr(self, attr) for attr in ("_next_step_size",
                                                    "_alpha")):
            self._update_time_stepping_coefficients()

        self._setup_boundary_conditions()

        # creating test and trial functions
        (v, p) = dlfn.TrialFunctions(self._Wh)
        (w, q) = dlfn.TestFunctions(self._Wh)

        # split solutions
        velocity, pressure = dlfn.split(self._solution)
        old_velocity, _ = dlfn.split(self._old_solution)
        old_old_velocity, _ = dlfn.split(self._old_old_solution)

        # volume element
        dV = dlfn.Measure("dx", domain=self._mesh)
        dA = dlfn.Measure("ds", domain=self._mesh, subdomain_data=self._boundary_markers)

        # dimensionless parameters
        assert hasattr(self, "_Re")
        Re = self._Re

        # step size
        k = self._next_step_size

        # time stepping coefficients
        alpha = self._alpha

        # viscous operator
        def a(phi, psi):
            return dlfn.Constant(0.5) * inner(grad(phi) + grad(phi).T,
                                              grad(psi) + grad(psi).T)

        # divergence operator
        def b(phi, psi): return inner(div(phi), psi)
        # non-linear convection operator
        def c(phi, chi, psi): return dot(dot(grad(chi), phi), psi)
        # weak forms
        # mass balance
        F_mass = -b(velocity, q) * dV

        # momentum balance
        F_momentum = (
                        (
                            alpha[0] * dot(velocity, w)
                            + alpha[1] * dot(old_velocity, w)
                            + alpha[2] * dot(old_old_velocity, w)
                        ) / k
                        + c(velocity, velocity, w) - b(w, pressure)
                        + a(velocity, w) / Re
                        ) * dV

        # add body force term
        if hasattr(self, "_body_force"):
            assert hasattr(self, "_Fr"), "Froude number is not specified."
            F_momentum -= dot(self._body_force, w) / self._Fr**2 * dV

        # add boundary tractions
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
                    F_momentum += dot(const_function, w) * dA(bndry_id)

                elif bc_type is TractionBCType.constant_component:
                    assert isinstance(traction, float)
                    const_function = dlfn.Constant(traction)
                    F_momentum += const_function * w[component_index] * dA(bndry_id)

                elif bc_type is TractionBCType.function:
                    assert isinstance(traction, dlfn.Expression)
                    F_momentum += dot(traction, w) * dA(bndry_id)

                elif bc_type is TractionBCType.function_component:
                    assert isinstance(traction, dlfn.Expression)
                    F_momentum += traction * w[component_index] * dA(bndry_id)

        self._F = F_mass + F_momentum

        # linearization using Picard's method
        J_picard_mass = -b(v, q) * dV
        J_picard_momentum = (
                alpha[0] * dot(v, w) / k
                + c(velocity, v, w) - b(w, p) + a(v, w) / Re) * dV
        self._J_picard = J_picard_mass + J_picard_momentum

        # linearization using Newton's method
        self._J_newton = dlfn.derivative(self._F, self._solution)

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
        self._newton_problem = CustomNonlinearProblem(self._F,
                                                      self._dirichlet_bcs,
                                                      self._J_newton)

    def _set_time(self):
        """Set time of boundary condition objects and body force."""
        assert hasattr(self, "_velocity_bcs")
        # extract next time
        next_time = self._time_stepping.next_time

        # auxiliary function
        def modify_time(expression):
            # modify time
            if isinstance(expression, dlfn.Expression):
                if hasattr(expression, "time"):
                    expression.time = next_time
                elif hasattr(expression, "t"):
                    expression.t = next_time
        # velocity boundary conditions
        for bc in self._velocity_bcs:
            # unpack values
            if len(bc) == 3:
                value = bc[2]
            elif len(bc) == 4:
                value = bc[3]
            else:  # pragma: no cover
                raise RuntimeError()
            # modify time
            modify_time(value)
        # pressure boundary conditions
        if hasattr(self, "_pressure_bcs"):
            for bc in self._pressure_bcs:
                # unpack values
                if len(bc) == 3:
                    value = bc[2]
                else:  # pragma: no cover
                    raise RuntimeError()
                # modify time
                modify_time(value)
        # traction boundary conditions
        if hasattr(self, "_traction_bcs"):
            for bc in self._traction_bcs:
                # unpack values
                if len(bc) == 3:
                    value = bc[2]
                elif len(bc) == 4:
                    value = bc[3]
                else:  # pragma: no cover
                    raise RuntimeError()
                # modify time
                modify_time(value)
        # body force
        if hasattr(self, "_body_force"):
            # modify time
            modify_time(self._body_force)

    def _update_time_stepping_coefficients(self):
        """Update time stepping coefficients ``_alpha`` and ``_next_step_size``."""
        # update time steps
        next_step_size = self._time_stepping.get_next_step_size()
        if not hasattr(self, "_next_step_size"):
            self._next_step_size = dlfn.Constant(next_step_size)
        else:
            self._next_step_size.assign(next_step_size)
        # update coefficients
        alpha = self._time_stepping.alpha
        assert len(alpha) == 3
        if not hasattr(self, "_alpha"):
            self._alpha = [dlfn.Constant(alpha[0]), dlfn.Constant(alpha[1]),
                           dlfn.Constant(alpha[2])]
        else:
            for i in range(3):
                self._alpha[i].assign(alpha[i])

    def advance_time(self):
        """Advance relevant objects by one time step."""
        self._advance_solution()

    def set_initial_conditions(self, initial_conditions):
        """Setup the initial conditions of the problem."""
        # input check
        assert isinstance(initial_conditions, dict)
        assert "velocity" in initial_conditions
        # check that function spaces exist
        if not all(hasattr(self, attr) for attr in ("_Wh",
                                                    "_solution",
                                                    "_old_solution",
                                                    "_old_old_solution")):
            self._setup_function_spaces()
        # split functions
        old_velocity, old_pressure = self._old_solution.split()

        # velocity part
        # extract velocity initial condition
        velocity_condition = initial_conditions["velocity"]
        # check format of initial condition
        assert isinstance(velocity_condition, (dlfn.Expression, tuple, list))
        if isinstance(velocity_condition, dlfn.Expression):
            # check rank of expression
            assert velocity_condition.value_rank() == 1
            velocity_expression = velocity_condition
        elif isinstance(velocity_condition, (tuple, list)):
            # size of tuple or list
            assert len(velocity_condition) == self._space_dim
            # type of the entries
            assert all(isinstance(x, float) for x in velocity_condition)
            velocity_expression = dlfn.Constant(velocity_condition)

        # project and assign
        subspace_index = self._field_association["velocity"]
        velocity_space = dlfn.FunctionSpace(self._Wh.mesh(),
                                            self._Wh.sub(subspace_index).ufl_element())
        projected_velocity_condition = dlfn.project(velocity_expression,
                                                    velocity_space)
        dlfn.assign(old_velocity, projected_velocity_condition)

        # pressure part
        if "pressure" in initial_conditions:
            pressure_condition = initial_conditions["pressure"]
            # check format of initial condition
            assert isinstance(pressure_condition, (dlfn.Expression, float))
            if isinstance(pressure_condition, dlfn.Expression):
                # check rank of expression
                assert pressure_condition.value_rank() == 0
                pressure_expression = pressure_condition
            else:
                pressure_expression = dlfn.Constant(pressure_condition)
            # project and assign
            subspace_index = self._field_association["pressure"]
            pressure_space = dlfn.FunctionSpace(self._Wh.mesh(),
                                                self._Wh.sub(subspace_index).ufl_element())
            projected_pressure_condition = dlfn.project(pressure_expression,
                                                        pressure_space)

            dlfn.assign(old_pressure, projected_pressure_condition)
        # TODO: Implement Poisson equation for the initial pressure

    def solve(self):
        """Solves the nonlinear problem."""
        # setup problem
        if not all(hasattr(self, attr) for attr in ("_nonlinear_solver",
                                                    "_newton_problem",
                                                    "_solution")):
            self._setup_problem()

        # update time
        self._set_time()
        # update coefficients if necessary
        if self._time_stepping.coefficients_changed:
            self._update_time_stepping_coefficients()

        # compute initial residual
        residual_vector = dlfn.Vector(self._solution.vector())
        self._picard_problem.F(residual_vector, self._solution.vector())
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
        self._nonlinear_solver.solve(self._picard_problem, self._solution.vector())

        # solve problem
        dlfn.info("Starting Newton iteration...")
        self._nonlinear_solver.parameters["absolute_tolerance"] = self._tol
        self._nonlinear_solver.parameters["maximum_iterations"] = self._maxiter
        self._nonlinear_solver.parameters["relative_tolerance"] = 1.0e1 * self._tol
        self._nonlinear_solver.parameters["error_on_nonconvergence"] = True
        self._nonlinear_solver.solve(self._newton_problem, self._solution.vector())
