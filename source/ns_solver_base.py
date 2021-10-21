#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from auxiliary_classes import CustomNonlinearProblem
from auxiliary_methods import boundary_normal
from auxiliary_methods import extract_all_boundary_markers
from auxiliary_classes import AngularVelocityVector
import dolfin as dlfn
from dolfin import cross, curl, div, dot, grad, inner
from discrete_time import DiscreteTime
from enum import Enum, auto
import math
import numpy as np
import ufl


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


class TractionBCType(Enum):
    constant = auto()
    constant_component = auto()
    function = auto()
    function_component = auto()
    free = auto()


class WeakFormConvectiveTerm(Enum):
    """
    The weak form of the convective term used according to John (2016),
    pgs. 307-308.
    """
    standard_form = auto()
    rotational_form = auto()
    divergence_form = auto()
    skew_symmetric_form = auto()


class WeakFormViscousTerm(Enum):
    """
    The weak form of the viscous term.
    """
    reduced_form = auto()
    traction_form = auto()


class SolverBase:
    """
    Class to simulate stationary fluid flow of an incompressible fluid using
    P2-P1 finite elements. The system is solved hybrid Picard-Newton iteration.

    Parameters
    ----------
    """
    # class variables
    _null_scalar = dlfn.Constant(0., name="null")
    _one_half = dlfn.Constant(0.5, name="one_half")
    _form_function_types = (dlfn.function.argument.Argument,
                            dlfn.function.function.Function,
                            ufl.tensors.ListTensor,
                            ufl.indexed.Indexed)
    _form_trial_function_types = (dlfn.function.argument.Argument, ufl.tensors.ListTensor)
    _sub_space_association = {0: "velocity", 1: "pressure"}
    _field_association = {value: key for key, value in _sub_space_association.items()}

    def __init__(self, mesh, boundary_markers, form_convective_term="standard",
                 form_viscous_term="reduced"):
        # input check
        assert isinstance(mesh, dlfn.Mesh)
        assert isinstance(boundary_markers, (dlfn.cpp.mesh.MeshFunctionSizet,
                                             dlfn.cpp.mesh.MeshFunctionInt))
        assert isinstance(form_convective_term, str)
        assert form_convective_term.lower() in ("standard", "rotational",
                                                "divergence", "skew_symmetric")
        assert isinstance(form_viscous_term, str)
        assert form_viscous_term.lower() in ("standard", "reduced", "traction")

        # set mesh variables
        self._mesh = mesh
        self._boundary_markers = boundary_markers
        self._space_dim = self._mesh.geometry().dim()
        assert self._boundary_markers.dim() == self._space_dim - 1
        self._n_cells = self._mesh.num_cells()

        # dimension-dependent variables
        self._null_vector = dlfn.Constant((0., ) * self._space_dim)

        # set discretization parameters
        if form_convective_term.lower() == "standard":
            self._form_convective_term = WeakFormConvectiveTerm.standard_form
        elif form_convective_term.lower() == "rotational":
            self._form_convective_term = WeakFormConvectiveTerm.rotational_form
        elif form_convective_term.lower() == "divergence":
            self._form_convective_term = WeakFormConvectiveTerm.divergence_form
        elif form_convective_term.lower() == "skew_symmetric":
            self._form_convective_term = WeakFormConvectiveTerm.skew_symmetric_form

        if form_viscous_term.lower() == "standard":
            self._form_viscous_term = WeakFormViscousTerm.reduced_form
        elif form_viscous_term.lower() == "reduced":
            self._form_viscous_term = WeakFormViscousTerm.reduced_form
        elif form_viscous_term.lower() == "traction":
            self._form_viscous_term = WeakFormViscousTerm.traction_form

        # set discretization parameters
        # polynomial degree
        self._p_deg = 1

    def _add_boundary_tractions(self, F, w):
        """Method adding boundary traction terms to the weak form"""
        # input check
        assert isinstance(F, ufl.form.Form)
        assert isinstance(w, self._form_trial_function_types)

        dA = dlfn.Measure("ds", domain=self._mesh, subdomain_data=self._boundary_markers)

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
                    F += dot(const_function, w) * dA(bndry_id)

                elif bc_type is TractionBCType.constant_component:
                    assert isinstance(traction, float)
                    const_function = dlfn.Constant(traction)
                    F += const_function * w[component_index] * dA(bndry_id)

                elif bc_type is TractionBCType.function:
                    assert isinstance(traction, dlfn.Expression)
                    F += dot(traction, w) * dA(bndry_id)

                elif bc_type is TractionBCType.function_component:
                    assert isinstance(traction, dlfn.Expression)
                    F += traction * w[component_index] * dA(bndry_id)
        return F

    def _add_body_forces(self, F, w):
        if not hasattr(self, "_body_force"):
            return F
        # input check
        assert hasattr(self, "_equation_coefficients")
        if self._equation_coefficients["body_force_term"] is None:  # pragma: no cover
            raise RuntimeError()
        assert isinstance(F, ufl.form.Form)
        assert isinstance(w, self._form_trial_function_types)

        dV = dlfn.Measure("dx", domain=self._mesh)
        F -= self._equation_coefficients["body_force_term"] * dlfn.dot(self._body_force, w) * dV

        return F

    def _add_coriolis_acceleration(self, F, velocity, w):
        if not hasattr(self, "_angular_velocity"):
            return F
        # input check
        assert hasattr(self, "_equation_coefficients")
        if self._equation_coefficients["coriolis_term"] is None:  # pragma: no cover
            raise RuntimeError()
        assert isinstance(velocity, self._form_function_types)
        assert isinstance(w, self._form_function_types)
        omega = self._angular_velocity.value
        dV = dlfn.Measure("dx", domain=self._mesh)
        if self._space_dim == 2:
            F += dlfn.Constant(2.0) * self._equation_coefficients["coriolis_term"] * omega * \
                    dot(dlfn.as_vector((-velocity[1], velocity[0])), w) * dV
        else:  # pragma: no cover
            assert len(self._Omega) == 3
            F += dlfn.Constant(2.0) * self._equation_coefficients["coriolis_term"] * \
                dot(dlfn.cross(omega, velocity), w) * dV
        return F

    def _add_euler_acceleration(self, F, w):
        if not hasattr(self, "_angular_velocity"):
            return F
        # input check
        assert hasattr(self, "_equation_coefficients")
        if self._equation_coefficients["euler_term"] is None:  # pragma: no cover
            raise RuntimeError()
        assert isinstance(w, self._form_function_types)
        alpha = self._angular_velocity.derivative
        if alpha is None:
            return F
        else:  # pragma: no cover
            x = dlfn.SpatialCoordinate(self._mesh)
            dV = dlfn.Measure("dx", domain=self._mesh)
            if self._space_dim == 2:
                F += self._equation_coefficients["euler_term"] * dot(alpha * dlfn.as_vector((-x[1], x[0])), w) * dV
            else:
                F += self._equation_coefficients["euler_term"] * dot(dlfn.cross(alpha, x), w) * dV
            return F

    def _assign_function(self, receiving_functions, assigning_functions):
        """Assign functions from the joint function space to the subspaces or
        vice versa."""
        assert hasattr(self, "_Wh")
        assert isinstance(receiving_functions, (dlfn.Function, dict))
        assert isinstance(assigning_functions, (dlfn.Function, dict))
        # dictionary of subspaces
        WhSub = self._get_subspaces()
        # check whether a forward or backward assignment should be performed
        forward_assignment = False
        backward_assignment = False
        if isinstance(receiving_functions, dict):
            for key, function in receiving_functions.items():
                if function in WhSub[key]:
                    forward_assignment = True
                if forward_assignment is True:
                    assert function in WhSub[key]
        elif isinstance(assigning_functions, dict):
            for key, function in assigning_functions.items():
                if function in WhSub[key]:
                    backward_assignment = True
                if backward_assignment is True:
                    assert function in WhSub[key]
        elif isinstance(receiving_functions, dlfn.Function) and isinstance(assigning_functions, dlfn.Function):
            for key, space in WhSub.items():
                if receiving_functions in space:
                    forward_assignment = True
                    break
                elif assigning_functions in space:
                    backward_assignment = True
                    break
        else:  # pragma: no cover
            raise RuntimeError()
        assert forward_assignment or backward_assignment

        # forward assignment
        if forward_assignment is True:
            assert isinstance(assigning_functions, dlfn.Function)
            if not isinstance(receiving_functions, dict):
                assert isinstance(receiving_functions, dlfn.Function)
                for key, space in WhSub.items():
                    if receiving_functions in space:
                        break
                index = self._field_association[key]
                assert assigning_functions in self._Wh.sub(index)
                forward_subspace_assigners = self._get_forward_subspace_assigners()
                forward_subspace_assigners[key].assign(receiving_functions, assigning_functions)
            else:
                if len(receiving_functions) == 2:
                    receiving_function_list = [None] * 2
                    for key, function in receiving_functions.items():
                        receiving_function_list[self._field_association[key]] = function
                    forward_assigner = self._get_forward_assigner()
                    forward_assigner.assign(receiving_function_list, assigning_functions)
                elif len(receiving_functions) == 1:
                    key = list(receiving_functions.keys())[0]
                    index = self._field_association[key]
                    assert assigning_functions in self._Wh.sub(index)
                    forward_subspace_assigners = self._get_forward_subspace_assigners()
                    forward_subspace_assigners[key].assign(receiving_functions[key], assigning_functions)
                else:  # pragma: no cover
                    raise RuntimeError()
        else:
            assert isinstance(receiving_functions, dlfn.Function)
            if not isinstance(assigning_functions, dict):
                assert isinstance(assigning_functions, dlfn.Function)
                for key, space in WhSub.items():
                    if assigning_functions in space:
                        break
                index = self._field_association[key]
                assert receiving_functions in self._Wh.sub(index)
                backward_subspace_assigners = self._get_backward_subspace_assigners()
                backward_subspace_assigners[key].assign(receiving_functions, assigning_functions)
            else:
                if len(assigning_functions) == 2:
                    assigning_functions_list = [None] * 2
                    for key, function in assigning_functions.items():
                        assigning_functions_list[self._field_association[key]] = function
                    backward_assigner = self._get_backward_assigner()
                    backward_assigner.assign(receiving_functions, assigning_functions_list)
                elif len(assigning_functions) == 1:
                    key = list(assigning_functions.keys())[0]
                    index = self._field_association[key]
                    assert receiving_functions in self._Wh.sub(index)
                    backward_assigners = self._get_backward_subspace_assigners()
                    backward_assigners[key].assign(receiving_functions, assigning_functions[key])
                else:  # pragma: no cover
                    raise RuntimeError()

    def _check_boundary_condition_format(self, bc, internal_constraint=False):
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
        assert isinstance(internal_constraint, bool)
        # 1. check bc type
        assert isinstance(bc[0], (VelocityBCType, PressureBCType, TractionBCType))
        if isinstance(bc[0], PressureBCType):
            rank = 0
        else:
            rank = 1
        # 2. check boundary id
        if bc[0] is PressureBCType.mean_value:
            pass
        else:
            assert isinstance(bc[1], int)
            if internal_constraint:
                facet_id_found = False
                for f in dlfn.facets(self._mesh):
                    if self._boundary_markers[f] == bc[1]:
                        facet_id_found = True
                        break
                assert facet_id_found
            else:
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

    def _convective_term(self, u, v):
        assert hasattr(self, "_equation_coefficients")
        if self._equation_coefficients["convective_term"] is None:  # pragma: no cover
            raise RuntimeError()
        assert isinstance(u, self._form_function_types)
        assert isinstance(v, self._form_function_types)

        if self._form_convective_term is WeakFormConvectiveTerm.standard_form:
            return self._equation_coefficients["convective_term"] * dot(dot(grad(u), u), v)
        elif self._form_convective_term is WeakFormConvectiveTerm.rotational_form:
            if self._space_dim == 2:
                curl_u = curl(u)
                return self._equation_coefficients["convective_term"] * dot(dlfn.as_vector([-curl_u * u[1], curl_u * u[0]]), v)
            elif self._space_dim == 3:  # pragma: no cover
                return self._equation_coefficients["convective_term"] * dot(cross(curl(u), u), v)
        elif self._form_convective_term is WeakFormConvectiveTerm.divergence_form:
            return self._equation_coefficients["convective_term"] * \
                    (dot(dot(grad(u), u), v) + self._one_half * dot(div(u) * u, v))
        elif self._form_convective_term is WeakFormConvectiveTerm.skew_symmetric_form:
            return self._equation_coefficients["convective_term"] * \
                    self._one_half * (dot(dot(grad(u), u), v) - dot(dot(grad(v), u), u))

    def _divergence_term(self, u, v):
        assert hasattr(self, "_equation_coefficients")
        if self._equation_coefficients["pressure_term"] is None:  # pragma: no cover
            raise RuntimeError()
        assert isinstance(u, self._form_function_types), "{0}".format(type(u))
        assert isinstance(v, self._form_function_types), "{0}".format(type(v))

        return self._equation_coefficients["pressure_term"] * div(u) * v

    def _get_subspace(self, field):
        """Returns the subspace of the `field`."""
        assert isinstance(field, str)
        assert field in self._field_association
        if not hasattr(self, "_WhSub"):
            self._WhSub = dict()

        if field not in self._WhSub:
            subspace_index = self._field_association[field]
            self._WhSub[field] = self._Wh.sub(subspace_index).collapse()

        return self._WhSub[field]

    def _get_subspaces(self):
        """Returns a dictionary of the subspaces of all physical fields."""
        assert hasattr(self, "_Wh")
        if not hasattr(self, "_WhSub"):
            self._WhSub = dict()
        for key, index in self._field_association.items():
            if key not in self._WhSub:
                self._WhSub[key] = self._Wh.sub(index).collapse()
        return self._WhSub

    def _get_backward_assigner(self):
        """Returns a function assigner which assigns from all subspaces to the
        joint function space."""
        assert hasattr(self, "_Wh")
        assert hasattr(self, "_WhSub")
        if not hasattr(self, "_backward_assigner"):
            assigning_spaces = [None] * len(self._WhSub)
            for key, sub_space in self._WhSub.items():
                assigning_spaces[self._field_association[key]] = sub_space
            receiving_space = self._Wh
            self._backward_assigner = dlfn.FunctionAssigner(receiving_space,
                                                            assigning_spaces)
        return self._backward_assigner

    def _get_backward_subspace_assigners(self):
        """Returns a dictionary of function assigners which assign from one
        subspace to a component of the joint function space."""
        assert hasattr(self, "_Wh")
        assert hasattr(self, "_WhSub")
        if not hasattr(self, "_backward_subspace_assigners"):
            self._backward_subspace_assigners = dict()
            for key, assigning_space in self._WhSub.items():
                receiving_space = self._Wh.sub(self._field_association[key])
                self._backward_subspace_assigners[key] = dlfn.FunctionAssigner(receiving_space,
                                                                               assigning_space)
        return self._backward_subspace_assigners

    def _get_forward_assigner(self):
        """Returns a function assigner which assigns from the joint function
        space to all subspaces."""
        assert hasattr(self, "_Wh")
        assert hasattr(self, "_WhSub")
        if not hasattr(self, "_forward_assigner"):
            receiving_spaces = [None] * len(self._WhSub)
            for key, sub_space in self._WhSub.items():
                receiving_spaces[self._field_association[key]] = sub_space
            assigning_space = self._Wh
            self._forward_assigner = dlfn.FunctionAssigner(receiving_spaces,
                                                           assigning_space)
        return self._forward_assigner

    def _get_forward_subspace_assigners(self):
        """Returns a dictionary of function assigners which assign from one
        component of the joint function space to a subspace."""
        assert hasattr(self, "_Wh")
        assert hasattr(self, "_WhSub")
        if not hasattr(self, "_forward_subspace_assigners"):
            self._forward_subspace_assigners = dict()
            for key, receiving_space in self._WhSub.items():
                assigning_space = self._Wh.sub(self._field_association[key])
                self._forward_subspace_assigners[key] = \
                    dlfn.FunctionAssigner(receiving_space, assigning_space)
        return self._forward_subspace_assigners

    def _picard_linerization_convective_term(self, u, v, w):
        assert hasattr(self, "_equation_coefficients")
        if self._equation_coefficients["convective_term"] is None:  # pragma: no cover
            raise RuntimeError()
        assert isinstance(u, self._form_function_types)
        assert isinstance(v, self._form_trial_function_types)
        assert isinstance(w, self._form_trial_function_types)

        if self._form_convective_term is WeakFormConvectiveTerm.standard_form:
            return self._equation_coefficients["convective_term"] * dot(dot(grad(v), u), w)
        elif self._form_convective_term is WeakFormConvectiveTerm.rotational_form:
            if self._space_dim == 2:
                curl_u = curl(u)
                return self._equation_coefficients["convective_term"] * dot(dlfn.as_vector([-curl_u * v[1], curl_u * v[0]]), w)
            elif self._space_dim == 3:  # pragma: no cover
                return self._equation_coefficients["convective_term"] * dot(cross(curl(u), v), w)
        elif self._form_convective_term is WeakFormConvectiveTerm.divergence_form:
            return self._equation_coefficients["convective_term"] * \
                    (dot(dot(grad(v), u), w) + self._one_half * dot(div(u) * v, w))
        elif self._form_convective_term is WeakFormConvectiveTerm.skew_symmetric_form:
            return self._equation_coefficients["convective_term"] * self._one_half *\
                    (dot(dot(grad(v), u), w) - dot(dot(grad(w), u), v))

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
        if hasattr(self, "_constrained_domain"):
            self._Wh = dlfn.FunctionSpace(self._mesh, mixedElement,
                                          constrained_domain=self._constrained_domain)
        else:
            self._Wh = dlfn.FunctionSpace(self._mesh, mixedElement)
        self._n_dofs = self._Wh.dim()

        assert hasattr(self, "_n_cells")
        print("Number of cells {0}, number of DoFs: {1}".format(self._n_cells, self._n_dofs))

    def _setup_boundary_conditions(self):
        assert hasattr(self, "_Wh")
        assert hasattr(self, "_boundary_markers")
        # empty dirichlet bcs
        self._dirichlet_bcs = []
        # velocity part
        if hasattr(self, "_velocity_bcs"):
            velocity_space = self._Wh.sub(self._field_association["velocity"])
            self._setup_velocity_boundary_conditions(self._dirichlet_bcs,
                                                     self._velocity_bcs,
                                                     velocity_space)
        # pressure part
        if hasattr(self, "_pressure_bcs"):
            pressure_space = self._Wh.sub(self._field_association["pressure"])
            self._setup_pressure_boundary_conditions(self._dirichlet_bcs,
                                                     self._pressure_bcs,
                                                     pressure_space)
        if len(self._dirichlet_bcs) == 0:
            assert hasattr(self, "_constrained_domain")

    def _setup_velocity_boundary_conditions(self, dirichlet_bc_collection,
                                            velocity_bcs, velocity_space):
        assert isinstance(dirichlet_bc_collection, list)
        assert isinstance(velocity_bcs, list)
        assert isinstance(velocity_space, dlfn.FunctionSpace)

        for bc in velocity_bcs:
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
                dirichlet_bc_collection.append(bc_object)

            elif bc_type is VelocityBCType.no_normal_flux:
                # compute normal vector of boundary
                bndry_normal = boundary_normal(self._mesh, self._boundary_markers, bndry_id)
                # find associated component
                bndry_normal = np.array(bndry_normal)
                normal_component_index = int(np.abs(bndry_normal).argmax())
                # check that direction is either e_x, e_y or e_z
                assert abs(abs(bndry_normal[normal_component_index]) - 1.0) < 5.0e-15
                assert all([abs(bndry_normal[d]) < 5.0e-15 for d in range(self._space_dim) if d != normal_component_index])
                # construct boundary condition on subspace
                bc_object = dlfn.DirichletBC(velocity_space.sub(normal_component_index),
                                             self._null_scalar, self._boundary_markers,
                                             bndry_id)
                dirichlet_bc_collection.append(bc_object)

            elif bc_type is VelocityBCType.no_tangential_flux:
                # compute normal vector of boundary
                bndry_normal = boundary_normal(self._mesh, self._boundary_markers, bndry_id)
                # find associated component
                bndry_normal = np.array(bndry_normal)
                normal_component_index = int(np.abs(bndry_normal).argmax())
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
                    dirichlet_bc_collection.append(bc_object)

            elif bc_type is VelocityBCType.constant:
                assert isinstance(value, (tuple, list))
                const_function = dlfn.Constant(value)
                bc_object = dlfn.DirichletBC(velocity_space, const_function,
                                             self._boundary_markers, bndry_id)
                dirichlet_bc_collection.append(bc_object)

            elif bc_type is VelocityBCType.constant_component:
                assert isinstance(value, float)
                const_function = dlfn.Constant(value)
                bc_object = dlfn.DirichletBC(velocity_space.sub(component_index),
                                             const_function,
                                             self._boundary_markers, bndry_id)
                dirichlet_bc_collection.append(bc_object)

            elif bc_type is VelocityBCType.function:
                assert isinstance(value, dlfn.Expression)
                bc_object = dlfn.DirichletBC(velocity_space, value,
                                             self._boundary_markers, bndry_id)
                dirichlet_bc_collection.append(bc_object)

            elif bc_type is VelocityBCType.function_component:
                assert isinstance(value, dlfn.Expression)
                bc_object = dlfn.DirichletBC(velocity_space.sub(component_index),
                                             value,
                                             self._boundary_markers, bndry_id)
                dirichlet_bc_collection.append(bc_object)

            else:  # pragma: no cover
                raise RuntimeError()

    def _setup_pressure_boundary_conditions(self, dirichlet_bc_collection,
                                            pressure_bcs, pressure_space):
        assert isinstance(dirichlet_bc_collection, list)
        assert isinstance(pressure_bcs, list)
        assert isinstance(pressure_space, dlfn.FunctionSpace)
        for bc in pressure_bcs:
            # unpack values
            if len(bc) == 3:
                bc_type, bndry_id, value = bc
            else:  # pragma: no cover
                raise RuntimeError()
            # create dolfin.DirichletBC object
            if bc_type is PressureBCType.constant:
                assert isinstance(value, float)
                const_function = dlfn.Constant(value)
                bc_object = dlfn.DirichletBC(pressure_space, const_function,
                                             self._boundary_markers, bndry_id)
                dirichlet_bc_collection.append(bc_object)

            elif bc_type is PressureBCType.function:
                assert isinstance(value, dlfn.Expression)
                bc_object = dlfn.DirichletBC(pressure_space, value,
                                             self._boundary_markers, bndry_id)
                dirichlet_bc_collection.append(bc_object)

            elif bc_type is PressureBCType.mean_value:
                assert bndry_id is None
                assert isinstance(value, float)
                self._mean_pressure_value = value
            else:  # pragma: no cover
                raise RuntimeError()

    def _viscous_term(self, u, v):
        assert hasattr(self, "_equation_coefficients")
        if self._equation_coefficients["viscous_term"] is None:  # pragma: no cover
            raise RuntimeError()
        assert isinstance(u, self._form_function_types)
        assert isinstance(v, self._form_function_types)

        if self._form_viscous_term is WeakFormViscousTerm.traction_form:
            return self._equation_coefficients["viscous_term"] * \
                    inner((grad(u) + grad(u).T), self._one_half * (grad(v) + grad(v).T))
        elif self._form_viscous_term is WeakFormViscousTerm.reduced_form:
            return self._equation_coefficients["viscous_term"] * inner(grad(u), grad(v))

    @property
    def field_association(self):
        assert hasattr(self, "_field_association")
        return self._field_association

    def set_angular_velocity(self, angular_velocity):
        """
        Specifies the body force.

        Parameters
        ----------
        body_force : auxiliary_classes.AngularVelocityVector
            The angular velocity vector.
        """
        assert isinstance(angular_velocity, AngularVelocityVector)
        assert angular_velocity.space_dim == self._space_dim
        self._angular_velocity = angular_velocity

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
        self._body_force.rename("body_force", "")

    def set_periodic_boundary_conditions(self, constrained_domain,
                                         constrained_boundary_ids):
        """Set constraints due to the periodic boundary conditions of the
        problem.
        """
        assert isinstance(constrained_domain, dlfn.SubDomain)
        assert isinstance(constrained_boundary_ids, (tuple, list))
        assert all(isinstance(i, int) for i in constrained_boundary_ids)
        self._constrained_domain = constrained_domain
        self._constrained_boundary_ids = constrained_boundary_ids

    def set_boundary_conditions(self, bcs, internal_constraints=None):
        """Set the boundary conditions of the problem.
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
        An optional argument allows to also specify internal constraints.
        """
        assert isinstance(bcs, (list, tuple))
        if internal_constraints is not None:
            assert isinstance(internal_constraints, (list, tuple))
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
            if hasattr(self, "_constrained_domain"):
                assert bc[1] not in self._constrained_boundary_ids
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
        if not hasattr(self, "_constrained_domain"):
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

        # internal constraints
        if internal_constraints is not None:
            # check format of internal constraints
            for bc in internal_constraints:
                self._check_boundary_condition_format(bc, True)

            # check format of internal constraints
            velocity_constraints = []
            pressure_constraints = []
            for bc in internal_constraints:
                #  check that there is no conflict between bcs and constraints
                assert bc[1] not in velocity_bc_ids
                assert bc[1] not in traction_bc_ids
                assert bc[1] not in pressure_bc_ids

                if isinstance(bc[0], VelocityBCType):
                    velocity_constraints.append(bc)
                elif isinstance(bc[0], TractionBCType):  # pragma: no cover
                    raise NotImplementedError()
                elif isinstance(bc[0], PressureBCType):
                    pressure_constraints.append(bc)
            # add internal constraint to bc list
            velocity_bcs += velocity_constraints
            pressure_bcs += pressure_constraints

        # boundary conditions accepted
        self._velocity_bcs = velocity_bcs
        if len(traction_bcs) > 0:
            self._traction_bcs = traction_bcs
            self._form_viscous_term = WeakFormViscousTerm.traction_form
        if len(pressure_bcs) > 0:
            self._pressure_bcs = pressure_bcs

    def set_equation_coefficients(self, input_coefficients):
        assert isinstance(input_coefficients, dict)
        possible_keys = ("convective_term", "coriolis_term", "euler_term",
                         "pressure_term", "viscous_term", "body_force_term")
        for key in input_coefficients.keys():
            assert key in possible_keys
        if not hasattr(self, "_equation_coefficients"):
            self._equation_coefficients = dict()

            for key, value in input_coefficients.items():
                if value is not None:
                    assert isinstance(value, float)
                    assert math.isfinite(value)
                    assert value > 0.0
                    self._equation_coefficients[key] = dlfn.Constant(value)
                else:
                    self._equation_coefficients[key] = None
            for key in possible_keys:  # pragma: no cover
                if key not in self._equation_coefficients:
                    self._equation_coefficients[key] = None
        else:  # pragma: no cover
            for key, value in self._equation_coefficients.items():
                assert key in input_coefficients
                desired_value = input_coefficients[key]
                if value is not None:
                    assert desired_value is not None
                    value.assign(desired_value)

    @property
    def sub_space_association(self):
        assert hasattr(self, "_sub_space_association")
        return self._sub_space_association

    @property
    def solution(self):
        return self._solution

    def solve(self):  # pragma: no cover
        """
        Solves the nonlinear problem.
        """
        raise NotImplementedError("You are calling a purely virtual method.")


class StationarySolverBase(SolverBase):
    """
    Class to solve stationary fluid flow of an incompressible fluid using
    P2-P1 finite elements.
    """

    def __init__(self, mesh, boundary_markers, form_convective_term, tol=1e-10, maxiter=50,
                 tol_picard=1e-2, maxiter_picard=10):

        super().__init__(mesh, boundary_markers, form_convective_term)

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
        assert hasattr(self, "_equation_coefficients")
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
        # weak forms
        # mass balance
        F_mass = -self._divergence_term(sol_v, q) * dV
        # momentum balance
        F_momentum = (self._convective_term(sol_v, w)
                      - self._divergence_term(w, sol_p)
                      + self._viscous_term(sol_v, w)) * dV
        # add boundary tractions
        F_momentum = self._add_boundary_tractions(F_momentum, w)
        # add body force term
        F_momentum = self._add_body_forces(F_momentum, w)
        # add Coriolis acceleration
        F_momentum = self._add_coriolis_acceleration(F_momentum, sol_v, w)
        # add Euler acceleration
        F_momentum = self._add_euler_acceleration(F_momentum, w)

        self._F = F_mass + F_momentum

        # linearization using Picard's method
        J_picard_mass = -self._divergence_term(v, q) * dV
        J_picard_momentum = (self._picard_linerization_convective_term(sol_v, v, w)
                             - self._divergence_term(w, p)
                             + self._viscous_term(v, w)) * dV
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


class InstationarySolverBase(SolverBase):

    def __init__(self, mesh, boundary_markers, form_convective_term, time_stepping, tol=1e-10, max_iter=50):

        super().__init__(mesh, boundary_markers, form_convective_term)

        # input check
        assert isinstance(max_iter, int)
        assert max_iter > 0
        assert isinstance(tol, float)
        assert tol > 0.0
        assert isinstance(time_stepping, DiscreteTime)

        # time stepping scheme
        assert hasattr(time_stepping, "n_levels")
        self._time_stepping = time_stepping

        # set numerical tolerances
        self._tol = tol
        self._maxiter = max_iter

    def _advance_solution(self):
        """Advance solution objects in time."""
        assert hasattr(self, "_solutions")
        for i in range(len(self._solutions), 1, -1):
            self._solutions[i-1].assign(self._solutions[i-2])

    def _setup_function_spaces(self):
        """Class method setting up function spaces."""
        super()._setup_function_spaces()
        # create solution
        self._solutions = []
        for i in range(self._time_stepping.n_levels() + 1):
            name = i * "old" + (i > 0) * "_" + "solution"
            self._solutions.append(dlfn.Function(self._Wh, name=name))

    def _setup_problem(self):  # pragma: no cover
        """
        Purely virtual method for setting up solver objects of the instationary problem.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    def _set_time(self, next_time=None, current_time=None):
        """Set time of boundary condition objects and body force."""
        # input check
        if next_time is None:
            next_time = self._time_stepping.next_time
        if current_time is None:
            current_time = self._time_stepping.current_time
        assert isinstance(next_time, float)
        assert isinstance(current_time, float)
        assert next_time > current_time

        # auxiliary function
        def modify_time(expression, time=next_time):
            # modify time
            if isinstance(expression, dlfn.Expression):
                if hasattr(expression, "time"):
                    expression.time = next_time
                elif hasattr(expression, "t"):
                    expression.t = next_time
        # velocity boundary conditions
        if hasattr(self, "_velocity_bcs"):
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
        # traction boundary conditions at current time
        if hasattr(self, "_current_traction_bcs"):
            for bc in self._current_traction_bcs:
                # unpack values
                if len(bc) == 3:
                    value = bc[2]
                elif len(bc) == 4:
                    value = bc[3]
                else:  # pragma: no cover
                    raise RuntimeError()
                # modify time
                modify_time(value, current_time)
        # body force at current time
        if hasattr(self, "_current_body_force"):
            modify_time(self._current_body_force, current_time)

    def _solve_time_step(self):  # pragma: no cover
        """
        Purely virtual method for solving the one time step of the problem.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    def _update_time_stepping_coefficients(self):  # pragma: no cover
        """
        Purely virtual method for updating the coefficients of the time stepping
        scheme.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    def advance_time(self):
        """Advance relevant objects by one time step."""
        self._advance_solution()

    def set_initial_conditions(self, initial_conditions):
        """Setup the initial conditions of the problem."""
        # input check
        assert isinstance(initial_conditions, dict)
        assert "velocity" in initial_conditions
        # check that function spaces exist
        if not all(hasattr(self, attr) for attr in ("_Wh", "_solutions")):
            self._setup_function_spaces()
        # split functions
        velocity, pressure = self._solutions[0].split()
        old_velocity, old_pressure = self._solutions[1].split()

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
        velocity_space = self._get_subspace("velocity")
        projected_velocity_condition = dlfn.project(velocity_expression,
                                                    velocity_space)
        self._assign_function(old_velocity, projected_velocity_condition)
        self._assign_function(velocity, projected_velocity_condition)
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
            pressure_space = self._get_subspace("pressure")
            projected_pressure_condition = dlfn.project(pressure_expression,
                                                        pressure_space)
            self._assign_function(old_pressure, projected_pressure_condition)
            self._assign_function(pressure, projected_pressure_condition)
        # TODO: Implement Poisson equation for the initial pressure

    def solve(self):
        """Solves the problem for one time step."""
        # setup problem
        if not all(hasattr(self, attr) for attr in self._required_objects):
            self._setup_problem()

        # update time
        self._set_time()

        # update coefficients if necessary
        if self._time_stepping.coefficients_changed:
            self._update_time_stepping_coefficients()

        # perform one time
        self._solve_time_step()

        if hasattr(self, "_mean_pressure_value"):
            _, pressure = self.solution.split()
            # compute mean value
            dV = dlfn.Measure("dx", domain=self._mesh)
            mean_pressure_value = dlfn.assemble(pressure * dV) / dlfn.assemble(dlfn.Constant(1.0) * dV)
            # compute pressure shift
            pressure_shift = dlfn.Constant(mean_pressure_value - self._mean_pressure_value)
            # define modified pressure
            modified_pressure = pressure - pressure_shift
            # project modified pressure
            pressure_space = self._get_subspace("pressure")
            corrected_pressure = dlfn.project(modified_pressure, pressure_space)
            self._assign_function(pressure,
                                  {"pressure": corrected_pressure})

    @property
    def solution(self):
        return self._solutions[0]
