#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from auxiliary_methods import extract_all_boundary_markers
import dolfin
from dolfin import NonlinearProblem
from dolfin import SubDomain
from dolfin import SystemAssembler


__all__ = ["CustomNonlinearProblem"]


class CustomNonlinearProblem(NonlinearProblem):
    """Class for interfacing with not only :py:class:`NewtonSolver`."""
    def __init__(self, F, bcs, J):
        """Return subclass of :py:class:`dolfin.NonlinearProblem` suitable
        for :py:class:`NewtonSolver` based on
        :py:class:`field_split.BlockPETScKrylovSolver` and PCD
        preconditioners from :py:class:`preconditioners.PCDPreconditioner`.

        *Arguments*
            F (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                Linear form representing the equation.
            bcs (:py:class:`list` of :py:class:`dolfin.DirichletBC`)
                Boundary conditions applied to ``F``, ``J``, and ``J_pc``.
            J (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                Bilinear form representing system Jacobian for the iteration.

        All the arguments should be given on the common mixed function space.
        """
        super(CustomNonlinearProblem, self).__init__()

        # assembler for Newton/Picard system
        self.assembler = SystemAssembler(J, F, bcs)

        # store forms/bcs for later
        self.forms = {
            "F": F,
            "J": J
        }
        self._bcs = bcs

    def get_form(self, key):
        form = self.forms.get(key)
        if form is None:
            raise AttributeError("Form '%s' requested by NonlinearProblem not "
                                 "available" % key)
        return form

    def function_space(self):
        return self.forms["F"].arguments()[0].function_space()

    def F(self, b, x):
        self.assembler.assemble(b, x)

    def J(self, A, x):
        self.assembler.assemble(A)


class PeriodicDomain(SubDomain):
    def __init__(self, width, boundary_markers, master_boundary_marker,
                 map_tol=1e-10):
        assert isinstance(map_tol, float)
        assert map_tol > 0.0
        super().__init__(map_tol)

        assert isinstance(boundary_markers, (dolfin.cpp.mesh.MeshFunctionSizet,
                                             dolfin.cpp.mesh.MeshFunctionInt))
        self._boundary_markers = boundary_markers

        assert isinstance(master_boundary_marker, int)
        all_boundary_markers = extract_all_boundary_markers(boundary_markers.mesh(),
                                                            boundary_markers)
        assert master_boundary_marker in all_boundary_markers
        self._master_boundary_marker = master_boundary_marker

    def inside(self, x, on_boundary):
        """Return True if `x` is located on the master edge and False
        else.
        """
        return self._

    def map(self, x_slave, x_master):
        """Map the coordinates of the support points (nodes) of the degrees
        of freedom of the slave to the coordinates of the corresponding
        master edge.
        """
        x_master[0] = x_slave[0] - 1.0
        x_master[1] = x_slave[1]
