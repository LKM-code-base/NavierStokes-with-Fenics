#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dolfin import NonlinearProblem
from dolfin import SystemAssembler
from enum import Enum

class CustomNonlinearProblem(NonlinearProblem):
    """Class for interfacing with not only :py:class:`NewtonSolver`."""
    def __init__(self, F, bcs, J, J_pc=None):
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
                Bilinear form representing system Jacobian for Newton iteration.
            J_pc (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                Bilinear form representing Jacobian optionally passed to
                preconditioner instead of ``J``. In case of Navier-Stokes, 
                stabilized (0,0)-block can be passed to (0,0)-KSP solver.

        All the arguments should be given on the common mixed function space.
        """
        super(CustomNonlinearProblem, self).__init__()

        # assembler for Newton/Picard system
        self.assembler = SystemAssembler(J, F, bcs)

        # assembler for preconditioner of (0,0)-block
        if J_pc is not None:
            self.assembler_pc = SystemAssembler(J_pc, F, bcs)
        else:
            self.assembler_pc = None

        # store forms/bcs for later
        self.forms = {
            "F": F,
            "J": J
        }
        self._bcs = bcs

    def get_form(self, key):
        form = self.forms.get(key)
        if form is None:
            raise AttributeError("Form '%s' requested by preconditioner not available" % key)
        return form

    def function_space(self):
        return self.forms["F"].arguments()[0].function_space()

    def F(self, b, x):
        self.assembler.assemble(b, x)

    def J(self, A, x):
        self.assembler.assemble(A)