#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from os import path

import dolfin as dlfn

import numpy as np

from navier_stokes_solver import VelocityBCType, PressureBCType, TractionBCType

from navier_stokes_solver import StationaryNavierStokesSolver as Solver

class StationaryNavierStokesProblem():
    """
    Class to simulate stationary fluid flow using the `StationaryNavierStokesSolve`.
    
    Parameters
    ----------
    main_dir: str (optional)
        Directory to save the results.
    tol: float (optional)
        Final tolerance .
    maxiter: int (optional)
        Maximum number of iterations in total.
    tol_picard: float (optional)
        Tolerance for the Picard iteration.
    maxiter_picard: int (optional)
        Maximum number of Picard iterations.
    """
    def __init__(self, main_dir = None, tol = 1e-10, maxiter = 50,
                 tol_picard = 1e-2, maxiter_picard = 10):
        
        # input check
        assert all(isinstance(i, int) and i > 0 for i in (maxiter, maxiter_picard))
        assert all(isinstance(i, float) and i > 0.0 for i in (tol_picard, tol_picard))
    
        # set write and read directory
        if main_dir is None:
            self._main_dir = os.getcwd()
        else:
            assert isinstance(main_dir, str)
            assert path.exist(main_dir)
            self._main_dir = main_dir
        self._results_dir = path.join(self._main_dir, "results")
    
        # set numerical tolerances
        self._tol_picard = tol_picard
        self._maxiter_picard = maxiter_picard
        self._tol = tol
        self._maxiter = maxiter
        
        # setting discretization parameters
        # polynomial degree
        self._p_deg = 1 
        # quadrature degree
        q_deg = self._p_deg + 2 
        dlfn.parameters["form_compiler"]["quadrature_degree"] = q_deg
        
    def _get_filename(self):
        """
        Class method returning a filename for the given set of parameters.
        
        The method also updates the parameter file.
        
        Parameters
        ----------
        Re : float
            Kinetic Reynolds numbers.
        Fr : float (optional)
            Froude number.
        suffix : str (optional)
            Opitonal filename extension.

        Returns
        ----------
        fname : str
            filename
        """
        
        # input check
        assert hasattr(self, "_Re")
            
        assert hasattr(self, "_problem_name")
        problem_name = self._problem_name
        suffix = ".xdmf"
        
        fname = problem_name + "_Re" + "{0:01.4e}".format(self._Re) 
        if hasattr(self, "_Fr") and self._Fr is not None:
            fname += "_Fr" + "{0:01.4e}".format(self._Fr)
        fname += suffix
        
        return path.join(self._results_dir, fname)
    
    def _write_xdmf_file(self):
        # get filename
        fname = self._get_filename()
        assert fname.endswith(".xdmf")
        
        # create results directory
        assert hasattr(self, "_results_dir")
        if not path.exists(self._results_dir):
            os.makedirs(self._results_dir)
            
        assert hasattr(self, "_navier_stokes_solver")
        solver = self._navier_stokes_solver
        solution = solver.solution
        # serialize
        with dlfn.XDMFFile(fname) as results_file:
            results_file.parameters["flush_output"] = True
            results_file.parameters["functions_share_mesh"] = True
            solution_components = solution.split()
            for index, name in solver.sub_space_association.items():
                solution_components[index].rename(name, "")
                results_file.write(solution_components[index], 0.)
                
    def set_parameters(self, Re = 1.0, Fr = None):
        """
        Sets up the parameters of the model by creating or modifying class
        objects.
        
        Parameters
        ----------
        Re : float
            Kinetic Reynolds numbers.
        Fr : float
            Froude number.
        """
        assert isinstance(Re, float) and Re > 0.0
        self._Re = Re

        if Fr is not None:
            assert isinstance(Fr, float) and Fr > 0.0
        self._Fr = Fr
    
    def setup_mesh(self):
        raise NotImplementedError()

    def set_boundary_conditions(self):
        raise NotImplementedError()
        
    def set_body_force(self):
        pass
    
    def write_boundary_markers(self):
        assert hasattr(self, "_boundary_markers")
        assert hasattr(self, "_problem_name")
        
        # create results directory
        assert hasattr(self, "_results_dir")
        if not path.exists(self._results_dir):
            os.makedirs(self._results_dir)
        
        problem_name = self._problem_name
        suffix = ".pvd"
        fname = problem_name + "_BoundaryMarkers"
        fname += suffix
        fname = path.join(self._results_dir, fname)
        
        dlfn.File(fname) << self._boundary_markers
    
    def solve_problem(self):

        # setup mesh        
        self.setup_mesh()
        assert self._mesh is not None
        self._space_dim = self._mesh.geometry().dim()
        self._n_cells = self._mesh.num_cells()

        # setup boundary conditions
        self.set_boundary_conditions()
        
        # setup has body force
        self.set_body_force()
                
        # setup parameters
        if not hasattr(self, "_Re"):
            self.set_parameters()
        
        # create solver object
        if not hasattr(self, "_navier_stokes_solver"):
            self._navier_stokes_solver =  Solver(self._mesh, self._boundary_markers,
                                                 self._tol, self._maxiter,
                                                 self._tol_picard, self._maxiter_picard)
        
        # pass boundary conditions
        self._navier_stokes_solver.set_boundary_conditions(self._bcs)
        
        # pass dimensionless numbers
        self._navier_stokes_solver.set_dimensionless_numbers(self._Re, self._Fr)
        
        # pass body force
        if hasattr(self, "_body_force"):
            self._navier_stokes_solver.set_body_force(self._body_force)

        try:
            # solve problem
            dlfn.info("Solving problem with Re = {0:.2f} and Fr = {1:0.2f}".format(self._Re, self._Fr))
            self._navier_stokes_solver.solve()
        except:
            dlfn.info("Solving problem for Re = {0:.2f} and Fr = {1:0.2f} without ".format(self._Re, self._Fr) +
                      "suitable initial guess failed.")
            dlfn.info("Solving problem with parameter continuation...")
            # parameter continuation
            for Re in np.logspace(0.0, np.log10(self._Re), num=10, endpoint=True):
                # pass dimensionless numbers
                self._navier_stokes_solver.set_dimensionless_numbers(Re, self._Fr)
                # solve problem
                dlfn.info("Solving problem with Re = {0:.2f} and Fr = {1:0.2f}".format(Re, self._Fr))
                self._navier_stokes_solver.solve()

        # write XDMF-files
        self._write_xdmf_file()
