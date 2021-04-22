#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from dolfin import grad, div, dot, inner
from auxiliary_modules import CustomNonlinearProblem

def a(phi, psi):
    """
    Viscous operator: :math:`(\\nabla \\boldsymbol\phi, \\nabla \\boldsymbol\psi)`
    
    Parameters
    ----------
    phi : dolfin.Function
        Function.
    psi : dolfin.Function
        Test function.
    """
    return inner(grad(phi), grad(psi))

def b(phi, psi):
    """
    Divergence operator: :math:`(\\nabla \cdot \\boldsymbol\phi, \psi)`
    
    Parameters
    ----------
    phi : dolfin.Function
        Function (from velocity space).
    psi : dolfin.Function
        Test function (from pressure space).
    """
    return inner(div(phi), psi)

def c(phi, chi, psi):
    """
    Transport operator in standard form: :math:`(
    \\boldsymbol\phi \cdot \\nabla \\boldsymbol\chi, \\boldsymbol\psi)`

    Parameters
    ----------
    phi : dolfin.Function
        Function (from velocity space): advection field.
    chi : dolfin.Function
        Function (from scalar or tensor function space): advected field.
    psi : dolfin.Function
        Test function (from space corresponding to phi) .
    """
    return dot(dot(grad(chi), phi), psi)

class CavityFlowProblem():
    """
    Class to simulate stationary micropolar fluid in a quadratic cavity using
    P2-P1 finite elements. The system is solved using Dirichlet boundary at
    all boundaries.
    
    Parameters
    ----------
    npoints: int
        Number of points used for meshing.
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
    def __init__(self, n_points, main_dir = None, tol = 1e-10, maxiter = 50,
                 tol_picard = 1e-2, maxiter_picard = 10):
        
        # input check
        assert all(isinstance(i, int) and i > 0 for i in (n_points, maxiter, maxiter_picard))
        assert all(isinstance(i, float) and i > 0.0 for i in (tol_picard, tol_picard))
    
        # set number of discretization points
        self._n_points = n_points
    
        # set write and read directory
        import os
        from os import path
        if main_dir is None:
            self._main_dir = os.getcwd()
        else:
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
        
        # association of mixed element formulation
        self._dof_association = {0: "velocity", 1: "pressure"}
    
        # create mesh
        from grid_generator import square_cavity
        self._mesh, self._boundary_markers = square_cavity(2, self._n_points)
        self._space_dim = self._mesh.geometry().dim()
        self._n_cells = self._mesh.num_cells()
        
    def _create_function_spaces(self):
        """
        Class method setting up function spaces.
        """
        assert hasattr(self, "_mesh")
        cell = self._mesh.ufl_cell()
        # element formulation
        elemV = dlfn.VectorElement("CG", cell, self._p_deg + 1)
        elemP = dlfn.FiniteElement("CG", cell, self._p_deg)
        mixedElement = dlfn.MixedElement([elemV , elemP])
        
        # mixed function space
        self._Wh = dlfn.FunctionSpace(self._mesh, mixedElement)
        
        self._n_dofs = self._Wh.dim()
        assert hasattr(self, "_n_cells")
        dlfn.info("Number of cells {0}, number of DoFs: {1}".format(self._n_cells, self._n_dofs))

    def _get_fname(self, Re, Fr = None):
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
        from os import path
        
        # input check
        assert isinstance(Re, float) and Re > 0.0 
        if Fr is not None:
            assert isinstance(Re, float) and Re > 0.0 
            
        problem = "cavity"
        suffix = ".xdmf"
        
        fname = problem + "_Re" + "{0:01.4e}".format(Re) 
        if Fr is not None:
            fname += "_Re" + "{0:01.4e}".format(Fr)
        fname += suffix
        
        return path.join(self._results_dir, fname)

    
    def _get_dirichlet_bcs(self):
        """
        Returns Dirichlet boundary conditions of the problem as a list. For velocity
        and pressure standard boundary conditions are applied. That means no-slip
        boundary conditions are used at the walls and a tangential velocity is
        used at the wall. For the angular velocity field homogeneous boundary 
        conditions are used everywhere.
        
        Returns
        ----------
        dirichlet_bcs : list
            Dirichlet boundary conditions as a list of ``dolfin.DirichletBC``
            objects.
        """
        from grid_generator import RectangleBoundaryMarkers as BoundaryMarkers
        assert hasattr(self, "_Wh") and hasattr(self, "_space_dim")
        
        # dirichlet boundary conditions for velocity
        Vh, Ph = self._Wh.split()
        null_scalar = dlfn.Constant(0.)
        null_vector = dlfn.Constant((0., ) *  self._space_dim)
        inlet_velocity = dlfn.Constant((1.0 , 0.0) if self._space_dim == 2 else (1.0 , 0.0, 0.0))
        dirichlet_bcs = []
        
        # get boundary markers
        top = BoundaryMarkers.top.value
        bottom = BoundaryMarkers.bottom.value
        left = BoundaryMarkers.left.value
        right = BoundaryMarkers.right.value
        
        # velocity boundary conditions
        dirichlet_bcs.append(dlfn.DirichletBC(Vh, null_vector, self._boundary_markers, left))
        dirichlet_bcs.append(dlfn.DirichletBC(Vh, null_vector, self._boundary_markers, right))
        dirichlet_bcs.append(dlfn.DirichletBC(Vh, null_vector, self._boundary_markers, bottom))
        dirichlet_bcs.append(dlfn.DirichletBC(Vh, inlet_velocity, self._boundary_markers, top))
        
        return dirichlet_bcs
    
    def _update_parameters(self, Re, Fr = None):
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
    
    def _setup_stationary_problem(self):
        """
        Method setting up non-linear solver objects for the stationary problem.
        """
        assert hasattr(self, "_mesh")
        assert hasattr(self, "_boundary_markers")
    
        self._create_function_spaces()
        
        dirichlet_bcs = self._get_dirichlet_bcs()
    
        # creating test and trial functions
        (v, p) = dlfn.TrialFunctions(self._Wh)
        (w, q) = dlfn.TestFunctions(self._Wh)
        
        # solution
        self._sol = dlfn.Function(self._Wh)
        sol_v, sol_p = dlfn.split(self._sol)
    
        # volume element
        dV = dlfn.Measure("dx", domain = self._mesh)
    
        # dimensionless parameters
        assert all(hasattr(self, attr) for attr in ("_Re", ))
        Re = self._Re
        # Fr = self._Fr

        # nonlinear forms
        F_mass = - b(sol_v, q)  * dV
        F_momentum = ( c(sol_v, sol_v, w) - b(w, sol_p) + (1. / Re) * a(sol_v, w) ) * dV

        self._F = F_mass + F_momentum
    
        # linearization using Picard's method
        J_picard_mass = -b(v, q) * dV
        J_picard_momentum = ( c(sol_v, v, w) - b(w, p) + (1. / Re) * a(v, w) ) * dV
        self._J_picard = J_picard_mass + J_picard_momentum
    
        # linearization using Newton's method
        self._J_newton = dlfn.derivative(self._F, self._sol)
    
        # setup linear solver
        linear_solver = dlfn.PETScLUSolver()
        comm = dlfn.MPI.comm_world
        factory = dlfn.PETScFactory.instance()
        self._solver = dlfn.NewtonSolver(comm, linear_solver, factory)
    
        # setup problem with Picard linearization
        self._picard_problem = CustomNonlinearProblem(self._F,
                                                      dirichlet_bcs,
                                                      self._J_picard)
    
        # setup problem with Newton linearization        
        self._newton_problem = CustomNonlinearProblem(self._F,
                                                      dirichlet_bcs,
                                                      self._J_newton)

    def _write_xdmf_file(self, Re, Fr):

        assert isinstance(Re, float) and Re > 0.0 
        
        if Fr is not None:
            assert isinstance(Fr, float) and Fr > 0.0

        assert hasattr(self, "_dof_association")
        assert hasattr(self, "_main_dir")
        assert hasattr(self, "_sol")
        
        # get filename
        fname = self._get_fname(Re, Fr)
        assert fname.endswith(".xdmf")
        
        # create results directory
        from os import path
        if not path.exists(self._results_dir):
            import os
            os.makedirs(self._results_dir)
            
        # serialize
        with dlfn.XDMFFile(fname) as results_file:
            results_file.parameters["flush_output"] = True
            results_file.parameters["functions_share_mesh"] = True
            sol_components = self._sol.split()
            for i, name in self._dof_association.items():
                sol_components[i].rename(name, "")
                results_file.write(sol_components[i], 0.)

    def solve_problem(self, Re = 1.0, Fr = None):
        """
        Solve the nonlinear problem for a given set dimensionless parameters.
        
        Parameters
        ----------
        Re : float
            Kinetic Reynolds numbers.
        Fr : float
            Froude number.
        """
        assert isinstance(Re, float) and Re > 0.0
        if Fr is not None:
            assert isinstance(Fr, float) and Fr > 0.0

        # update parameters
        self._update_parameters(Re, Fr)
                
        # setup problem        
        if not all(hasattr(self, attr) for attr in ("_solver",
                                                    "_picard_problem", 
                                                    "_newton_problem",
                                                    "_sol")):
            self._setup_stationary_problem()
    
        # Picard iteration
        dlfn.info("Starting Picard iteration...")
        self._solver.parameters["maximum_iterations"] = self._maxiter_picard
        self._solver.parameters["absolute_tolerance"] = self._tol_picard
        self._solver.solve(self._picard_problem, self._sol.vector())
        
        # Newton's method
        dlfn.info("Starting Newton iteration...")
        self._solver.parameters["absolute_tolerance"] = self._tol
        self._solver.parameters["maximum_iterations"] = self._maxiter
        self._solver.parameters["error_on_nonconvergence"] = False
        self._solver.solve(self._newton_problem, self._sol.vector())
        
        # write XDMF-files
        self._write_xdmf_file(Re, Fr)