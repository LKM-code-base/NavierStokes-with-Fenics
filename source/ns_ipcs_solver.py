#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from bdf_time_stepping import BDFTimeStepping
import dolfin as dlfn
from ns_solver_base import SolverBase, InstationarySolverBase, VelocityBCType, PressureBCType


class IPCSSolver(InstationarySolverBase):
    
    _required_objects = ("_Wh", "_Vh","_joint_solution","_velocities","_pressure", "_old_pressure")

    def __init__(self, mesh, boundary_markers, form_convective_term, time_stepping, tol=1e-10, max_iter=50):

        # input check
        assert isinstance(time_stepping, BDFTimeStepping)

        super().__init__(mesh, boundary_markers, form_convective_term,
                         time_stepping, tol, max_iter)
    
    def _acceleration_term(self, velocity_solutions, w):
        # input check
        assert isinstance(velocity_solutions, (list, tuple))
        #assert all(isinstance(x, self._form_function_types) for x in velocity_solutions)
        #assert isinstance(w, self._form_function_types)
        # step size
        k = self._next_step_size
        # time stepping coefficients
        alpha = self._alpha
        assert len(alpha) == len(velocity_solutions)
        # compute accelerations
        accelerations = []
        for i in range(len(alpha)):
            accelerations.append(alpha[i] * dlfn.dot(velocity_solutions[i], w))
        return sum(accelerations)
        
        #asserts need to be reworked for ipcs
    
    def _setup_boundary_conditions(self):
        assert hasattr(self, "_Wh")
        assert hasattr(self, "_boundary_markers")
        assert hasattr(self, "_velocity_bcs")
        # empty dirichlet bcs
        
        self._dirichlet_bcs = dict()
        self._dirichlet_bcs['velocity'] = []
        self._dirichlet_bcs['pressure'] = []

        # velocity part
        velocity_space = self._Vh['velocity']
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
                self._dirichlet_bcs['velocity'].append(bc_object)

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
                self._dirichlet_bcs['velocity'].append(bc_object)

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
                    self._dirichlet_bcs['velocity'].append(bc_object)

            elif bc_type is VelocityBCType.constant:
                assert isinstance(value, (tuple, list))
                const_function = dlfn.Constant(value)
                bc_object = dlfn.DirichletBC(velocity_space, const_function,
                                             self._boundary_markers, bndry_id)
                self._dirichlet_bcs['velocity'].append(bc_object)

            elif bc_type is VelocityBCType.constant_component:
                assert isinstance(value, float)
                const_function = dlfn.Constant(value)
                bc_object = dlfn.DirichletBC(velocity_space.sub(component_index),
                                             const_function,
                                             self._boundary_markers, bndry_id)
                self._dirichlet_bcs['velocity'].append(bc_object)

            elif bc_type is VelocityBCType.function:
                assert isinstance(value, dlfn.Expression)
                bc_object = dlfn.DirichletBC(velocity_space, value,
                                             self._boundary_markers, bndry_id)
                self._dirichlet_bcs['velocity'].append(bc_object)

            elif bc_type is VelocityBCType.function_component:
                assert isinstance(value, dlfn.Expression)
                bc_object = dlfn.DirichletBC(velocity_space.sub(component_index),
                                             value,
                                             self._boundary_markers, bndry_id)
                self._dirichlet_bcs['velocity'].append(bc_object)

            else:  # pragma: no cover
                raise RuntimeError()

        # pressure part
        pressure_space = self._Vh["pressure"]
        if hasattr(self, "_pressure_bcs"):
            for bc in self._pressure_bcs:
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
                    self._dirichlet_bcs['pressure'].append(bc_object)

                elif bc_type is PressureBCType.function:
                    assert isinstance(value, dlfn.Expression)
                    bc_object = dlfn.DirichletBC(pressure_space, value,
                                                 self._boundary_markers, bndry_id)
                    self._dirichlet_bcs['pressure'].append(bc_object)

                else:  # pragma: no cover
                    raise RuntimeError()
        # HINT: traction boundary conditions are covered in _setup_problem

    def _setup_function_spaces(self):
        """
        Class method setting up function spaces.
        """
        # create joint function space
        SolverBase._setup_function_spaces(self)

        # create subspaces
        self._Vh = dict()
        for key, index in self._field_association.items():
            space = dlfn.FunctionSpace(self._Wh.mesh(),
                                       self._Wh.sub(index).ufl_element())
            self._Vh[key] = space

        # create joint solution
        self._solutions = [dlfn.Function(self._Wh), dlfn.Function(self._Wh)]

        # create separate solutions
        self._velocities = [dlfn.Function(self._Vh["velocity"]) for _ in range(self._time_stepping.n_levels() + 1)]
        self._intermediate_velocity = dlfn.Function(self._Vh["velocity"])
        self._pressure = dlfn.Function(self._Vh["pressure"])
        self._old_pressure = dlfn.Function(self._Vh["pressure"])

    def _setup_problem(self):
        """Method setting up solvers object of the instationary problem.
        """
        assert hasattr(self, "_mesh")
        assert hasattr(self, "_boundary_markers")
    
        if not all(hasattr(self, attr) for attr in ("_Wh", "_Vh"
                                                    "_joint_solution",
                                                    "_velocities",
                                                    "_pressure", "_old_pressure")):  # pragma: no cover
            self._setup_function_spaces()

        if not all(hasattr(self, attr) for attr in ("_next_step_size",
                                                    "_alpha")):
            self._update_time_stepping_coefficients()

        self._setup_boundary_conditions()

        # setup the diffusion step
        self._setup_diffusion_step()
        # setup the projection step
        self._setup_projection_step()
        # setup the correction step
        self._setup_correction_step()

    def _setup_diffusion_step(self):
        """Method setting up solver object of the diffusion step."""
        assert hasattr(self, "_Vh")
        assert hasattr(self, "_intermediate_velocity")
        assert hasattr(self, "_old_pressure")
        assert hasattr(self, "_velocities")

        # creating test and trial functions
        Vh = self._Vh["velocity"]
        w = dlfn.TestFunction(Vh)
        # volume element
        dV = dlfn.Measure("dx", domain=self._mesh)

        # dimensionless parameters
        assert hasattr(self, "_Re")
        Re = self._Re

        # sought-for velocity
        velocity = self._intermediate_velocity

        # velocities used in acceleration term
        velocities = []
        velocities.append(velocity)
        for i in range(1, self._time_stepping.n_levels() + 1):
            velocities.append(self._velocities[i])
            
        # momentum equation           
        self._F = (
                    self._acceleration_term(velocities, w)
                    + self._convective_term(velocity, w)
                    - self._divergence_term(w, self._old_pressure)
                    + self._viscous_term(velocity, w) / Re
                    ) * dV

        # add boundary tractions
        self._F = self._add_boundary_tractions(self._F, w)

        # add body force term
        if hasattr(self, "_body_force"):
            assert hasattr(self, "_Fr"), "Froude number is not specified."
            self._F -= dlfn.dot(self._body_force, w) / self._Fr**2 * dV

        # linearization using Newton's method
        self._J_newton = dlfn.derivative(self._F, velocity)

        # setup problem with Newton linearization
        self._diffusion_problem = dlfn.NonlinearVariationalProblem(self._F,
                                                                   velocity,
                                                                   self._dirichlet_bcs['velocity'],
                                                                   self._J_newton)
        # setup non-linear solver
        self._diffusion_solver = dlfn.NonlinearVariationalSolver(self._diffusion_problem)
        self._diffusion_solver.parameters["newton_solver"]["absolute_tolerance"] = self._tol
        self._diffusion_solver.parameters["newton_solver"]["maximum_iterations"] = self._maxiter
        self._diffusion_solver.parameters["newton_solver"]["relative_tolerance"] = 1.0e1 * self._tol
        self._diffusion_solver.parameters["newton_solver"]["error_on_nonconvergence"] = True

    def _setup_projection_step(self):
        """Method setting up solver object of the projection step."""
        assert hasattr(self, "_Vh")
        assert hasattr(self, "_intermediate_velocity")
        assert hasattr(self, "_old_pressure")

        # creating test and trial functions
        Vh = self._Vh["pressure"]
        p = dlfn.TrialFunction(Vh)
        q = dlfn.TestFunction(Vh)

        # volume element
        dV = dlfn.Measure("dx", domain=self._mesh)

        # pressure projection equation
        self._pressure_correction_lhs = dlfn.dot(dlfn.grad(p), dlfn.grad(q)) * dV
        self._pressure_correction_rhs = (dlfn.dot(dlfn.grad(self._old_pressure),dlfn.grad(q)) 
                                         + (-self._alpha[0]/self._next_step_size) * dlfn.dot(dlfn.div(self._intermediate_velocity), q)) * dV

        # setup linear problem
        self._projection_problem = dlfn.LinearVariationalProblem(self._pressure_correction_lhs,
                                                                 self._pressure_correction_rhs,
                                                                 self._pressure,
                                                                 self._dirichlet_bcs['pressure'])
        # setup linear solver
        self._projection_solver = dlfn.LinearVariationalSolver(self._projection_problem)

    def _setup_correction_step(self):
        """Method setting up solver object of the correction step."""

        # volume element
        dV = dlfn.Measure("dx", domain=self._mesh)

        # velocity correction step
        # creating test and trial functions
        Vh = self._Vh["velocity"]
        v = dlfn.TrialFunction(Vh)
        w = dlfn.TestFunction(Vh)

        # velocity correction equation
        self._velocity_correction_lhs = dlfn.dot(v, w) * dV
        self._velocity_correction_rhs = (dlfn.dot(self._intermediate_velocity, w) 
                                         - (self._next_step_size / self._alpha[0]) * dlfn.dot(dlfn.grad(self._pressure-self._old_pressure), w)) * dV

        # setup linear problem
        self._velocity_correction_problem = \
            dlfn.LinearVariationalProblem(self._velocity_correction_lhs,
                                          self._velocity_correction_rhs,
                                          self._velocities[0],
                                          self._dirichlet_bcs['velocity'])
        # setup linear solver
        self._velocity_correction_solver = \
            dlfn.LinearVariationalSolver(self._velocity_correction_problem)

        ## pressure correction step
        ## creating test and trial functions
        #Vh = self._Vh["pressure"]
        #p = dlfn.TrialFunction(Vh)
        #q = dlfn.TestFunction(Vh)

        ## pressure correction equation
        #self._pressure_correction_lhs = dlfn.dot(dlfn.grad(self._phi), dlfn.grad(q)) * dV
        #self._pressure_correction_rhs = (-self._alpha[0]/self._next_step_size) * dlfn.dot(dlfn.div(self._intermediate_velocity), q) * dV
        ## setup linear problem
        #self._pressure_correction_problem = \
        #    dlfn.LinearVariationalProblem(self._pressure_correction_lhs,
        #                                  self._pressure_correction_rhs,
        #                                  self._pressure,
        #                                  self._dirichlet_bcs_pressure)
        ## setup linear solver
        #self._pressure_correction_solver = \
        #    dlfn.LinearVariationalSolver(self._pressure_correction_problem)
        
        # if phi is necessary for the solver then you need an additional step for calculating the corrected pressure
        # pressure projection step needs to be reworked with phi and pressure correction step has to be implemented

    def _solve_time_step(self):
        """Solves the nonlinear problem for one time step."""
        dlfn.info("Solving diffusion step...")
        dlfn.info("Starting Newton iteration...")
        self._diffusion_solver.solve()

        dlfn.info("Solving projection step...")
        self._projection_solver.solve()

        dlfn.info("Solving velocity correction step...")
        self._velocity_correction_solver.solve()

        #dlfn.info("Solving pressure correction step...")
        #self._pressure_correction_solver.solve()

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
                
                
    @property
    def solution(self):
        
        velocity, pressure = self._solutions[0].split()

        dlfn.assign(velocity, dlfn.project(self._velocities[0], self._Vh['velocity']))
        dlfn.assign(pressure, dlfn.project(self._pressure, self._Vh['pressure']))
        
        old_velocity, old_pressure = self._solutions[1].split()

        dlfn.assign(old_velocity, dlfn.project(self._velocities[1], self._Vh['velocity']))
        dlfn.assign(old_pressure, dlfn.project(self._old_pressure, self._Vh['pressure']))
        
        
        return self._solutions[0]
