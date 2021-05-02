#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dolfin as dlfn
from navier_stokes_problem import StationaryNavierStokesProblem, VelocityBCType

from grid_generator import open_hyper_cube, HyperCubeBoundaryMarkers

class GravityDrivenFlowProblem(StationaryNavierStokesProblem):
    def __init__(self, n_points, main_dir = None):
        super().__init__(main_dir)
        
        self._n_points = n_points
        self._problem_name  = "OpenCube"
        
        self.set_parameters(Re=250.0,  Fr=10.0)
    
    def setup_mesh(self):
        # create mesh
        openings = (("bottom", (0.2, 0.0), 0.1),
                    ("left", (0.0, 0.5), 0.1),
                    ("right", (1.0, 0.7), 0.1),
                    ("bottom", (0.7, 0.0), 0.05),
                    ("top", (0.5, 1.0), 0.8))
        self._mesh, self._boundary_markers = open_hyper_cube(2, self._n_points, openings)
        self.write_boundary_markers()
        
    def set_boundary_conditions(self):
        # velocity boundary conditions
        velocity_bcs = (
                (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.left.value, None),
                (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.right.value, None),
                (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.bottom.value, None))
        self._bcs = {"velocity": velocity_bcs}
    
    def set_body_force(self):
        self._body_force = dlfn.Constant((0.0, -1.0))
        
gravity_flow = GravityDrivenFlowProblem(75)
gravity_flow.solve_problem()