#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from navier_stokes_problem import StationaryNavierStokesProblem, VelocityBCType
from grid_generator import hyper_cube, HyperCubeBoundaryMarkers

class CavityProblem(StationaryNavierStokesProblem):
    def __init__(self, n_points, main_dir = None):
        super().__init__(main_dir)
        
        self._n_points = n_points
        self._problem_name  = "Cavity"
    
    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)
        
    def set_boundary_conditions(self):
        # velocity boundary conditions
        velocity_bcs = (
                (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.left.value, None),
                (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.right.value, None),
                (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.bottom.value, None),
                (VelocityBCType.constant, HyperCubeBoundaryMarkers.right.value, (1.0, 0.0)))
        self._bcs = {"velocity": velocity_bcs}
        
cf = CavityProblem(10)
cf.solve_problem()