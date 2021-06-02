#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dolfin as dlfn

import numpy as np

from navier_stokes_problem import StationaryNavierStokesProblem

from navier_stokes_solver import VelocityBCType

from grid_generator import unit_square, UnitSquareBoundaryMarkers

dlfn.set_log_level(40)


class CavityProblem(StationaryNavierStokesProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir)

        self._n_points = n_points
        self._problem_name = "Lid-Driven Cavity Flow"

        self.set_parameters(Re=1000.0)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = unit_square(self._n_points)
        dlfn.plot(self._mesh)
        

    def set_boundary_conditions(self):
        # velocity boundary conditions
        no_slip = VelocityBCType.no_slip
        constant = VelocityBCType.constant
        BoundaryMarkers = UnitSquareBoundaryMarkers
        velocity_bcs = (
                (no_slip, BoundaryMarkers.left.value, None),
                (no_slip, BoundaryMarkers.right.value, None),
                (no_slip, BoundaryMarkers.bottom.value, None),
                (constant, BoundaryMarkers.top.value, (1.0, 0.0)))
        self._bcs = {"velocity": velocity_bcs}


if __name__ == "__main__":

##########################
##########################
    #error = np.zeros(5)
    #n_cells = np.zeros(5)
    
    #n_points = [10,50,100,150,200]
    
    #for i in range(np.size(n_points)):
        
        #u_analytic = dlfn.Expression(("8*(pow(x[0],4)-2*pow(x[0],3)+pow(x[0],2))*(4*pow(x[1],3)-2*x[1])",
        #                          "-8*(4*pow(x[0],3)-6*pow(x[0],2)+2*x[0])*(pow(x[1],4)-pow(x[1],2))"), degree=4)
    
        #cavity_flow = CavityProblem(n_points[i])
        #cavity_flow.solve_problem()
    
        #u_numeric = cavity_flow._get_velocity()
        #n_cells[i] = cavity_flow._mesh.num_cells()
    
        #Vh = dlfn.VectorFunctionSpace(cavity_flow._mesh, 'CG', 2)
        #u_analytic = dlfn.project(u_analytic, Vh)
    
        #error[i] = dlfn.errornorm(u_numeric, u_analytic,'L2')

##########################
##########################
    cavity_flow = CavityProblem(20)
    cavity_flow.solve_problem()
    u_numeric = cavity_flow._get_velocity()
    velocity_x = np.zeros((17,2))
    velocity_y = np.zeros((17,2))
    x_coord = [ 1, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000,                     0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, .0625, .0000]
    y_coord = [ 1, 0.9766, 0.9688, 0.9609, 0.953, 0.8516, 0.7344, 0.6172, 
             0.5000, 0.453, 0.2813, 0.1719, 0.1016, 0.0703, .0625, .0547, .0000]

    for i in range(np.size(x_coord)):
        velocity_x[i,:] = u_numeric(x_coord[i],0.5)
        velocity_y[i,:] = u_numeric(0.5, y_coord[i])