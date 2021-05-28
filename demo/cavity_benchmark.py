# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:21:49 2021

@author: david
"""
import dolfin as dlfn

import numpy as np 

import pandas as pd

import matplotlib 
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure

from navier_stokes_problem import StationaryNavierStokesProblem

from navier_stokes_solver import VelocityBCType

from grid_generator import hyper_cube, HyperCubeBoundaryMarkers

dlfn.set_log_level(40)


class CavityProblem(StationaryNavierStokesProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir)

        self._n_points = n_points
        self._problem_name = "Cavity"

        self.set_parameters(Re=100.0)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_boundary_conditions(self):
        # velocity boundary conditions
        no_slip = VelocityBCType.no_slip
        constant = VelocityBCType.constant
        BoundaryMarkers = HyperCubeBoundaryMarkers
        velocity_bcs = (
                (no_slip, BoundaryMarkers.left.value, None),
                (no_slip, BoundaryMarkers.right.value, None),
                (no_slip, BoundaryMarkers.bottom.value, None),
                (constant, BoundaryMarkers.top.value, (1.0, 0.0)))
        self._bcs = {"velocity": velocity_bcs}
        
    def postprocess_solution(self): 
        #pressure = self._get_pressure()
        velocity = self._get_velocity()
        file = r'../data/giagia_tab1.csv'
        data = pd.read_csv(file, header=0, sep=';')
        yref = np.array(data['y'])
        #yref = np.array([ 1, 0.9766, 0.9688, 0.9609, 0.953, 0.8516, 0.7344, 0.6172, 
                         #0.5000, 0.453, 0.2813, 0.1719, 0.1016, 0.0703, .0625, .0547, .0000])
        uVelref = np.array(data['100'])
        uVel = np.zeros(len(yref))
        i = 0
        for y in yref: 
            p = dlfn.Point(0.5, y)
            #print(str(velocity(p)[0])+ '\t' + str(uVelref[i]) + '\t' + str(velocity(p)[0]-uVelref[i]))
            uVel[i]=velocity(p)[0]
            i = i+1
        diff = np.sqrt((uVel- uVelref)**2)
        print(diff)
        font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}
        matplotlib.rc('font', **font)
        matplotlib. pyplot.close('all')
        plt.figure(0, figsize=(40,20))
        #plt.xticks(np.arange(0,tend,0.05))
        plt.plot(yref, diff, linewidth=3.0)
        plt.savefig('Error_uVeloverY.eps', format='eps')
        #plt.grid(b=0.5)
        # # compute potential energy
        # strings = tuple("x[{0:d}]".format(i) for i in range(self._space_dim))
        # position_vector = dlfn.Expression(strings, degree=1)
        # potential_energy = dlfn.dot(self._body_force, position_vector)
        # # compute Bernoulli potential
        # Phi = dlfn.Constant(0.5) * dlfn.dot(velocity, velocity)
        # Phi += pressure + potential_energy / dlfn.Constant(self._Fr)**2
        # # project on Bernoulli potential on the mesh
        # Vh = dlfn.FunctionSpace(self._mesh, "CG", 1)
        # phi = dlfn.project(Phi, Vh)
        # phi.rename("Bernoulli potential", "")
        # # add Bernoulli potential to the field output
        # self._add_to_field_output(phi)
        # # add pressure gradient to the field output
        # self._add_to_field_output(self._compute_pressure_gradient())
        # # add vorticity to the field output
        # self._add_to_field_output(self._compute_vorticity())
        # # add stream potential to the field output
        # self._add_to_field_output(self._compute_stream_potential())
        
        # # compute mass flux over the entire boundary
        # normal = dlfn.FacetNormal(self._mesh)
        # dA = dlfn.Measure("ds", domain=self._mesh, subdomain_data=self._boundary_markers)
        # mass_flux = dlfn.assemble(dlfn.dot(normal, velocity) * dA)
        # dlfn.info("Value of the total mass flux: {0:6.2e}".format(mass_flux))


if __name__ == "__main__":
    cavity_flow = CavityProblem(50)
    cavity_flow.solve_problem()
