#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
import math
import numpy as np
from dolfin import grad, div, dot, inner

dlfn.parameters["form_compiler"]["representation"] = "uflacs"
dlfn.parameters["form_compiler"]["cpp_optimize"] = True
dlfn.parameters["form_compiler"]["optimize"] = True
dlfn.set_log_level(30)

# parameters
Re = 100.0
gamma = 2.0 * dlfn.pi
n_points = 64
poly_deg = 1 # polynomial degree
probe_solution = False

# periodic subdomain
class PeriodicDomain(dlfn.SubDomain):
    def inside(self, x, on_boundary):
        """Return True if `x` is located on the master edge and False
        else.
        """
        inside = False
        if (dlfn.near(x[0], 0.0) and on_boundary):
            inside = True
        elif (dlfn.near(x[1], 0.0) and on_boundary):
            inside = True
        return inside

    def map(self, x_slave, x_master):
        """Map the coordinates of the support points (nodes) of the degrees
        of freedom of the slave to the coordinates of the corresponding
        master edge.
        """
        # points at the right edge
        if dlfn.near(x_slave[0], 1.0):
            x_master[0] = x_slave[0] - 1.0
            x_master[1] = x_slave[1]
        # points at the top edge
        elif dlfn.near(x_slave[1], 1.0):
            x_master[0] = x_slave[0]
            x_master[1] = x_slave[1] - 1.0
        else:
            # map other outside of the domain
            x_master[0] = -10.0
            x_master[1] = -10.0

def solve_problem(time_step):
    # time
    current_time = 0.0
    end_time = 1
    k = dlfn.Constant(time_step)
    
    # time stepping coefficients
    alpha = [dlfn.Constant(1.0), dlfn.Constant(-1.0), dlfn.Constant(0.0)]
    
    # mesh
    corner_points = (dlfn.Point(0., 0.), dlfn.Point(1., 1.))
    mesh = dlfn.RectangleMesh(*corner_points, n_points, n_points)
    
    # periodic subdomain
    periodic_bcs = PeriodicDomain()
    
    # setup function space
    cell = mesh.ufl_cell()
    elemV = dlfn.VectorElement("CG", cell, poly_deg + 1)
    elemP = dlfn.FiniteElement("CG", cell, poly_deg)
    mixedElement = dlfn.MixedElement([elemV , elemP])
    Wh = dlfn.FunctionSpace(mesh, mixedElement, constrained_domain=periodic_bcs)
    
    print("Number of cells {0}, number of DoFs: {1}".format(mesh.num_cells(), Wh.dim()))
    # creating test and trial functions
    (w, q) = dlfn.TestFunctions(Wh)
    
    # creating solutions
    solution = dlfn.Function(Wh)
    velocity, pressure = dlfn.split(solution)
    old_solution = dlfn.Function(Wh)
    old_velocity, _ = dlfn.split(old_solution)
    old_old_solution = dlfn.Function(Wh)
    old_old_velocity, _ = dlfn.split(old_old_solution)
    
    # surface and volume element
    dV = dlfn.Measure("dx", domain = mesh)
    
    # weak forms
    F_mass = - div(velocity) * q * dV
    F_momentum = ( dot(alpha[0] * velocity 
                       + alpha[1] * old_velocity
                       + alpha[2] * old_old_velocity, w) / k
                    + dot(dot(grad(velocity), velocity), w)
                    - div(w) * pressure
                    + dlfn.Constant(1.0 / Re) * inner(grad(velocity), grad(w))
                 ) * dV
    F = F_mass + F_momentum
    
    # newton solver object
    J_newton = dlfn.derivative(F, solution)
    problem = dlfn.NonlinearVariationalProblem(F, solution, J = J_newton)
    solver = dlfn.NonlinearVariationalSolver(problem)
    
    # split solution
    velocity_solution, pressure_solution = solution.split()
    
    # create subspaces
    sub_space_dict = dict()
    sub_space_dict["velocity"] = dlfn.FunctionSpace(mesh, elemV, constrained_domain=periodic_bcs)
    sub_space_dict["pressure"] = dlfn.FunctionSpace(mesh, elemP, constrained_domain=periodic_bcs)
    sub_spaces = [sub_space_dict["velocity"], sub_space_dict["pressure"]]

    # extract solutions
    aux_solution_dict = dict()
    for key in sub_space_dict:
        aux_solution_dict[key] = dlfn.Function(sub_space_dict[key])
    aux_solutions = [aux_solution_dict["velocity"], aux_solution_dict["pressure"]]
    function_extractor = dlfn.FunctionAssigner(sub_spaces, Wh)
    
    # compose solution
    function_composer = dlfn.FunctionAssigner(Wh, sub_spaces)
    
    # initial condition
    initial_conditions = dict()
    initial_conditions["velocity"] = \
            dlfn.Expression(("cos(gamma * x[0]) * sin(gamma * x[1])",
                             "-sin(gamma * x[0]) * cos(gamma * x[1])"),
                            gamma=gamma, degree=5)
    initial_conditions["pressure"] = \
            dlfn.Expression("-1.0/4.0 * (cos(2.0 * gamma * x[0]) + cos(2.0 * gamma * x[1]))",
                            gamma=gamma, degree=5)
    aux_solutions[0].assign(dlfn.project(initial_conditions["velocity"], sub_space_dict["velocity"]))
    aux_solutions[1].assign(dlfn.project(initial_conditions["pressure"], sub_space_dict["pressure"]))
    
    function_composer.assign(old_solution, aux_solutions)
    function_composer.assign(solution, aux_solutions)
    
    # time loop
    cnt = 0
    if probe_solution:
        n_steps = math.ceil(end_time / time_step) + 1
        probes = {"pressure": np.zeros((n_steps, )), "velocity": np.zeros((n_steps, 2))}
        probes["velocity"][cnt] = velocity_solution(0.1, 0.1)
        probes["pressure"][cnt] = pressure_solution(0.1, 0.1)
    while current_time < end_time:
        next_time = current_time + time_step
        if (next_time + time_step) - end_time > 1e-9:
            time_step = end_time - current_time
            k.assign(time_step)
        # solve
        print(f"step number {cnt:8d}, current time {current_time:10.2e}, next step size {time_step:10.2e}")
        solver.solve()
        # advance in time
        old_old_solution.assign(old_solution)
        old_solution.assign(solution)
        # correct pressure 
        function_extractor.assign(aux_solutions, solution)
        pressure_mean_value = dlfn.assemble(pressure_solution * dV) / dlfn.assemble(dlfn.Constant(1.0) * dV)
        modified_pressure =  pressure_solution - dlfn.Constant(pressure_mean_value)
        corrected_pressure = dlfn.project(modified_pressure, sub_space_dict["pressure"])
        aux_solution_dict["pressure"].assign(corrected_pressure)
        function_composer.assign(solution, aux_solutions)
        # print
        if probe_solution:
            print("velocity(x = 0.1, y = 0.1) = ", velocity_solution(0.1, 0.1))
            print("pressure(x = 0.1, y = 0.1) = ", pressure_solution(0.1, 0.1))
            probes["velocity"][cnt+1] = velocity_solution(0.1, 0.1)
            probes["pressure"][cnt+1] = pressure_solution(0.1, 0.1)
        # advance time
        current_time += time_step
        cnt += 1
        # update alpha
        if cnt == 0:
            alpha[0].assign(3./2.)
            alpha[1].assign(-2.)
            alpha[2].assign(1./2.)
    print(f"step number {cnt:8d}, current time {current_time:10.2e}")
    assert ((next_time - end_time) / end_time) < 1e-9
    # save probes
    if probe_solution:
        for key in probes:
            np.save(f"probe_{key}.npy", probes[key])
    # compute error
    exact_solution = dict()
    exact_solution["velocity"] = \
        dlfn.Expression(("exp(-2.0 * gamma * gamma / Re * t) * cos(gamma * x[0]) * sin(gamma * x[1])",
                         "-exp(-2.0 * gamma * gamma / Re * t) * sin(gamma * x[0]) * cos(gamma * x[1])"),
                        gamma=gamma, Re=Re, t=current_time, degree=5)
    exact_solution["pressure"] = \
    dlfn.Expression("-1.0/4.0 * exp(-4.0 * gamma * gamma / Re * t) * (cos(2.0 * gamma * x[0]) + cos(2.0 * gamma * x[1]))",
                    gamma=gamma, Re=Re, t=current_time, degree=5)
    return (dlfn.errornorm(exact_solution["velocity"], velocity_solution),
            dlfn.errornorm(exact_solution["pressure"], pressure_solution))

# convergence test
if __name__ == "__main__":
    n_levels = 4
    errors = {"pressure": np.zeros((n_levels, )), "velocity": np.zeros((n_levels, ))}
    time_step_sizes = np.zeros((n_levels, ))
    initial_time_step = 1e-1
    time_step_reduction_factor = 0.5
    for i in range(n_levels):
        time_step = initial_time_step * time_step_reduction_factor**i
        time_step_sizes[i] = time_step
        velocity_error, pressure_error = solve_problem(time_step)
        errors["velocity"][i] = velocity_error
        errors["pressure"][i] = pressure_error
    print(errors)
