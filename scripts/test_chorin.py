#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from dolfin import grad, div, dot, inner
from grid_generator import hyper_rectangle
from grid_generator import HyperRectangleBoundaryMarkers
dlfn.set_log_level(20)

# parameters
problem_name = "ChannelFlow"
Re = dlfn.Constant(100.0)
gamma = 2.0 * dlfn.pi
n_points = 10
poly_deg = 1  # polynomial degree
probe_solution = False
null_vector = dlfn.Constant((0.0, 0.0))
null = dlfn.Constant(0.0)
one = dlfn.Constant(1.0)


def solve_problem(time_step):
    # time
    current_time = 0.0
    end_time = 1.0
    k = dlfn.Constant(time_step, name="dt")
    # time stepping coefficients
    alpha = [dlfn.Constant(1.0, name="alpha00"),
             dlfn.Constant(-1.0, name="alpha01")]

    def acceleration_term(velocity_solutions, w):
        # input check
        assert isinstance(velocity_solutions, (list, tuple))
        assert len(alpha) == len(velocity_solutions)
        # compute accelerations
        accelerations = []
        for i in range(len(alpha)):
            accelerations.append(alpha[i] * velocity_solutions[i])
        return dot(sum(accelerations), w) / k

    def convective_term(u, v):
        return dot(dot(grad(u), u), v)

    def divergence_term(u, v):
        return div(u) * v

    def viscous_term(u, v):
        return inner(grad(u), grad(v)) / Re

    # xdmf file
    xdmf_file = dlfn.XDMFFile(problem_name + ".xdmf")
    xdmf_file.parameters["flush_output"] = True
    xdmf_file.parameters["functions_share_mesh"] = True
    xdmf_file.parameters["rewrite_function_mesh"] = False
    # mesh
    mesh, boundary_markers = hyper_rectangle((0.0, 0.0), (10.0, 1.0),
                                             (10 * n_points, n_points))
    dlfn.File(problem_name + "BoundaryMarker" + ".pvd") << boundary_markers
    # setup function spaces
    cell = mesh.ufl_cell()
    elemV = dlfn.VectorElement("CG", cell, poly_deg + 1)
    elemP = dlfn.FiniteElement("CG", cell, poly_deg)
    mixedElement = dlfn.MixedElement([elemV, elemP])
    Wh = dlfn.FunctionSpace(mesh, mixedElement)
    WhSub = dict()
    WhSub["velocity"] = Wh.sub(0).collapse()
    WhSub["pressure"] = Wh.sub(1).collapse()
    print("Number of cells {0}, number of DoFs: {1}".format(mesh.num_cells(), Wh.dim()))
    # setup initial conditions
    initial_conditions = dict()
    initial_conditions["velocity"] = null_vector
    # boundary conditions
    velocity_bcs = []
    velocity_bcs.append(dlfn.DirichletBC(WhSub["velocity"], null_vector,
                                         boundary_markers,
                                         HyperRectangleBoundaryMarkers.top.value))
    velocity_bcs.append(dlfn.DirichletBC(WhSub["velocity"], null_vector,
                                         boundary_markers,
                                         HyperRectangleBoundaryMarkers.bottom.value))
    pressure_bcs = []
    inlet_pressure = dlfn.Expression("sin(M_PI * time)", time=0.0, degree=0)
    pressure_bcs.append(dlfn.DirichletBC(WhSub["pressure"], inlet_pressure,
                                         boundary_markers,
                                         HyperRectangleBoundaryMarkers.left.value))
    pressure_bcs.append(dlfn.DirichletBC(WhSub["pressure"], null,
                                         boundary_markers,
                                         HyperRectangleBoundaryMarkers.right.value))
    # creating solutions
    velocities = []
    for i in range(2):
        name = i * "old" + (i > 0) * "_" + "velocity"
        velocities.append(dlfn.Function(WhSub["velocity"], name=name))
    intermediate_velocity = dlfn.Function(WhSub["velocity"], name="intermediate_velocity")
    pressure = dlfn.Function(WhSub["pressure"], name="pressure")
    # volume element
    dV = dlfn.Measure("dx", domain=mesh)
    # diffusion step
    w = dlfn.TestFunction(WhSub["velocity"])
    F_diffusion = (acceleration_term([intermediate_velocity, velocities[1]], w)
                   + convective_term(intermediate_velocity, w)
                   + viscous_term(intermediate_velocity, w)
                   ) * dV
    J_newton = dlfn.derivative(F_diffusion, intermediate_velocity)
    diffusion_problem = dlfn.NonlinearVariationalProblem(F_diffusion,
                                                         intermediate_velocity,
                                                         velocity_bcs,
                                                         J_newton)
    diffusion_solver = dlfn.NonlinearVariationalSolver(diffusion_problem)
    diffusion_solver.parameters["newton_solver"]["error_on_nonconvergence"] = True
    # projection step
    phi = dlfn.TrialFunction(WhSub["pressure"])
    q = dlfn.TestFunction(WhSub["pressure"])
    projection_lhs = dot(grad(phi), grad(q)) * dV
    projection_rhs = - alpha[0] / k * div(intermediate_velocity) * q * dV
    projection_problem = dlfn.LinearVariationalProblem(projection_lhs,
                                                       projection_rhs,
                                                       pressure,
                                                       pressure_bcs)
    projection_solver = dlfn.LinearVariationalSolver(projection_problem)
    # correction step
    v = dlfn.TrialFunction(WhSub["velocity"])
    correction_lhs = dot(v, w) * dV
    correction_rhs = (dot(intermediate_velocity, w)
                      - (k / alpha[0]) * dot(grad(pressure), w)) * dV
    correction_problem = dlfn.LinearVariationalProblem(correction_lhs,
                                                       correction_rhs,
                                                       velocities[0],
                                                       velocity_bcs)
    correction_solver = dlfn.LinearVariationalSolver(correction_problem)
    # assign initial condition
    dlfn.project(initial_conditions["velocity"], WhSub["velocity"],
                 function=velocities[0])
    velocities[1].assign(velocities[0])
    # output
    for solution in (*velocities, intermediate_velocity, pressure):
        xdmf_file.write(solution, current_time)
    # time loop
    cnt = 0
    while current_time < end_time:
        next_time = current_time + time_step
        if (next_time + time_step) - end_time > 1e-9:
            time_step = end_time - current_time
            k.assign(time_step)
        # solve
        print(f"step number {cnt:8d}, current time {current_time:10.2e}, next step size {time_step:10.2e}")
        inlet_pressure.time = current_time
        diffusion_solver.solve()
        projection_solver.solve()
        correction_solver.solve()
        # advance in time
        for i in range(2, 1, -1):
            velocities[i-1].assign(velocities[i-2])
        current_time += time_step
        cnt += 1
        # output
        for solution in (*velocities, intermediate_velocity, pressure):
            xdmf_file.write(solution, current_time)
    print(f"step number {cnt:8d}, current time {current_time:10.2e}")


if __name__ == "__main__":
    time_step = 1e-2
    solve_problem(time_step)
