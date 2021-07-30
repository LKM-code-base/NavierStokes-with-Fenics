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
    end_time = 3.0
    k = dlfn.Constant(time_step, name="dt")
    # time stepping coefficients
    alpha = [dlfn.Constant(1.0, name="alpha00"),
             dlfn.Constant(-1.0, name="alpha01"),
             dlfn.Constant(0.0, name="alpha02")]
    eta = [dlfn.Constant(2.0, name="eta00"), dlfn.Constant(-1.0, name="eta01")]

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
    initial_conditions["pressure"] = null
    # boundary conditions
    velocity_bcs = []
    inlet_velocity = dlfn.Expression(("6.0*x[1]/h*(1.0-x[1]/h)", "0.0"), h=1.0, degree=2)
    velocity_bcs.append(dlfn.DirichletBC(WhSub["velocity"], inlet_velocity,
                                         boundary_markers,
                                         HyperRectangleBoundaryMarkers.left.value))
    velocity_bcs.append(dlfn.DirichletBC(WhSub["velocity"], null_vector,
                                         boundary_markers,
                                         HyperRectangleBoundaryMarkers.top.value))
    velocity_bcs.append(dlfn.DirichletBC(WhSub["velocity"], null_vector,
                                         boundary_markers,
                                         HyperRectangleBoundaryMarkers.bottom.value))
    phi_bcs = []
    phi_bcs.append(dlfn.DirichletBC(WhSub["pressure"], null,
                                    boundary_markers,
                                    HyperRectangleBoundaryMarkers.right.value))
    # creating solutions
    velocities = []
    for i in range(3):
        name = i * "old" + (i > 0) * "_" + "velocity"
        velocities.append(dlfn.Function(WhSub["velocity"], name=name))
    intermediate_velocity = dlfn.Function(WhSub["velocity"], name="intermediate_velocity")
    phi = dlfn.Function(WhSub["pressure"], name="phi")
    pressure = dlfn.Function(WhSub["pressure"], name="pressure")
    old_pressure = dlfn.Function(WhSub["pressure"], name="old_pressure")
    # volume element
    dV = dlfn.Measure("dx", domain=mesh)
    # diffusion step
    w = dlfn.TestFunction(WhSub["velocity"])
    F_diffusion = (acceleration_term([intermediate_velocity, velocities[1], velocities[2]], w)
                   + convective_term(intermediate_velocity, w)
                   - divergence_term(w, eta[0] * pressure + eta[1] * old_pressure)
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
    p = dlfn.TrialFunction(WhSub["pressure"])
    q = dlfn.TestFunction(WhSub["pressure"])
    projection_lhs = dot(grad(p), grad(q)) * dV
    projection_rhs = - alpha[0] / k * div(intermediate_velocity) * q * dV
    projection_problem = dlfn.LinearVariationalProblem(projection_lhs,
                                                       projection_rhs,
                                                       phi,
                                                       phi_bcs)
    projection_solver = dlfn.LinearVariationalSolver(projection_problem)
    # correction step
    pressure_correction_lhs = dot(p, q) * dV
    pressure_correction_rhs = (old_pressure + phi) * q * dV
    pressure_correction_problem = dlfn.LinearVariationalProblem(pressure_correction_lhs,
                                                                pressure_correction_rhs,
                                                                pressure)
    pressure_correction_solver = dlfn.LinearVariationalSolver(pressure_correction_problem)
    # correction step
    v = dlfn.TrialFunction(WhSub["velocity"])
    velocity_correction_lhs = dot(v, w) * dV
    velocity_correction_rhs = (dot(intermediate_velocity, w) - (k / alpha[0]) * dot(grad(phi), w)) * dV
    velocity_correction_problem = dlfn.LinearVariationalProblem(velocity_correction_lhs,
                                                                velocity_correction_rhs,
                                                                velocities[0])
    velocity_correction_solver = dlfn.LinearVariationalSolver(velocity_correction_problem)
    # assign initial condition
    dlfn.project(initial_conditions["velocity"], WhSub["velocity"],
                 function=velocities[0])
    velocities[1].assign(velocities[0])
    dlfn.project(initial_conditions["pressure"], WhSub["pressure"],
                 function=pressure)
    old_pressure.assign(pressure)
    # output
    for solution in (*velocities, intermediate_velocity,
                     pressure, old_pressure, phi):
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
        diffusion_solver.solve()
        projection_solver.solve()
        velocity_correction_solver.solve()
        pressure_correction_solver.solve()
        # advance in time
        for i in range(3, 1, -1):
            velocities[i-1].assign(velocities[i-2])
        old_pressure.assign(pressure)
        current_time += time_step
        cnt += 1
        # output
        for solution in (*velocities, intermediate_velocity,
                         pressure, old_pressure, phi):
            xdmf_file.write(solution, current_time)
        # update alpha
        if cnt == 1:
            alpha[0].assign(3./2.)
            alpha[1].assign(-2.)
            alpha[2].assign(1./2.)
    print(f"step number {cnt:8d}, current time {current_time:10.2e}")


if __name__ == "__main__":
    time_step = 1e-2
    solve_problem(time_step)
