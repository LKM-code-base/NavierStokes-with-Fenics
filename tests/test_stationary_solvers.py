#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from auxiliary_classes import EquationCoefficientHandler
from ns_problem import StationaryProblem
from ns_solver_base import VelocityBCType
from ns_solver_base import PressureBCType
from ns_solver_base import TractionBCType
from grid_generator import blasius_plate
from grid_generator import hyper_cube
from grid_generator import hyper_rectangle
from grid_generator import open_hyper_cube
from grid_generator import HyperCubeBoundaryMarkers
from grid_generator import HyperRectangleBoundaryMarkers

dlfn.set_log_level(20)


class PeriodicDomain(dlfn.SubDomain):
    def inside(self, x, on_boundary):
        """Return True if `x` is located on the master edge and False
        else.
        """
        return (dlfn.near(x[0], 0.0) and on_boundary)

    def map(self, x_slave, x_master):
        """Map the coordinates of the support points (nodes) of the degrees
        of freedom of the slave to the coordinates of the corresponding
        master edge.
        """
        x_master[0] = x_slave[0] - 1.0
        x_master[1] = x_slave[1]


class CavityProblem(StationaryProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir)
        self._n_points = n_points
        self._problem_name = "Cavity"

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_boundary_conditions(self):
        # velocity boundary conditions
        self._bcs = ((VelocityBCType.no_slip, HyperCubeBoundaryMarkers.left.value, None),
                     (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.right.value, None),
                     (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.bottom.value, None),
                     (VelocityBCType.constant, HyperCubeBoundaryMarkers.top.value, (1.0, 0.0)))

    def set_equation_coefficients(self):
        self._coefficient_handler = EquationCoefficientHandler(Re=10.0)


class GravityDrivenFlowProblem(StationaryProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir)

        self._n_points = n_points
        self._problem_name = "OpenCube"

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
        self._bcs = ((VelocityBCType.no_slip, HyperCubeBoundaryMarkers.left.value, None),
                     (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.right.value, None),
                     (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.bottom.value, None),
                     (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.top.value, None))

    def set_equation_coefficients(self):
        self._coefficient_handler = EquationCoefficientHandler(Re=200.0, Fr=10.0)

    def postprocess_solution(self):
        pressure = self._get_pressure()
        velocity = self._get_velocity()
        # compute potential energy
        strings = tuple("x[{0:d}]".format(i) for i in range(self._space_dim))
        position_vector = dlfn.Expression(strings, degree=1)
        potential_energy = dlfn.dot(self._body_force, position_vector)
        # compute Bernoulli potential
        Phi = dlfn.Constant(0.5) * dlfn.dot(velocity, velocity)
        Phi += pressure + potential_energy / dlfn.Constant(self._coefficient_handler.Fr)**2
        # project on Bernoulli potential on the mesh
        Vh = dlfn.FunctionSpace(self._mesh, "CG", 1)
        phi = dlfn.project(Phi, Vh)
        phi.rename("Bernoulli potential", "")
        # add Bernoulli potential to the field output
        self._add_to_field_output(phi)
        # add pressure gradient to the field output
        self._add_to_field_output(self._compute_pressure_gradient())
        # add vorticity to the field output
        self._add_to_field_output(self._compute_vorticity())
        # add stream potential to the field output
        self._add_to_field_output(self._compute_stream_potential())

        # compute mass flux over the entire boundary
        normal = dlfn.FacetNormal(self._mesh)
        dA = dlfn.Measure("ds", domain=self._mesh, subdomain_data=self._boundary_markers)
        mass_flux = dlfn.assemble(dlfn.dot(normal, velocity) * dA)
        dlfn.info("Value of the total mass flux: {0:6.2e}".format(mass_flux))

    def set_body_force(self):
        self._body_force = dlfn.Constant((0.0, -1.0))


class CouetteProblem(StationaryProblem):
    """Couette flow problem with periodic boundary conditions in x-direction."""
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir)

        self._n_points = n_points
        self._problem_name = "Couette"

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_boundary_conditions(self):
        # velocity boundary conditions
        self._bcs = ((VelocityBCType.no_slip, HyperCubeBoundaryMarkers.bottom.value, None),
                     (TractionBCType.constant_component, HyperCubeBoundaryMarkers.top.value, 0, 1.0),
                     (VelocityBCType.no_normal_flux, HyperCubeBoundaryMarkers.top.value, None))

    def set_equation_coefficients(self):
        self._coefficient_handler = EquationCoefficientHandler(Re=1.0)

    def set_periodic_boundary_conditions(self):
        """Set periodic boundary condition in x-direction."""
        self._periodic_bcs = PeriodicDomain()
        self._periodic_boundary_ids = (HyperCubeBoundaryMarkers.left.value,
                                       HyperCubeBoundaryMarkers.right.value)


class ChannelFlowProblem(StationaryProblem):
    def __init__(self, n_points, main_dir=None, bc_type="inlet",
                 form_convective_term="standard"):
        super().__init__(main_dir, form_convective_term=form_convective_term)

        assert isinstance(n_points, int)
        assert n_points > 0
        self._n_points = n_points

        assert isinstance(bc_type, str)
        assert bc_type in ("inlet", "pressure_gradient", "inlet_pressure", "inlet_component")
        self._bc_type = bc_type

        if self._bc_type == "inlet":
            self._problem_name = "ChannelFlowInlet"
        elif self._bc_type == "pressure_gradient":
            self._problem_name = "ChannelFlowPressureGradient"
        elif self._bc_type == "inlet_pressure":
            self._problem_name = "ChannelFlowInletPressure"
        elif self._bc_type == "inlet_component":
            self._problem_name = "ChannelFlowInletComponent"

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_rectangle((0.0, 0.0), (10.0, 1.0),
                                                             (10 * self._n_points, self._n_points))

    def set_boundary_conditions(self):
        # functions
        inlet_profile_str = "6.0*x[1]*(1.0-x[1])"
        inlet_velocity = dlfn.Expression((inlet_profile_str, "0.0"), degree=2)
        inlet_velocity_component = dlfn.Expression(inlet_profile_str, degree=2)
        outlet_pressure = dlfn.Expression("0.0", degree=0)
        # boundary markers
        Markers = HyperRectangleBoundaryMarkers

        # boundary conditions
        self._bcs = []

        if self._bc_type == "inlet":
            # inlet velocity profile
            self._bcs.append((VelocityBCType.function, Markers.left.value, inlet_velocity))
            # no-slip on the walls
            self._bcs.append((VelocityBCType.no_slip, Markers.bottom.value, None))
            self._bcs.append((VelocityBCType.no_slip, Markers.top.value, None))
        elif self._bc_type == "pressure_gradient":
            # pressure at the inlet (as a constant)
            self._bcs.append((PressureBCType.constant, Markers.left.value, 1.0))
            # pressure at the outlet (as a constant)
            self._bcs.append((PressureBCType.constant, Markers.right.value, -1.0))
            # no-slip on the walls
            self._bcs.append((VelocityBCType.no_slip, Markers.bottom.value, None))
            self._bcs.append((VelocityBCType.no_slip, Markers.top.value, None))
        elif self._bc_type == "inlet_pressure":
            # inlet velocity profile
            self._bcs.append((VelocityBCType.function, Markers.left.value, inlet_velocity))
            # no-slip on the walls
            self._bcs.append((VelocityBCType.no_slip, Markers.bottom.value, None))
            self._bcs.append((VelocityBCType.no_slip, Markers.top.value, None))
            # pressure at the outlet (as a function)
            self._bcs.append((PressureBCType.function, Markers.right.value, outlet_pressure))
        elif self._bc_type == "inlet_component":
            # inlet velocity profile (component)
            self._bcs.append((VelocityBCType.function_component, Markers.left.value, 0, inlet_velocity_component))
            # no-slip on the walls
            self._bcs.append((VelocityBCType.no_slip, Markers.bottom.value, None))
            self._bcs.append((VelocityBCType.no_slip, Markers.top.value, None))
            # pressure at the outlet (as a constant)
            self._bcs.append((PressureBCType.constant, Markers.right.value, 0.0))

    def set_equation_coefficients(self):
        self._coefficient_handler = EquationCoefficientHandler(Re=1.0)

    def postprocess_solution(self):
        # add pressure gradient to the field output
        self._add_to_field_output(self._compute_pressure_gradient())
        # add vorticity to the field output
        self._add_to_field_output(self._compute_vorticity())


class BlasiusFlowProblem(StationaryProblem):
    def __init__(self, main_dir=None):
        super().__init__(main_dir)
        self._problem_name = "BlasiusFlow"

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers, self._boundary_marker_map, _, _ = blasius_plate()

    def set_boundary_conditions(self):
        # velocity boundary conditions
        inlet_velocity = dlfn.Expression(("1.0", "0.0"),
                                         h=0.5, y0=0.5, degree=2)
        self._bcs = ((VelocityBCType.function, self._boundary_marker_map["inlet"], inlet_velocity),
                     (VelocityBCType.no_normal_flux, self._boundary_marker_map["bottom"], None),
                     (VelocityBCType.no_normal_flux, self._boundary_marker_map["top"], None))

    def set_equation_coefficients(self):
        self._coefficient_handler = EquationCoefficientHandler(Re=200.0)

    def set_internal_constraints(self):
        self._internal_constraints = ((VelocityBCType.no_slip, self._boundary_marker_map["plate"], None), )

    def postprocess_solution(self):
        # add pressure gradient to the field output
        self._add_to_field_output(self._compute_pressure_gradient())
        # add vorticity to the field output
        self._add_to_field_output(self._compute_vorticity())


def test_blasius_flow():
    blasius_flow = BlasiusFlowProblem()
    blasius_flow.solve_problem()


def test_cavity():
    cavity_flow = CavityProblem(10)
    cavity_flow.solve_problem()


def test_channel_flow():
    for bc_type in ("inlet", "pressure_gradient", "inlet_pressure", "inlet_component"):
        channel_flow = ChannelFlowProblem(10, bc_type=bc_type)
        channel_flow.solve_problem()


def test_channel_flow_convective_term():
    for form_convective_term in ("standard", "rotational", "divergence", "skew_symmetric"):
        channel_flow = ChannelFlowProblem(10, form_convective_term=form_convective_term)
        channel_flow.solve_problem()


def test_couette_flow():
    couette_flow = CouetteProblem(10)
    couette_flow.solve_problem()


def test_gravity_driven_flow():
    gravity_flow = GravityDrivenFlowProblem(50)
    gravity_flow.solve_problem()


if __name__ == "__main__":
    test_blasius_flow()
    test_cavity()
    test_channel_flow()
    test_channel_flow_convective_term()
    test_couette_flow()
    test_gravity_driven_flow()
