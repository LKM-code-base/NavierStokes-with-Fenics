import dolfin as dlfn
from auxiliary_classes import AngularVelocityVector, FunctionTime, EquationCoefficientHandler
from ns_problem import StationaryProblem
from ns_solver_base import VelocityBCType
from grid_generator import spherical_shell
from grid_generator import SphericalAnnulusBoundaryMarkers

dlfn.set_log_level(20)


class AngularVelocityFunction(FunctionTime):
    def __init__(self):
        super().__init__(1)

    def value(self):
        return 1.0


class RotatingCouetteFlow(StationaryProblem):
    def __init__(self, n_points, radii, main_dir=None):
        super().__init__(main_dir)
        self._radii = radii
        self._n_points = n_points
        self._problem_name = "RotationalCouette"

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = spherical_shell(2, self._radii, self._n_points)

    def set_angular_velocity(self):
        function = AngularVelocityFunction()
        self._angular_velocity = AngularVelocityVector(2, function=function)

    def set_equation_coefficients(self):
        self._coefficient_handler = EquationCoefficientHandler(Re=1000.0, Ro=1.0)

    def set_boundary_conditions(self):
        # velocity boundary conditions
        velocity_str = ("x[1]", "-x[0]")
        velocity = dlfn.Expression(velocity_str, degree=2)
        self._bcs = ((VelocityBCType.no_slip, SphericalAnnulusBoundaryMarkers.exterior_boundary.value, None),
                     (VelocityBCType.function, SphericalAnnulusBoundaryMarkers.interior_boundary.value, velocity))


def test_rotating_couette_flow():
    rotational_couette_flow = RotatingCouetteFlow(60, (0.25, 1.0))
    rotational_couette_flow.solve_problem()


if __name__ == "__main__":
    test_rotating_couette_flow()
