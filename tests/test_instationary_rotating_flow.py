import dolfin as dlfn
from auxiliary_classes import AngularVelocityVector, FunctionTime, EquationCoefficientHandler
from ns_problem import InstationaryProblem
from ns_bdf_solver import ImplicitBDFSolver
from ns_solver_base import VelocityBCType
from grid_generator import spherical_shell
from grid_generator import SphericalAnnulusBoundaryMarkers

dlfn.set_log_level(20)


class AngularVelocityFunction(FunctionTime):
    def __init__(self):
        super().__init__(1)

        self._ramp_time = 1.0
        self._alpha_acc = 1.0

    def value(self):
        if self._current_time < self._ramp_time:
            return self._alpha_acc * self._current_time
        else:
            return self._alpha_acc * self._ramp_time

    def derivative(self):
        if self._current_time < self._ramp_time:
            return self._alpha_acc
        else:
            return 0.0


class InstationaryRotatingCouetteFlow(InstationaryProblem):
    def __init__(self, n_points, radii, main_dir=None):
        super().__init__(main_dir, start_time=0.0, end_time=2.0,
                         desired_start_time_step=0.1, n_max_steps=10)

        # parameters for mesh
        self._radii = radii
        self._n_points = n_points
        self._problem_name = "InstationaryRotatingCouette"

        self._alpha_start = 1.0
        self._acceleration_time = 1.0

        self._output_frequency = 20
        self._postprocessing_frequency = 20

        # parameters for error calculation
        self._error_frequency = 400
        ri, ro = self._radii

        # calculating omega after the acceleration phase
        omega_end = self._alpha_start * self._acceleration_time

        # analytical solution for rotating outer cylinder transformed into the rotating coordinate system
        velocity_exact_x = "- x[1] * omega * (ri * pow(ro,2) / (pow(ri,2) - pow(ro,2)) *" + \
            "(ri / (pow(x[0],2) + pow(x[1],2)) - 1 / ri) - 1)"
        velocity_exact_y = "omega * x[0] * (ri * pow(ro,2) / (pow(ri,2) - pow(ro,2)) *" + \
            " (ri / (pow(x[0],2) + pow(x[1],2)) - 1 / ri) - 1)"
        self._velocity_exact = dlfn.Expression([velocity_exact_x, velocity_exact_y],
                                               degree=2, omega=omega_end, ri=ri, ro=ro)

        self.set_solver_class(ImplicitBDFSolver)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = spherical_shell(2, self._radii, self._n_points)

    def set_angular_velocity(self):
        function = AngularVelocityFunction()
        self._angular_velocity = AngularVelocityVector(2, function=function)

    def set_equation_coefficients(self):
        self._coefficient_handler = EquationCoefficientHandler(Re=200.0, Ro=1.0)

    def set_initial_conditions(self):
        self._initial_conditions = dict()
        self._initial_conditions["velocity"] = (0.0, 0.0)

    def set_boundary_conditions(self):
        # velocity boundary conditions
        omega_end = self._alpha_start * self._acceleration_time
        velocity_str = ("x[1]*omega* ( (t >= t_acceleration) ? 1.0: t / t_acceleration)",
                        "-x[0]*omega* ( (t >= t_acceleration) ? 1.0: t / t_acceleration)")
        velocity = dlfn.Expression(velocity_str, degree=2, omega=omega_end, t_acceleration=self._acceleration_time, t=0.0)
        self._bcs = ((VelocityBCType.no_slip, SphericalAnnulusBoundaryMarkers.exterior_boundary.value, None),
                     (VelocityBCType.function, SphericalAnnulusBoundaryMarkers.interior_boundary.value, velocity))


def test_instat_rotational_couette():
    instat_rotational_couette_flow = InstationaryRotatingCouetteFlow(10, (0.25, 0.5))
    instat_rotational_couette_flow.solve_problem()


if __name__ == "__main__":
    test_instat_rotational_couette()
