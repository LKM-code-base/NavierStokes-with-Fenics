import dolfin as dlfn
from auxiliary_classes import EquationCoefficientHandler
from ns_problem import InstationaryProblem
from ns_problem import VelocityBCType, PressureBCType
from ns_ipcs_solver import IPCSSolver
from grid_generator import hyper_rectangle
from grid_generator import HyperRectangleBoundaryMarkers

dlfn.set_log_level(30)


class ChannelFlowProblem(InstationaryProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir, start_time=0.0, end_time=1.0,
                         desired_start_time_step=0.002, n_max_steps=10)
        self._n_points = n_points
        self._problem_name = "ChannelFlow"
        self._output_frequency = 1
        self._postprocessing_frequency = 1
        self.set_solver_class(IPCSSolver)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_rectangle((0.0, 0.0), (10.0, 1.0),
                                                             (10 * self._n_points, self._n_points))

    def set_equation_coefficients(self):
        self._coefficient_handler = EquationCoefficientHandler(Re=10.0)

    def set_initial_conditions(self):
        self._initial_conditions = dict()
        self._initial_conditions["velocity"] = (0.0, 0.0)
        self._initial_conditions["pressure"] = 0.0

    def set_boundary_conditions(self):
        # velocity boundary conditions
        inlet_velocity = dlfn.Expression(("6.0*x[1]/h*(1.0-x[1]/h)", "0.0"), h=1.0,
                                         degree=2)
        Markers = HyperRectangleBoundaryMarkers
        self._bcs = ((PressureBCType.constant, Markers.right.value, 0.0),
                     (VelocityBCType.function, Markers.left.value, inlet_velocity),
                     (VelocityBCType.no_slip, Markers.bottom.value, None),
                     (VelocityBCType.no_slip, Markers.top.value, None))

    def postprocess_solution(self):
        # add pressure gradient to the field output
        self._add_to_field_output(self._compute_pressure_gradient())
        # add vorticity to the field output
        self._add_to_field_output(self._compute_vorticity())


def test_channel_flow():
    channel_flow = ChannelFlowProblem(5)
    channel_flow.solve_problem()


if __name__ == "__main__":
    test_channel_flow()
