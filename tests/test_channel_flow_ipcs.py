import dolfin as dlfn
from ns_problem import InstationaryProblem
from ns_problem import VelocityBCType
from ns_ipcs_solver import IPCSSolver
from grid_generator import hyper_rectangle
from grid_generator import HyperRectangleBoundaryMarkers

dlfn.set_log_level(30)


class ChannelFlowProblem(InstationaryProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir, start_time=0.0, end_time=1.0,
                         desired_start_time_step=0.002, n_max_steps=500)

        self._n_points = n_points
        self._problem_name = "ChannelFlow"

        self.set_parameters(Re=500.0)

        self._output_frequency = 10
        self._postprocessing_frequency = 10
        self.set_solver_class(IPCSSolver)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_rectangle((0.0, 0.0), (10.0, 1.0),
                                                             (10 * self._n_points, self._n_points))

    def set_initial_conditions(self):
        self._initial_conditions = dict()
        self._initial_conditions["velocity"] = (1.0, 1.0)
        # self._initial_conditions["pressure"] = (1.0)

    def set_boundary_conditions(self):
        # velocity boundary conditions
        inlet_velocity = dlfn.Expression(("1.0*x[1]*(1.0-x[1]) * (1.0 + 0.5 * sin(M_PI * 1))", "0.0"),
                                         degree=2)
        self._bcs = ((VelocityBCType.function, HyperRectangleBoundaryMarkers.left.value, inlet_velocity),
                     (VelocityBCType.no_slip, HyperRectangleBoundaryMarkers.bottom.value, None),
                     (VelocityBCType.no_slip, HyperRectangleBoundaryMarkers.top.value, None))

    def postprocess_solution(self):
        # add pressure gradient to the field output
        self._add_to_field_output(self._compute_pressure_gradient())
        # add vorticity to the field output
        self._add_to_field_output(self._compute_vorticity())


def test_channel_flow():
    channel_flow = ChannelFlowProblem(10)
    channel_flow.solve_problem()


if __name__ == "__main__":
    test_channel_flow()
