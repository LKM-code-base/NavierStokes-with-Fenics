import dolfin as dlfn
from ns_problem import StationaryProblem
from ns_solver_base import VelocityBCType
from ns_solver_base import PressureBCType
from ns_solver_base import TractionBCType
from grid_generator import hyper_cube
from grid_generator import hyper_rectangle
from grid_generator import open_hyper_cube
from grid_generator import HyperCubeBoundaryMarkers
from grid_generator import HyperRectangleBoundaryMarkers

dlfn.set_log_level(20)


class RotationalProblem(StationaryProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir)

        self._n_points = n_points
        self._problem_name = "Rotational"

        self.set_parameters(Re=10.0, Ro=1.0, Omega=dlfn.Constant(50), Alpha=dlfn.Constant(0))

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_boundary_conditions(self):
        # velocity boundary conditions
        self._bcs = ((VelocityBCType.no_slip, HyperCubeBoundaryMarkers.left.value, None),
                     (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.right.value, None),
                     (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.bottom.value, None),
                     (VelocityBCType.constant, HyperCubeBoundaryMarkers.top.value, (1.0, 0.0)))
 
def test_rotational():
    rotational_flow = RotationalProblem(25)
    rotational_flow.solve_problem()


if __name__ == "__main__":
    test_rotational()
