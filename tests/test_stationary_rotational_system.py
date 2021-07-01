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

        self.set_parameters(Re=10.0, Ro=1.0)

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_boundary_conditions(self):
        # velocity boundary conditions
        self._bcs = ((VelocityBCType.no_slip, HyperCubeBoundaryMarkers.left.value, None),
                     (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.right.value, None),
                     (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.bottom.value, None),
                     (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.top.value, None)
    
    # defining angular velocity and acceleration
    Omega = dlfn.Constant(1.0)
    Alpha = dlfn.Constant(1.0)
    
    def set_coriolis_force(self):
        assert isinstance(Omega, dlfn.Constant)
        if self._mesh.geometry().dim() is 2:
            assert len(Omega) == 1
            set._coriolis_force = 2 * dlfn.as_vector((-Omega * sol_v[1], Omega * sol_v[0]))  
        else:
            assert len(Omega) == 3
            self._coriolis_force = 2 * dlfn.cross(Omega, sol_v)
            
    def set_euler_force(self):
        assert isinstance(Alpha, dlfn.Constant)
        if self._mesh.geometry().dim() is 2:
            assert len(Alpha) == 1
            set._euler_force = dlfn.as_vector((-Alpha * sol_v[1], Alpha * sol_v[0]))
        else:
            assert len(Alpha) == 3
            self._euler_force = dlfn.cross(Alpha, sol_v)

def test_rotational():
    rotational_flow = RotationalProblem(25)
    rotational_flow.solve_problem()


if __name__ == "__main__":
    test_rotational()
