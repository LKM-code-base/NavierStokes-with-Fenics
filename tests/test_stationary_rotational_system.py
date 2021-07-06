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


class RotationalCavityProblem(StationaryProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir)

        self._n_points = n_points
        self._problem_name = "Rotational"

        self.set_parameters(Re=10.0, Ro=0.01, Omega=dlfn.Constant(50), Alpha=dlfn.Constant(0))

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_boundary_conditions(self):
        # velocity boundary conditions
        self._bcs = ((VelocityBCType.no_slip, HyperCubeBoundaryMarkers.left.value, None),
                     (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.right.value, None),
                     (VelocityBCType.no_slip, HyperCubeBoundaryMarkers.bottom.value, None),
                     (VelocityBCType.constant, HyperCubeBoundaryMarkers.top.value, (1.0, 0.0)))
        
        
class RotationalCouetteFlow(StationaryProblem):
    def __init__(self, n_refinements, radii, main_dir=None):
        super().__init__(main_dir)

        assert isinstance(radii, (list, tuple)) and len(radii) == 2, "radii must be tuple of length 2"
        self._radii = radii
        self._n_refinements = n_refinements
        self._problem_name = "RotationalCouette"

        self.set_parameters(Re=10.0, Ro=0.01, Omega=dlfn.Constant(50), Alpha=dlfn.Constant(0))

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = spherical_shell(2, self._radii, self._n_refinements)

    def set_boundary_conditions(self):
        # velocity boundary conditions
        self._bcs = ((VelocityBCType.no_slip,SphericalAnnulusBoundaryMarkers.interior_boundary.value, None),
                     (VelocityBCType.constant, SphericalAnnulusBoundaryMarkers.exterior_boundary.value, (1.0, 0.0)))
        
 
def test_rotational_cavity():
    rotational_cavity_flow = RotationalCavityProblem(25)
    rotational__cavity_flow.solve_problem()
    
def test_rotational_couette():
    rotational_couette_flow = RotationalCouetteFlow(3, (1.0, 0.25))
    rotational__couette_flow.solve_problem()


if __name__ == "__main__":
    test_rotational_cavity()
    test_rotational_couette()
