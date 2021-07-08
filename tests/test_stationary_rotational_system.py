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
from grid_generator import spherical_shell
from grid_generator import SphericalAnnulusBoundaryMarkers

dlfn.set_log_level(20)


class RotationalCouetteFlow(StationaryProblem):
    def __init__(self, n_refinements, radii, main_dir=None):
        super().__init__(main_dir)

        self._radii = radii
        self._n_refinements = n_refinements
        self._problem_name = "RotationalCouette"

        self.set_parameters(Re=1000.0, Ro=1.0, Omega=dlfn.Constant(1), Alpha=dlfn.Constant(0))

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = spherical_shell(2, self._radii, self._n_refinements)

    def set_boundary_conditions(self):
        # velocity boundary conditions
        velocity_str = ("x[1]","-x[0]")
        velocity = dlfn.Expression(velocity_str, degree=2)
        self._bcs = ((VelocityBCType.no_slip,SphericalAnnulusBoundaryMarkers.exterior_boundary.value, None),
                     (VelocityBCType.function, SphericalAnnulusBoundaryMarkers.interior_boundary.value, velocity))
        
    
def test_rotational_couette():
    rotational_couette_flow = RotationalCouetteFlow(0, (0.25, 1.0))
    rotational_couette_flow.solve_problem()


if __name__ == "__main__":
    test_rotational_couette()
