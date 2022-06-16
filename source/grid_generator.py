#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from enum import Enum, auto
import glob
import math
from mshr import Sphere, Circle, generate_mesh
from os import path


class GeometryType(Enum):
    spherical_annulus = auto()
    rectangle = auto()
    square = auto()
    other = auto()


class SphericalAnnulusBoundaryMarkers(Enum):
    """
    Simple enumeration to identify the boundaries of a spherical annulus uniquely.
    """
    interior_boundary = auto()
    exterior_boundary = auto()


class SymmetricPipeBoundaryMarkers(Enum):
    """
    Simple enumeration to identify the boundaries of a symmetric pipe mesh uniquely.
    """
    wall = 100
    symmetry = 101
    inlet = 102
    outlet = 103


class HyperCubeBoundaryMarkers(Enum):
    """
    Simple enumeration to identify the boundaries of a hyper rectangle uniquely.
    """
    left = auto()
    right = auto()
    bottom = auto()
    top = auto()
    back = auto()
    front = auto()
    opening = auto()


HyperRectangleBoundaryMarkers = HyperCubeBoundaryMarkers


class CircularBoundary(dlfn.SubDomain):
    def __init__(self, **kwargs):
        super().__init__()
        assert(isinstance(kwargs["mesh"], dlfn.Mesh))
        assert(isinstance(kwargs["radius"], float) and kwargs["radius"] > 0.0)
        self._hmin = kwargs["mesh"].hmin()
        self._radius = kwargs["radius"]

    def inside(self, x, on_boundary):
        # tolerance: half length of smallest element
        tol = self._hmin / 2.
        result = abs(math.sqrt(x[0]**2 + x[1]**2) - self._radius) < tol
        return result and on_boundary


def spherical_shell(dim, radii, n_points=10):
    """
    Creates the mesh of a spherical shell using the mshr module.
    """
    assert isinstance(dim, int)
    assert dim == 2 or dim == 3

    assert isinstance(radii, (list, tuple)) and len(radii) == 2
    ri, ro = radii
    assert isinstance(ri, float) and ri > 0.
    assert isinstance(ro, float) and ro > 0.
    assert ri < ro

    assert isinstance(n_points, int) and n_points >= 0

    # mesh generation
    if dim == 2:
        center = dlfn.Point(0., 0.)
    elif dim == 3:
        center = dlfn.Point(0., 0., 0.)

    if dim == 2:
        domain = Circle(center, ro) \
               - Circle(center, ri)
        mesh = generate_mesh(domain, n_points)
    elif dim == 3:
        domain = Sphere(center, ro) \
               - Sphere(center, ri)
        mesh = generate_mesh(domain, n_points)

    # subdomains for boundaries
    facet_marker = dlfn.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    facet_marker.set_all(0)

    # mark boundaries
    BoundaryMarkers = SphericalAnnulusBoundaryMarkers
    gamma_inner = CircularBoundary(mesh=mesh, radius=ri)
    gamma_inner.mark(facet_marker, BoundaryMarkers.interior_boundary.value)
    gamma_outer = CircularBoundary(mesh=mesh, radius=ro)
    gamma_outer.mark(facet_marker, BoundaryMarkers.exterior_boundary.value)

    return mesh, facet_marker


def hyper_cube(dim, n_points=10):
    """
    Creates a unit hyper cube with an equidistant mesh size.
    """
    assert isinstance(dim, int)
    assert dim == 2 or dim == 3
    assert isinstance(n_points, int) and n_points >= 0

    # mesh generation
    if dim == 2:
        corner_points = (dlfn.Point(0., 0.), dlfn.Point(1., 1.))
        mesh = dlfn.RectangleMesh(*corner_points, n_points, n_points)
    else:
        corner_points = (dlfn.Point(0., 0., 0.), dlfn.Point(1., 1., 1.))
        mesh = dlfn.BoxMesh(*corner_points, n_points, n_points, n_points)
    assert dim == mesh.topology().dim()
    # subdomains for boundaries
    facet_marker = dlfn.MeshFunction("size_t", mesh, dim - 1)
    facet_marker.set_all(0)

    # mark boundaries
    BoundaryMarkers = HyperCubeBoundaryMarkers

    gamma01 = dlfn.CompiledSubDomain("near(x[0], 0.0) && on_boundary")
    gamma02 = dlfn.CompiledSubDomain("near(x[0], 1.0) && on_boundary")
    gamma03 = dlfn.CompiledSubDomain("near(x[1], 0.0) && on_boundary")
    gamma04 = dlfn.CompiledSubDomain("near(x[1], 1.0) && on_boundary")

    gamma01.mark(facet_marker, BoundaryMarkers.left.value)
    gamma02.mark(facet_marker, BoundaryMarkers.right.value)
    gamma03.mark(facet_marker, BoundaryMarkers.bottom.value)
    gamma04.mark(facet_marker, BoundaryMarkers.top.value)

    if dim == 3:
        gamma05 = dlfn.CompiledSubDomain("near(x[2], 0.0) && on_boundary")
        gamma06 = dlfn.CompiledSubDomain("near(x[2], 1.0) && on_boundary")

        gamma05.mark(facet_marker, BoundaryMarkers.back.value)
        gamma06.mark(facet_marker, BoundaryMarkers.front.value)

    return mesh, facet_marker


def hyper_rectangle(first_point, second_point, n_points=10):
    """
    Create a hyper rectangle where the `first_point` and the `second_point`
    are two diagonally opposite corner points.
    """
    assert isinstance(first_point, (tuple, list))
    assert isinstance(second_point, (tuple, list))
    dim = len(first_point)
    assert dim == 2 or dim == 3
    assert len(second_point) == dim
    assert all(all(isinstance(x, float) for x in p) for p in (first_point, second_point))
    assert all(y - x > 0.0 for x, y in zip(first_point, second_point))
    corner_points = (dlfn.Point(*first_point), dlfn.Point(*second_point))

    assert isinstance(n_points, (int, tuple, list))
    if isinstance(n_points, (tuple, list)):
        assert all(isinstance(n, int) and n > 0 for n in n_points)
        assert len(n_points) == dim
    else:
        n_points > 0
        n_points = (n_points, ) * dim

    # mesh generation
    if dim == 2:

        mesh = dlfn.RectangleMesh(*corner_points, *n_points)
    else:
        mesh = dlfn.BoxMesh(*corner_points, *n_points)
    assert dim == mesh.topology().dim()

    # subdomains for boundaries
    facet_marker = dlfn.MeshFunction("size_t", mesh, dim - 1)
    facet_marker.set_all(0)

    # mark boundaries
    BoundaryMarkers = HyperCubeBoundaryMarkers

    gamma01 = dlfn.CompiledSubDomain("near(x[0], val) && on_boundary", val=first_point[0])
    gamma02 = dlfn.CompiledSubDomain("near(x[0], val) && on_boundary", val=second_point[0])
    gamma03 = dlfn.CompiledSubDomain("near(x[1], val) && on_boundary", val=first_point[1])
    gamma04 = dlfn.CompiledSubDomain("near(x[1], val) && on_boundary", val=second_point[1])

    gamma01.mark(facet_marker, BoundaryMarkers.left.value)
    gamma02.mark(facet_marker, BoundaryMarkers.right.value)
    gamma03.mark(facet_marker, BoundaryMarkers.bottom.value)
    gamma04.mark(facet_marker, BoundaryMarkers.top.value)

    if dim == 3:
        gamma05 = dlfn.CompiledSubDomain("near(x[2], val) && on_boundary", val=first_point[2])
        gamma06 = dlfn.CompiledSubDomain("near(x[2], val) && on_boundary", val=second_point[2])

        gamma05.mark(facet_marker, BoundaryMarkers.back.value)
        gamma06.mark(facet_marker, BoundaryMarkers.front.value)

    return mesh, facet_marker


def open_hyper_cube(dim, n_points=10, openings=None):
    """
    Create a hyper cube with openings.

    The openings are specified as a list of tuples containing the location,
    the center and the width of the opening. For example,

        openings = ( ("top", center, width), )

    where in 2D

        center = (0.5, 1.0)
        width = 0.25

    and in 3D

        center = (0.5, 0.5, 1.0)
        width = (0.25, 0.25)
    """
    tol = 1.0e3 * dlfn.DOLFIN_EPS

    if openings is None:  # pragma: no cover
        return hyper_cube(dim, n_points)

    # input check
    assert isinstance(openings, (tuple, list))
    assert all(isinstance(o, (tuple, list)) for o in openings)
    for position, center, width in openings:
        assert position in ("top", "bottom", "left", "right", "front", "back")

        assert isinstance(center, (tuple, list))
        assert len(center) == dim
        assert all(isinstance(x, float) for x in center)

        if isinstance(width, float):
            assert dim == 2
        else:
            assert isinstance(width, (tuple, list))
            assert len(width) == dim - 1
            assert all(isinstance(x, float) and x > 0.0 for x in width)

    # get hyper cube mesh with marked boundaries
    mesh, facet_markers = hyper_cube(dim, n_points)

    # the boundary markers are modified where an opening is located
    for position, center, width in openings:
        if isinstance(width, float):
            width = (width, )
        # find on which the facet the center point is located
        center_point = dlfn.Point(*center)
        facet = None
        for f in dlfn.facets(mesh):
            if f.exterior():
                normal = f.normal()
                center_point_in_facet_plane = False
                for v in dlfn.vertices(f):
                    d = v.point().dot(normal)
                    if abs(normal.dot(center_point) - d) < tol:
                        center_point_in_facet_plane = True
                        break
                if center_point_in_facet_plane is True:
                    facet = f
                    break
        assert facet is not None, "Center point is not on the boundary"

        # get boundary id of the corresponding boundary
        bndry_id = facet_markers[facet]

        # check that center point is on the right part of boundary and
        # modify the boundary markers
        BoundaryMarkers = HyperCubeBoundaryMarkers
        if position in ("left", "right"):
            assert bndry_id in (BoundaryMarkers.left.value, BoundaryMarkers.right.value)
            if position == "left":
                assert bndry_id == BoundaryMarkers.left.value
                str_standard_condition = "near(x[0], 0.0) && on_boundary"
            elif position == "right":
                assert bndry_id == BoundaryMarkers.right.value
                str_standard_condition = "near(x[0], 1.0) && on_boundary"
            if dim == 2:
                str_condition = " && ".join(
                        [str_standard_condition,
                         "-l_y / 2.0 <= (x[1] - c_y) <= l_y / 2.0"])
                gamma = dlfn.CompiledSubDomain(str_condition,
                                               l_y=width[0], c_y=center[1])
            elif dim == 3:
                str_condition = " && ".join(
                        [str_standard_condition,
                         "-l_y / 2.0 <= (x[1] - c_y) <= l_y / 2.0",
                         "-l_z / 2.0 <= (x[2] - c_z) <= l_z / 2.0"])
                gamma = dlfn.CompiledSubDomain(str_condition,
                                               l_y=width[0], l_z=width[1],
                                               c_y=center[1], c_z=center[2])
            gamma.mark(facet_markers, BoundaryMarkers.opening.value)

        elif position in ("bottom", "top"):
            assert bndry_id in (BoundaryMarkers.bottom.value, BoundaryMarkers.top.value), \
                "Boundary id {0} does not mactch the expected values "\
                "({1}, {2})".format(bndry_id, BoundaryMarkers.bottom.value, BoundaryMarkers.top.value)
            if position == "bottom":
                assert bndry_id == BoundaryMarkers.bottom.value, \
                    "Boundary id {0} does not mactch the expected value {1}".format(bndry_id, BoundaryMarkers.bottom.value)
                str_standard_condition = "near(x[1], 0.0) && on_boundary"
            elif position == "top":
                assert bndry_id == BoundaryMarkers.top.value, \
                    "Boundary id {0} does not mactch the expected value {1}".format(bndry_id, BoundaryMarkers.top.value)
                str_standard_condition = "near(x[1], 1.0) && on_boundary"
            if dim == 2:
                str_condition = " && ".join(
                        [str_standard_condition,
                         "std::abs(x[0] - c_x) <= (l_x / 2.0)"])
                gamma = dlfn.CompiledSubDomain(str_condition,
                                               l_x=width[0], c_x=center[0])
            elif dim == 3:
                str_condition = " && ".join(
                        [str_standard_condition,
                         "-l_x / 2.0 <= (x[0] - c_x) <= l_x / 2.0",
                         "-l_z / 2.0 <= (x[2] - c_z) <= l_z / 2.0"])
                gamma = dlfn.CompiledSubDomain(str_condition,
                                               l_x=width[0], l_z=width[1],
                                               c_x=center[0], c_z=center[2])
            gamma.mark(facet_markers, BoundaryMarkers.opening.value)

        elif position in ("back", "front"):
            assert dim == 3
            assert bndry_id in (BoundaryMarkers.back.value, BoundaryMarkers.front.value)
            if position == "back":
                assert bndry_id == BoundaryMarkers.back.value
                str_standard_condition = "near(x[2], 0.0) && on_boundary"
            elif position == "front":
                assert bndry_id == BoundaryMarkers.front.value
                str_standard_condition = "near(x[2], 1.0) && on_boundary"
            str_condition = " && ".join(
                    [str_standard_condition,
                     "-l_x / 2.0 <= (x[0] - c_x) <= l_x / 2.0",
                     "-l_y / 2.0 <= (x[1] - c_y) <= l_y / 2.0"])
            gamma = dlfn.CompiledSubDomain(str_condition,
                                           l_x=width[0], l_y=width[1],
                                           c_x=center[0], c_y=center[1])
            gamma.mark(facet_markers, BoundaryMarkers.opening.value)
        else:  # pragma: no cover
            raise RuntimeError()

    return mesh, facet_markers


def _extract_facet_markers(geo_filename):
    """Extract facet markers from a geo-file and returns them as a dictionary.
    """
    # input check
    assert isinstance(geo_filename, str)
    assert path.exists(geo_filename)
    assert geo_filename.endswith(".geo")
    # read file
    facet_markers = dict()
    with open(geo_filename, "r") as file:
        lines = file.readlines()
        for line in lines:
            if "Physical Curve" in line or "Physical Line" in line:
                line = line[line.index("(")+1:line.index(")")]
                assert "," in line
                description, number = line.split(",")
                # facet id
                number = number.strip(" ")
                assert number.isnumeric()
                facet_id = int(number)
                # boundary description
                description = description.strip(" ")
                description = description.strip("'")
                description = description.strip('"')
                assert description.replace(" ", "").isalpha()
                # add to dictionary
                assert description not in facet_markers
                facet_markers[description] = facet_id

    return facet_markers


def _locate_file(basename):
    """Locate a file in the current directory.
    """
    file_extension = path.splitext(basename)[1]
    files = glob.glob("./*" + file_extension, recursive=True)
    files += glob.glob("./*/*" + file_extension, recursive=True)
    files += glob.glob("./*/*/*" + file_extension, recursive=True)
    file = None
    for f in files:
        if basename in f:
            file = f
            break
    if file is not None:
        assert path.exists(file)
    return file


def _read_external_mesh(basename):
    """This script reads a gmsh file. This file must be located inside the project
    directory. If this script is used inside the docker container, the
    associated xdmf files must already exist.
    """
    # locate geo file
    assert isinstance(basename, str)
    assert basename.endswith(".geo")
    geo_file = _locate_file(basename)
    assert geo_file is not None
    facet_marker_map = _extract_facet_markers(geo_file)
    # define xdmf files
    xdmf_file = _locate_file(basename.replace(".geo", ".xdmf"))
    xdmf_facet_marker_file = _locate_file(basename.replace(".geo", "_facet_markers.xdmf"))
    # check if xdmf files exist
    if xdmf_file is None or xdmf_facet_marker_file is None:
        from grid_tools import generate_xdmf_mesh
        xdmf_file, xdmf_facet_marker_file = generate_xdmf_mesh(geo_file)
    # read xdmf files
    assert path.exists(xdmf_file)
    mesh = dlfn.Mesh()
    with dlfn.XDMFFile(xdmf_file) as infile:
        infile.read(mesh)
    # read facet markers
    space_dim = mesh.geometry().dim()
    mvc = dlfn.MeshValueCollection("size_t", mesh, space_dim - 1)
    assert path.exists(xdmf_facet_marker_file)
    with dlfn.XDMFFile(xdmf_facet_marker_file) as infile:
        infile.read(mvc, "facet_markers")
    facet_markers = dlfn.cpp.mesh.MeshFunctionSizet(mesh, mvc)

    return mesh, facet_markers, facet_marker_map


def backward_facing_step():
    """Create a mesh of a channel with a backward facing step.
    """
    return _read_external_mesh("BackwardFacingStep.geo")


def blasius_plate():
    """Create a mesh of a plate embedded in free space.
    """
    return _read_external_mesh("BlasiusFlowProblem.geo")


def channel_with_cylinder():
    """Create a mesh of a channel with a cylinder.
    """
    return _read_external_mesh("DFGBenchmark.geo")
