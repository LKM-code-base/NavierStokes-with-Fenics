#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os import path
import subprocess

from enum import Enum, auto

import math

import dolfin as dlfn
from mshr import Sphere, Circle, generate_mesh


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


def spherical_shell(dim, radii, n_refinements=0):
    assert isinstance(dim, int)
    assert dim == 2 or dim == 3

    assert isinstance(radii, (list, tuple)) and len(radii) == 2
    ri, ro = radii
    assert isinstance(ri, float) and ri > 0.
    assert isinstance(ro, float) and ro > 0.
    assert ri < ro

    assert isinstance(n_refinements, int) and n_refinements >= 0

    # mesh generation
    if dim == 2:
        center = dlfn.Point(0., 0.)
    elif dim == 3:
        center = dlfn.Point(0., 0., 0.)

    if dim == 2:
        domain = Circle(center, ro) \
               - Circle(center, ri)
        mesh = generate_mesh(domain, 75)
    elif dim == 3:
        domain = Sphere(center, ro) \
               - Sphere(center, ri)
        mesh = generate_mesh(domain, 15)

    # mesh refinement
    for i in range(n_refinements):
        mesh = dlfn.refine(mesh)

    # subdomains for boundaries
    facet_marker = dlfn.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    facet_marker.set_all(0)

    # mark boundaries
    BoundaryMarkers = SphericalAnnulusBoundaryMarkers
    gamma_inner = CircularBoundary(mesh=mesh, radius=ro)
    gamma_inner.mark(facet_marker, BoundaryMarkers.interior_boundary.value)
    gamma_outer = CircularBoundary(mesh=mesh, radius=ro)
    gamma_outer.mark(facet_marker, BoundaryMarkers.exterior_boundary.value)

    return mesh, facet_marker


def hyper_cube(dim, n_points=10):
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

    if openings is None:
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
        else:
            raise RuntimeError()

    return mesh, facet_markers


def converging_diverging_pipe():
    # define location of gmsh files
    geo_file = path.join(os.getcwd(), "gmsh", "converging_diverging_pipe.geo")
    assert path.exists(geo_file)
    msh_file = geo_file.replace(".geo", ".msh")

    # check if msh file exists
    if not path.exists(msh_file):
        subprocess.run(["gmsh", geo_file, "-2", "-o " + msh_file], check=True)
    assert path.exists(msh_file)

    # convert msh files
    xml_file = geo_file.replace(".geo", ".xml")
    subprocess.run(["dolfin-convert", msh_file, xml_file], check=True)
    assert path.exists(msh_file)

    physical_regions_xml_file = xml_file.replace(".xml", "_physical_region.xml")
    assert path.exists(physical_regions_xml_file)

    BoundaryMarkers = SymmetricPipeBoundaryMarkers
    cnt = 0
    with open(geo_file, "r") as finput:
        for line in finput:
            if "boundary" in line:
                boundary_id = int(line.split("=")[1].split(";")[0])
                if "wall" in line:
                    assert boundary_id == BoundaryMarkers.wall.value
                    cnt += 1
                elif "symmetry" in line or "symmetric" in line:
                    assert boundary_id == BoundaryMarkers.symmetry.value
                    cnt += 1
                elif "inlet" in line:
                    assert boundary_id == BoundaryMarkers.inlet.value
                    cnt += 1
                elif "outlet" in line:
                    assert boundary_id == BoundaryMarkers.outlet.value
                    cnt += 1
            if cnt == 4:
                break

    mesh = dlfn.Mesh(xml_file)
    facet_marker = dlfn.MeshFunction("size_t", mesh, physical_regions_xml_file)

    return mesh, facet_marker
