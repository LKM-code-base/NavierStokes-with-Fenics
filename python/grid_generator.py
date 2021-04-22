#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from enum import Enum, auto

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

class RectangleBoundaryMarkers(Enum):
    """
    Simple enumeration to identify the boundaries of a rectangle uniquely.
    """
    top = auto()
    bottom = auto()
    left = auto()
    right = auto()
    
class CuboidBoundaryMarkers(Enum):
    """
    Simple enumeration to identify the boundaries of a rectangle uniquely.
    """
    top = auto()
    bottom = auto()
    left = auto()
    right = auto()
    front = auto()
    back = auto()

import dolfin
class CircularBoundary(dolfin.SubDomain):
    def __init__(self, **kwargs):
        super().__init__()
        assert(isinstance(kwargs["mesh"], dolfin.Mesh))
        assert(isinstance(kwargs["radius"], float) and kwargs["radius"] > 0.0)
        self._hmin = kwargs["mesh"].hmin()
        self._radius = kwargs["radius"]
    def inside(self, x, on_boundary):
        # tolerance: half length of smallest element
        tol = self._hmin / 2. 
        from math import sqrt
        result = abs(sqrt(x[0]**2 + x[1]**2) - self._radius) < tol
        return result and on_boundary

def spherical_shell(dim, radii, n_refinements = 0):
    assert isinstance(dim, int)
    assert dim == 2 or dim == 3

    assert isinstance(radii, (list, tuple)) and len(radii) == 2
    ri, ro = radii
    assert isinstance(ri, float) and ri > 0.
    assert isinstance(ro, float) and ro > 0.
    assert ri < ro

    assert isinstance(n_refinements, int) and n_refinements >= 0
    
    # mesh generation
    from dolfin import Point
    if dim == 2:
        center = Point(0., 0.)
    elif dim == 3:
        center = Point(0., 0., 0.)
    
    from mshr import Sphere, Circle, generate_mesh
    if dim == 2:
        domain = Circle(center, ro) \
               - Circle(center, ri)
        mesh = generate_mesh(domain, 75)
    elif dim == 3:
        domain = Sphere(center, ro) \
               - Sphere(center, ri)
        mesh = generate_mesh(domain, 15)
               
    # mesh refinement
    from dolfin import refine
    for i in range(n_refinements):
        mesh = refine(mesh)
        
    # subdomains for boundaries
    from dolfin import MeshFunction
    facet_marker = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    facet_marker.set_all(0)
    
    # mark boundaries
    BoundaryMarkers = SphericalAnnulusBoundaryMarkers
    gamma_inner = CircularBoundary(mesh = mesh, radius = ro)
    gamma_inner.mark(facet_marker, BoundaryMarkers.interior_boundary.value)
    gamma_outer = CircularBoundary(mesh = mesh, radius = ro)
    gamma_outer.mark(facet_marker, BoundaryMarkers.exterior_boundary.value)
    
    return mesh, facet_marker
    
def square_cavity(dim, n_points = 10):
    assert isinstance(dim, int)
    assert dim == 2 or dim == 3
    assert isinstance(n_points, int) and n_points >= 0
    
    # mesh generation
    from dolfin import Point, RectangleMesh, BoxMesh
    if dim == 2:
        mesh = RectangleMesh(Point(0., 0.), Point(1., 1.),
                             n_points, n_points)
    else:
        mesh = BoxMesh(Point(0., 0., 0.), Point(1., 1., 1.),
                       n_points, n_points, n_points)
               
    # subdomains for boundaries
    from dolfin import MeshFunction
    facet_marker = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    facet_marker.set_all(0)
    
    # mark boundaries
    from dolfin import CompiledSubDomain
    if dim == 2:
        BoundaryMarkers = RectangleBoundaryMarkers
        
        gamma01 = CompiledSubDomain("near(x[0], 0.0) && on_boundary")
        gamma02 = CompiledSubDomain("near(x[0], 1.0) && on_boundary")
        gamma03 = CompiledSubDomain("near(x[1], 0.0) && on_boundary")
        gamma04 = CompiledSubDomain("near(x[1], 1.0) && on_boundary")
        
        gamma01.mark(facet_marker, BoundaryMarkers.left.value)
        gamma02.mark(facet_marker, BoundaryMarkers.right.value)
        gamma03.mark(facet_marker, BoundaryMarkers.bottom.value)
        gamma04.mark(facet_marker, BoundaryMarkers.top.value)
    else:
        BoundaryMarkers = CuboidBoundaryMarkers
        
        gamma01 = CompiledSubDomain("near(x[0], 0.0) && on_boundary")
        gamma02 = CompiledSubDomain("near(x[0], 1.0) && on_boundary")
        gamma03 = CompiledSubDomain("near(x[1], 0.0) && on_boundary")
        gamma04 = CompiledSubDomain("near(x[1], 1.0) && on_boundary")
        gamma05 = CompiledSubDomain("near(x[2], 0.0) && on_boundary")
        gamma06 = CompiledSubDomain("near(x[2], 1.0) && on_boundary")

        gamma01.mark(facet_marker, BoundaryMarkers.left.value)
        gamma02.mark(facet_marker, BoundaryMarkers.right.value)
        gamma03.mark(facet_marker, BoundaryMarkers.bottom.value)
        gamma04.mark(facet_marker, BoundaryMarkers.top.value)
        gamma05.mark(facet_marker, BoundaryMarkers.back.value)
        gamma06.mark(facet_marker, BoundaryMarkers.front.value)

    return mesh, facet_marker
    
def converging_diverging_pipe():
    import os
    import subprocess
    from os import path
    # define location of gmsh files
    geo_file = path.join(os.getcwd(), "gmsh", "converging_diverging_pipe.geo")
    assert path.exists(geo_file)
    msh_file = geo_file.replace(".geo", ".msh")
    
    # check if msh file exists
    if not path.exists(msh_file):
        subprocess.run(["gmsh", geo_file, "-2", "-o "  + msh_file], check=True)
    assert path.exists(msh_file)
    
    # convert msh files
    xml_file = geo_file.replace(".geo", ".xml")
    subprocess.run(["dolfin-convert", msh_file, xml_file], check=True)
    assert path.exists(msh_file)

    physical_regions_xml_file = xml_file.replace(".xml","_physical_region.xml")
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

    from dolfin import Mesh, MeshFunction
    mesh = Mesh(xml_file)
    facet_marker = MeshFunction("size_t", mesh, physical_regions_xml_file)
    
    return mesh, facet_marker