#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn

__all__ = ["boundary_normal"]

def boundary_normal(mesh, facet_markers, bndry_id):
    """
    Extracts the normal vector of the boundary marked by the boundary id
    by checking that
        1. the facet normal vectors are co-linear
        2. the vector connecting two face midpoints is tangential to both
           normal vectors.

    Returns a tuple of float representing the normal.
    """
    assert isinstance(mesh, dlfn.Mesh)
    assert isinstance(mesh, dlfn.MeshFunctionSizet)
    assert isinstance(bndry_id, int)

    tol = 1.0e3 * dlfn.DOLFIN_EPS
    normal_vectors = []
    midpoints = []
    for f in dlfn.facets(mesh):
        if f.exterior() and facet_markers[f] == bndry_id:
            current_normal = f.normal()
            current_midpoint = f.midpoint()
            for normal, midpoint in zip(normal_vectors, midpoints):
                # check that normal vectors point in the same direction
                assert current_normal.dot(normal) > 0.0
                # check that normal vector are parallel
                if abs(current_normal.dot(normal) - 1.0) > tol:
                    raise ValueError("Boundary facets do not share common normal.")
                # compute a tangential vector as connection vector of two
                # midpoints
                midpoint_connection = midpoint - current_midpoint
                # check that tangential vector is orthogonal to both normal
                # vectors
                if abs(midpoint_connection.dot(normal)) > tol:
                    raise ValueError("Midpoint connection vector is not tangential to boundary facets.")
                if abs(midpoint_connection.dot(current_normal)) > tol:
                    raise ValueError("Midpoint connection vector is not tangential to boundary facets.")
            normal_vectors.append(current_normal)
            midpoints.append(midpoint)

    dim = mesh.topology().dim()
    normal = normal_vectors[0]

    return (normal[d] for d in range(dim))
