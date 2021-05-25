#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from grid_generator import hyper_cube, HyperCubeBoundaryMarkers
from auxiliary_methods import boundary_normal


def compare_tuples(a, b):
    assert a == b, "The tuple {0} is not equal to the tuple {1}".format(a, b)


def test_boundary_normal():
    # two-dimensional case
    mesh, boundary_markers = hyper_cube(2, 8)

    normal_vectors = [(-1.0, 0.0), (1.0, 0.0),
                      (0.0, -1.0), (0.0, 1.0)]
    boundary_ids = (HyperCubeBoundaryMarkers.left.value,
                    HyperCubeBoundaryMarkers.right.value,
                    HyperCubeBoundaryMarkers.bottom.value,
                    HyperCubeBoundaryMarkers.top.value)

    for normal, bndry_id in zip(normal_vectors, boundary_ids):
        computed_normal = boundary_normal(mesh, boundary_markers, bndry_id)
        compare_tuples(normal, computed_normal)

    # three-dimensional case
    mesh, boundary_markers = hyper_cube(3, 8)

    normal_vectors = [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                      (0.0, -1.0, 0.0), (0.0, 1.0, 0.0),
                      (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)]
    boundary_ids = (HyperCubeBoundaryMarkers.left.value,
                    HyperCubeBoundaryMarkers.right.value,
                    HyperCubeBoundaryMarkers.bottom.value,
                    HyperCubeBoundaryMarkers.top.value,
                    HyperCubeBoundaryMarkers.back.value,
                    HyperCubeBoundaryMarkers.front.value)

    for normal, bndry_id in zip(normal_vectors, boundary_ids):
        computed_normal = boundary_normal(mesh, boundary_markers, bndry_id)
        compare_tuples(normal, computed_normal)


if __name__ == "__main__":
    test_boundary_normal()
