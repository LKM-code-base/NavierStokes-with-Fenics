#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
from grid_generator import hyper_cube
from grid_generator import open_hyper_cube
from grid_generator import hyper_rectangle
from grid_generator import spherical_shell
from grid_generator import _extract_facet_markers
from grid_generator import _extract_cell_markers
from grid_generator import backward_facing_step
from grid_generator import blasius_plate
from grid_generator import channel_with_cylinder
from grid_generator import rectangle_with_two_materials
from grid_generator import rectangle_with_three_materials
from grid_generator import cube_with_single_material
from grid_generator import cube_with_three_materials
import subprocess
from os import path


def test_hyper_cube():
    # two-dimensional case
    _, _ = hyper_cube(2, 8)
    # three-dimensional case
    _, _ = hyper_cube(3, 8)


def test_open_hyper_cube():
    # two-dimensional case
    openings = (("left", (0.0, 0.5), 0.1),
                ("right", (1.0, 0.7), 0.1),
                ("bottom", (0.7, 0.0), 0.05),
                ("top", (0.5, 1.0), 0.8))
    _, _ = open_hyper_cube(2, 8, openings)
    # three-dimensional case
    openings = (("left", (0.0, 0.5, 0.5), (0.1, 0.2)),
                ("right", (1.0, 0.7, 0.3), (0.1, 0.1)),
                ("bottom", (0.7, 0.0, 0.7), (0.05, 0.2)),
                ("top", (0.5, 1.0, 0.2), (0.8, 0.8)),
                ("back", (0.7, 0.3, 0.0), (0.05, 0.1)),
                ("front", (0.5, 0.25, 1.0), (0.2, 0.3))
                )
    _, _ = open_hyper_cube(3, 8, openings)


def test_hyper_rectangle():
    # two-dimensional case
    _, _ = hyper_rectangle((0.0, 0.0), (10.0, 1.0), 10)
    _, _ = hyper_rectangle((0.0, 0.0), (10.0, 1.0), (50, 5))
    # three-dimensional case
    _, _ = hyper_rectangle((0.0, 0.0, 0.0), (10.0, 1.0, 2.0), 8)
    _, _ = hyper_rectangle((0.0, 0.0, 0.0), (10.0, 1.0, 2.0), (50, 5, 10))


def test_spherical_shell():
    # two-dimensional case
    _, _ = spherical_shell(2, (0.3, 1.0), 25)
    # three-dimensional case
    _, _ = spherical_shell(3, (0.3, 1.0), 25)


def test_extract_markers():
    url_str = "https://raw.githubusercontent.com/LKM-code-base/Gmsh-collection/main/meshes/CubeThreeMaterials.geo"
    subprocess.run(["wget", "--no-verbose", url_str], check=True, stdout=subprocess.DEVNULL)
    fname = "CubeThreeMaterials.geo"
    geo_files = glob.glob("*.geo", recursive=True)
    for file in geo_files:
        if fname in file:
            geo_file = file
            break
    assert path.exists(geo_file)
    _ = _extract_facet_markers(geo_file, 3)
    _ = _extract_cell_markers(geo_file, 3)
    subprocess.run(["rm", geo_file], check=True)


def test_external_meshes():
    _ = backward_facing_step()
    _ = blasius_plate()
    _ = channel_with_cylinder()
    _ = cube_with_single_material()
    _ = cube_with_three_materials()
    _ = rectangle_with_two_materials()
    _ = rectangle_with_three_materials()


if __name__ == "__main__":
    test_hyper_cube()
    test_open_hyper_cube()
    test_spherical_shell()
    test_extract_markers()
    test_external_meshes()
