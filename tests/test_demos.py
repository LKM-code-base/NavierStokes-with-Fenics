#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from os import path

current_path = os.getcwd()
demo_path = path.join(current_path, "demo")

def test_cavity():
    fname = path.join(demo_path, "cavity_flow.py")
    assert path.exists(fname)
    exec(open(fname).read())
    
def test_gravity_driven_flow():
    fname = path.join(demo_path, "gravity_driven_flow.py")
    assert path.exists(fname)
    exec(open(fname).read())