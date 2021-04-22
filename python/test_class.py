#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cavity_flow as cf
# general parameters
Re = 200.

cavity_problem = cf.CavityFlowProblem(25)
cavity_problem.solve_problem(Re=10.)