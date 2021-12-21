[![codecov](https://codecov.io/gh/LKM-code-base/NavierStokes-with-Fenics/branch/main/graph/badge.svg?token=3WG1X3GHE1)](https://codecov.io/gh/LKM-code-base/NavierStokes-with-Fenics)
[![Tester](https://github.com/LKM-code-base/NavierStokes-with-Fenics/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/LKM-code-base/NavierStokes-with-Fenics/actions/workflows/python-package-conda.yml)

# Flow problems with FEniCS

Solving flow problems in FEniCS.


ToDo list:

- [ ] add channel flow test with periodic bcs
- [ ] add parameter structure
- [ ] add parameter structure
- [ ] fix channel flow test with pure pressure bcs
- [ ] add Taylor-Green vortex
- [ ] modify computation of flow potential and stream function
- [ ] add serialization and restart features
- [ ] add Schur complement preconditioners

Solvers included:

- [x] stationary Navier-Stokes: monolithic, direct LU
- [x] instationary Navier-Stokes: monolithic, implicit BDF-2 time-stepping, direct LU

Solvers to be included:

- [ ] instationary Navier-Stokes: monolithic, implicit fractional theta time-stepping, direct LU
- [ ] instationary Navier-Stokes: monolithic, explicit fractional theta time-stepping, direct LU
- [ ] instationary Navier-Stokes: decoupled, explicit fractional theta time-stepping, direct LU
- [ ] instationary Navier-Stokes: decoupled, implicit fractional Glowinksi theta time-stepping, direct LU
- [ ] instationary Navier-Stokes: decoupled, explicit fractional Glowinksi theta time-stepping, direct LU
- [ ] instationary Navier-Stokes: decoupled, BDF-2 pressure projection scheme, iterative solvers


Applications and demos:
- [x] stationary backward facing step
- [x] stationary Blasius flow problem
- [x] stationary channel flow with various boundary conditions
- [x] stationary Couette flow
- [x] stationary gravity-driven flow
- [x] stationary lid-driven cavity
- [ ] DFG benchmark problem 2D-1
- [x] instationary channel flow
- [x] instationary gravity-driven flow
- [ ] DFG benchmark problem 2D-2
- [ ] DFG benchmark problem 2D-3
- [ ] Taylor-Green vortex
- [ ] two-phase flow solver / breaking dam problem
