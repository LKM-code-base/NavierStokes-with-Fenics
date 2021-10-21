#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from auxiliary_classes import AngularVelocityVector, FunctionTime, EquationCoefficientHandler


class AngularVelocityFunction01(FunctionTime):
    def __init__(self):
        super().__init__(1)

    def value(self):
        return 1.0


class AngularVelocityFunction02(FunctionTime):
    def __init__(self):
        super().__init__(1)

    def value(self):
        return self._current_time

    def derivative(self):
        return 1.0


class AngularVelocityFunction03(FunctionTime):
    def __init__(self):
        super().__init__(3)

    def value(self):
        return (1.0, 1.0, 1.0)

    def derivative(self):
        return (0.0, 0.0, 0.0)


def test_angular_velocity_vector():
    omega01 = AngularVelocityVector(2)
    omega01.set_angular_velocity_function(AngularVelocityFunction01())
    omega01.set_time(1.0)
    _ = omega01.derivative
    _ = omega01.space_dim
    _ = omega01.value

    omega02 = AngularVelocityVector(2)
    omega02.set_angular_velocity_function(AngularVelocityFunction02())
    omega02.set_time(1.0)
    _ = omega02.derivative
    _ = omega02.space_dim
    _ = omega02.value

    omega03 = AngularVelocityVector(3, function=AngularVelocityFunction03())
    omega03.set_time(1.0)
    _ = omega03.derivative
    _ = omega03.space_dim
    _ = omega03.value


def test_equation_coefficients():
    eq01 = EquationCoefficientHandler(Reynolds=50.0)
    print(eq01)
    eq01.Fr = 10.0
    _ = eq01.equation_coefficients
    print(eq01)
    eq01.Ek = 25.0
    _ = eq01.equation_coefficients
    print(eq01)
    _ = eq01.Fr
    _ = eq01.Ek
    _ = eq01.Re

    eq02 = EquationCoefficientHandler()
    print(eq02)
    eq02.Ro = 1.0
    _ = eq02.equation_coefficients
    print(eq02)
    eq02.Ek = 25.0
    _ = eq02.equation_coefficients
    print(eq02)
    eq02.Fr = 10.0
    _ = eq02.equation_coefficients
    print(eq02)
    _ = eq01.Fr
    _ = eq01.Ek
    _ = eq01.Ro
    eq02.clear()

    eq03 = EquationCoefficientHandler()
    print(eq02)
    eq03.Ek = 1.0
    _ = eq03.equation_coefficients
    print(eq03)
    eq03.Re = 25.0
    _ = eq03.equation_coefficients
    print(eq03)
    eq03.modify_dimensionless_number("Re", 10.0)
    print(eq03.get_file_suffix())


if __name__ == "__main__":
    test_angular_velocity_vector()
    test_equation_coefficients()
