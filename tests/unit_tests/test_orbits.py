#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is MCOrbit, an MCMC application for fitting orbits near Sgr A*.
# Copyright (C) 2017-2018  J. Andrew Casey-Clyde
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Orbit tests.

Collection of tests to ensure that the orbit generation functions are
working as expected.

"""
import numpy as np
import astropy.units as u

from mcorbit import orbits


class TestMass(object):
    def test_inf(self):
        assert orbits.mass(np.inf * u.pc) == np.inf

    def test_data_points(self):
        mass_r = [orbits.mass(r).value for r in orbits.Mdist]
        np.testing.assert_array_almost_equal(mass_r, np.power(10, orbits.Menc),
                                             decimal=3)


class TestPotential(object):
    def test_inf(self):
        assert orbits.potential(np.inf * u.pc) == 0

    def test_0(self):
        assert orbits.potential(0 * u.pc) == -np.inf

    def test_units(self):
        assert orbits.potential(1 * u.pc).unit == (u.pc ** 2) / (u.yr ** 2)


class TestGradient(object):
    def test_inf(self):
        assert orbits.potential_grad(np.inf * u.pc) == 0

    def test_0(self):
        assert orbits.potential_grad(0 * u.pc) == np.inf

    def test_units(self):
        assert orbits.potential_grad(1 * u.pc).unit == u.pc / (u.yr ** 2)

    def test_condition(self):
        r1 = 1. + np.random.rand()
        r2 = r1 + 4. * np.random.rand()
        assert (((r1 ** 3) * orbits.potential_grad(r1 * u.pc))
                <= ((r2 ** 3) * orbits.potential_grad(r2 * u.pc)))


class TestAngularMomentum(object):
    def setup_method(self):
        self.r1 = 1. + np.random.rand()
        self.r2 = self.r1 + 4. * np.random.rand()
        self.r_circ = 1. + 4. * np.random.rand()

        self.l_cons = orbits.angular_momentum(self.r1, self.r2).value
        self.l_circ = orbits.angular_momentum(self.r_circ, self.r_circ).value

    def test_lower_bound(self):
        lower_bound = ((self.r1 ** 3) * orbits.potential_grad(self.r1)).value
        assert (self.l_cons ** 2) >= lower_bound

    def test_upper_bound(self):
        upper_bound = ((self.r2 ** 3) * orbits.potential_grad(self.r2)).value
        assert (self.l_cons ** 2) <= upper_bound

    def test_circle(self):
        a_l = (self.l_circ ** 2) / (self.r_circ ** 3)
        a_g = orbits.potential_grad(self.r_circ).value
        np.testing.assert_approx_equal(a_l, a_g, significant=3)


class TestCircularOrbit(object):
    def setup_method(self):
        self.r_circ = 1. + 4. * np.random.rand()

        (self.r_pos,
         self.r_vel,
         self.ang_pos,
         self.ang_vel) = orbits.orbit(self.r_circ, self.r_circ, 500 * u.yr)

    def test_radial_maxima(self):
        np.testing.assert_approx_equal(np.max(self.r_pos.value), self.r_circ,
                                       significant=3)

    def test_radial_minima(self):
        np.testing.assert_approx_equal(np.min(self.r_pos.value), self.r_circ,
                                       significant=3)


class TestEllipticalOrbit(object):
    def setup_method(self):
        self.r1 = 1. + 1.5 * np.random.rand()
        self.r2 = self.r1 + 1.5 * np.random.rand()

        self.l_cons = orbits.angular_momentum(self.r1, self.r2)

        (self.r_pos,
         self.r_vel,
         self.ang_pos,
         self.ang_vel) = orbits.orbit(self.r1, self.r2, 500 * u.yr)

    def test_ellipse_raidal_minima(self):
        np.testing.assert_approx_equal(np.min(self.r_pos.value),
                                       self.r1,
                                       significant=3)
        # assert all(r_pos.value >= self.r1) and all(r_pos.value <= self.r2)

    def test_ellipse_radial_maxima(self):
        np.testing.assert_approx_equal(np.max(self.r_pos.value),
                                       self.r2,
                                       significant=3)

    def test_acceleration_periapsis(self):
        assert (orbits.centrifugal_acceleration(self.r1, self.l_cons).value
                - orbits.potential_grad(self.r1).value) >= 0

    def test_acceleration_apoapsis(self):
        assert (orbits.centrifugal_acceleration(self.r2, self.l_cons).value
                - orbits.potential_grad(self.r2).value) <= 0
