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

"""This integrates orbits.

The functions defined here calculate integrated orbits around
Sagittarius A*, tracking both positional and velocity data. There are
additionally functions built to rotate the integrated orbits into the
FK5 coordinate system for comparison to radio data.

"""
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.constants import G
from astropy.coordinates import Galactocentric, FK5, ICRS, Angle

outpath = '../out/'

# galactic center
gc = ICRS(ra=Angle('17h45m40.0409s'), dec=Angle('-29:0:28.118 degrees'))

Mdat = np.genfromtxt(os.path.join(os.path.dirname(__file__),
                                  '../dat/enclosed_mass_distribution.txt'))
Mdist = Angle(Mdat[:, 0], unit=u.arcsec).to(u.rad) * 8.0e3 * u.pc / u.rad
Menc = 10 ** Mdat[:, 1] * u.Msun

h = 500 * u.yr  # timestep

G = G.to((u.pc ** 3) / (u.Msun * u.yr ** 2))


def mass_func(dist):
    """Finds the interpolated central mass of a spherical distribution.

    Calculates the mass contained in a sperical distribution at a given
    radius from Sgr A*.

    Parameters
    ----------
    dist : float
        The distance from Sgr A*, in units of length. If no units are
        given, parsecs are assumed.

    Returns
    -------
    float
        The mass (in solar masses) enclosed by the sphere of radius
        `dist`.

    """
    try:
        if (dist.unit != u.pc):
            try:
                dist = dist.to(u.pc)
            except u.UnitConversionError as e:
                raise e
    except AttributeError as e:
        # Assume units in pc if no units given
        dist = dist * u.pc

    return np.interp(dist, Mdist, Menc) * u.Msun


def potential(dist):
    """Calculates gravitational potential

    Calculates the gravitational potential at a given distance from
    Sgr A*.

    Parameters
    ----------
    dist : float
        The radial distance from Sgr A*, in units of length. If no
        units are given, parsecs are assumed.

    Returns
    -------
    float
        The gravitational potential at the given distance, in units of
        pc^2 / yr^2.

    """
    try:
        if (dist.unit != u.pc):
            try:
                dist = dist.to(u.pc)
            except u.UnitConversionError as e:
                raise e
    except AttributeError as e:
        # Assume units in pc if no units given
        dist = dist * u.pc

    with warnings.catch_warnings(record=True):
        return - G * mass_func(dist) / dist


def mass_grad(dist, dr=0.0005):
    """Automates calculation of the mass gradient.

    Wrapper function to evaluate the derivative of the non-analytic
    mass fuction at a point.

    Parameters
    ----------
    dist : float
        The distance at which to evaluate the gradient of the spherical
        mass function.
    dr : float, optional (default=0.001)
        The step size to use for gradient evaluation. Default size is
        the rough spacing (in pc) of our mass data.

    Returns
    -------
    float
        The estimated rate of change of the mass over the distance.

    """
    sample_dists = np.array([dist.value - dr,
                             dist.value,
                             dist.value + dr]) * u.pc

    with warnings.catch_warnings(record=True):
        grad = np.gradient(mass_func(sample_dists.value).value,
                           sample_dists.value)

    grad[np.isnan(grad)] = 0

    return grad[1] * u.Msun / u.pc


def potential_grad(dist, dr=0.0005):
    """Calculates the gradient of the potential at a point.

    Calculates the approximate gradient of the gravitational potential
    due to Sgr A* at the point indicated.

    Parameters
    ----------
    dist : float
        The distance from Sgr A* at which to calculate the potential.
    dr : float, optional (default=0.001)
        The step size to use for gradient evaluation. Default size is
        the rough spacing (in pc) of our mass data.

    Returns
    -------
    float
        The potential gradient at the given distance in units of
        pc / yr^2

    """
    if (dist == 0):
        return np.inf

    return -1 * (potential(dist) / dist) - G * mass_grad(dist) / dist


def angular_momentum(r1, r2):
    """Calculates the angular momentum per unit mass for given apsides.

    Calculates the angular momentum per unit mass required for a system
    to have the given apsides.

    Parameters
    ----------
    r1 : float
        One of the apsides for the system, with units of length. If no
        units are given, parsecs are assumed.
    r2 : float
        The other apside for the system, with units of length. If no
        units are given, parsecs are assumed.

    Returns
    -------
    float
        Angular momentum per unit mass, in units of pc^2 / yr

    """
    try:
        if (r1.unit != u.pc):
            try:
                r1 = r1.to(u.pc)
            except u.UnitConversionError as e:
                raise e
    except AttributeError as e:
        # Assume units in pc if no units given
        r1 = r1 * u.pc

    try:
        if (r2.unit != u.pc):
            try:
                r2 = r2.to(u.pc)
            except u.UnitConversionError as e:
                raise e
    except AttributeError as e:
        # Assume units in pc if no units given
        r2 = r2 * u.pc

    if (r1 == r2):
        return r1 * np.sqrt(-1 * potential(r1) - G * mass_grad(r1))

    # force the order of apsides such that r1 = periapsis
    if (r1 > r2):
        tmp = r1
        r1 = r2
        r2 = tmp

    return r1 * r2 * np.sqrt(2 * (potential(r2) - potential(r1))
                             / ((r2 ** 2) - (r1 ** 2)))


def orbit(r_per, r_ap, tstep):
    """Generates orbits.

    Takes in a peri/apoapsis and generates an integrated orbit around SgrA*.
    Returns 2 arrays of position and velocity vectors.

    x0 = [pc], v0 = [km/s], tstep = [yr]
    """
    # pylint: disable=E1101
    # sticking to 2D polar for initial integration since z = 0
    # TODO: Check if the issue in the integrated orbit is due to the
    # radial integrator
    r_per *= u.pc
    r_ap *= u.pc
    r_pos = np.array([r_per.value]) * u.pc
    r_vel = np.array([0.]) * u.pc / u.yr

    ang_pos = np.array([0.]) * u.rad

    l_cons = angular_momentum(r_per, r_ap)

    ang_v0 = l_cons / (r_per ** 2) * u.rad
    ang_vel = np.array([ang_v0.value]) * u.rad / u.yr

    while ang_pos[-1] < 2 * np.pi * u.rad:
        # radial portion first
        r_half = r_pos[-1] + 0.5 * h * r_vel[-1]  # first drift
        r_vel_new = r_vel[-1] + h * ((l_cons ** 2) / (r_half ** 3)
                                     - potential_grad(r_half))  # kick
        r_new = r_half + 0.5 * h * r_vel_new  # second drift
        r_pos = np.append(r_pos.value, r_new.value) * u.pc
        r_vel = np.append(r_vel.value, r_vel_new.value) * u.pc / u.yr

        # then angular
        ang_half = ang_pos[-1] + 0.5 * h * ang_vel[-1]
        ang_vel_new = l_cons * u.rad / (r_new ** 2)
        ang_new = ang_half + 0.5 * h * ang_vel_new
        ang_pos = np.append(ang_pos.value, ang_new.value) * u.rad
        ang_vel = np.append(ang_vel.value, ang_vel_new.value) * u.rad / u.yr

    return r_pos, r_vel, ang_pos, ang_vel


def polar_to_cartesian(r_pos, r_vel, ang_pos, ang_vel):
    pos = np.array([r_pos * np.cos(ang_pos),
                    r_pos * np.sin(ang_pos),
                    [0.] * len(r_pos)])

    vel = np.array([r_vel * np.cos(ang_pos)
                    - r_pos * ang_vel * np.sin(ang_pos) / u.rad,
                    r_vel * np.sin(ang_pos)
                    + r_pos * ang_vel * np.cos(ang_pos) / u.rad,
                    [0.] * len(r_vel)])

    return pos, vel


def rot_mat(aop, loan, inc):
    """
    Returns the rotation matrix for going from the orbit plane to the sky plane

    aop = [rad], loan = [rad], inc = [rad]
    """

    T = np.array([[np.cos(loan) * np.cos(aop) - np.sin(loan) * np.sin(aop)
                  * np.cos(inc),
                  - np.sin(loan) * np.cos(aop) - np.cos(loan) * np.sin(aop)
                  * np.cos(inc),
                  np.sin(aop) * np.sin(inc)],

                  [np.cos(loan) * np.sin(aop) + np.sin(loan) * np.cos(aop)
                  * np.cos(inc),
                  - np.sin(loan) * np.sin(aop) + np.cos(loan) * np.cos(aop)
                  * np.cos(inc),
                  - np.cos(aop) * np.sin(inc)],

                  [np.sin(loan) * np.sin(inc),
                   np.cos(loan) * np.sin(inc),
                   np.cos(inc)]])

    return T


def sky(p):
    """
    Takes in coordinates ~p in degrees~ and spits out f1 -- the constraint that
    the orbit must be elliptical and SgrA* lies at the focus
    """
    # pylint: disable=E1101
    (aop, loan, inc, r_per, r_ap) = p

    # convert from degrees to radians
    aop = aop * np.pi / 180.
    loan = loan * np.pi / 180.
    inc = inc * np.pi / 180.

    rot = rot_mat(aop, loan, inc)

    orb_r, orb_v = polar_to_cartesian(orbit(r_per, r_ap, h))

    # Rotate reference frame
    # We can use the transpose of the rotation matrix instead of the inverse
    # because it's Hermitian
    rot_R = np.matmul(rot.T, orb_r.T)
    rot_V = np.matmul(rot.T, orb_v.T)

    # take rotated reference frame to be galactocentric coordinates
    # then transform to FK5 (matching our data)
    c = Galactocentric(x=rot_R[0] * u.pc,
                       y=rot_R[1] * u.pc,
                       z=rot_R[2] * u.pc,
                       v_x=rot_V[0] * u.km / u.s,
                       v_y=rot_V[1] * u.km / u.s,
                       v_z=rot_V[2] * u.km / u.s,
                       galcen_distance=8. * u.kpc,
                       galcen_coord=gc).transform_to(FK5)

    return c


def model(p):
    # wrapper function to convert the coordinates to a numpy array of datapts
    c = sky(p)
    return np.array([c.ra.rad, c.dec.rad, c.radial_velocity.value]).T


def plot_func(p):
    orb_r, orb_v = orbit(p[-2], p[-1], h)
    sky_xyv = model(p)

    plt.figure(1)
    plt.plot(sky_xyv[:, 0], sky_xyv[:, 1], 'k-', label='Gas core')
#    plt.plot([0], [0], 'g*', label='Sgr A*')

    plt.grid(True)
    plt.axes().set_aspect('equal', 'datalim')
    plt.title('Sky Plane')  # . p = {}'.format(p))
    plt.xlabel('Offset (rad)')
    plt.ylabel('Offset (rad)')
    plt.savefig(outpath + 'skyplane.pdf', bbox_inches='tight')
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(orb_r[:, 0], orb_r[:, 1], 'k-', label='Gas core')
    plt.plot([0], [0], 'g*', label='Sgr A*')

    plt.grid(True)
    plt.axes().set_aspect('equal', 'datalim')
    plt.title('Orbit Plane')  # . p = {}'.format(p))
    plt.xlabel('Offset (pc)')
    plt.ylabel('Offset (pc)')
    plt.savefig(outpath + 'orbitplane.pdf', bbox_inches='tight')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    r_pos, r_vel, ang_pos, ang_vel = orbit(1.5, 3., 100 * u.yr)
    pos, vel = polar_to_cartesian(r_pos, r_vel, ang_pos, ang_vel)
    plt.plot(pos[0], pos[1])
    plt.grid()
    plt.show()
    plt.plot(vel[0], vel[1])
    plt.grid()
    plt.show()
    print(np.max(r_pos), np.min(r_pos), np.max(np.abs(r_vel)))
