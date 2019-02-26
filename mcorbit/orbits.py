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
# pylint: disable=E0611, E1101
# pylint: disable=W0621, W0611
import os
import warnings

import numpy as np

from scipy.interpolate import interp1d

import astropy.units as u
from astropy.constants import G
from astropy.coordinates import Galactocentric, FK5, ICRS, Angle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# =============================================================================
# =============================================================================
# # CONSTANTS
# =============================================================================
# =============================================================================

# =============================================================================
# User definable constants
# =============================================================================
TSTEP = 500 * u.yr  # timestep
FIGSIZE = (12, 12)

# =============================================================================
# Referenced constants
# =============================================================================
GAL_CENTER = ICRS(ra=Angle('17h45m40.0409s'),
                  dec=Angle('-29:0:28.118 degrees'))

G = G.to((u.pc ** 3) / (u.Msun * u.yr ** 2))

D_SGR_A = 8. * u.kpc

# =============================================================================
# Default data
# =============================================================================
M_DAT = np.genfromtxt(os.path.join(os.path.dirname(__file__),
                                   '../dat/enclosed_mass_distribution.txt'))
M_DIST = Angle(M_DAT[:, 0], unit=u.arcsec).to(u.rad) * D_SGR_A.to(u.pc) / u.rad
M_ENC = M_DAT[:, 1]  # 10 ** Mdat[:, 1] * u.Msun
M_GRAD = np.gradient(M_ENC, M_DIST.value) * u.Msun / u.pc

# uncomment to use point mass equal to mass enclosed at ~1pc
#M_ENC = len(M_ENC) * [M_ENC[175]]  # 1pc away from Sgr A*
#M_GRAD = np.zeros(len(M_ENC)) * u.Msun / u.pc

# uncomment to use point mass equal to mass enclosed at ~5pc
#M_ENC = len(M_ENC) * [M_ENC[240]]  # 5pc away from Sgr A*
#M_GRAD = np.zeros(len(M_ENC)) * u.Msun / u.pc

M_ENC_INTERP = interp1d(M_DIST.value, M_ENC,
                        kind='cubic', fill_value='extrapolate')
M_GRAD_INTERP = interp1d(M_DIST.value, M_GRAD,
                         kind='cubic', fill_value='extrapolate')


# =============================================================================
# =============================================================================
# # Functions
# =============================================================================
# =============================================================================

# =============================================================================
# Orbit functions
# =============================================================================
def mass(dist, interp=M_ENC_INTERP):
    """Finds the interpolated central mass of a spherical distribution.

    Calculates the mass contained in a sperical distribution at a given
    radius from Sgr A*.

    Parameters
    ----------
    dist : float
        The distance from Sgr A*, in units of length. If no units are
        given, parsecs are assumed.
    interp : func
        The mass function to use. Should accept a distance from the
        source in pc, and return the mass enclosed, in log(Msun).
        Defaults to interpolations of the mass distribution around
        Sgr A* found by Feldmeier-Krause et al. (2017).

    Returns
    -------
    float
        The mass (in solar masses) enclosed by the sphere of radius
        `dist`.

    """
    try:
        if dist.unit != u.pc:
            try:
                dist = dist.to(u.pc)
            except u.UnitConversionError as err:
                raise err
    except AttributeError as err:
        # Assume units in pc if no units given
        dist = dist * u.pc

    if dist == np.inf * u.pc:
        return np.inf * u.Msun

    mass_enc = interp(dist)
    return np.power(10, mass_enc) * u.Msun


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
        if dist.unit != u.pc:
            try:
                dist = dist.to(u.pc)
            except u.UnitConversionError as err:
                raise err
    except AttributeError as err:
        # Assume units in pc if no units given
        dist = dist * u.pc

    with warnings.catch_warnings(record=True):
        if dist == np.inf:
            return 0. * (u.pc ** 2) / (u.yr ** 2)
        return - G * mass(dist) / dist


@np.vectorize
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

    Todo
    ----
    - Figure out what's wrong with this calculation (too small initially)

    """
    try:
        if r1.unit != u.pc:
            try:
                r1 = r1.to(u.pc)
            except u.UnitConversionError as e:
                raise e
    except AttributeError as e:
        # Assume units in pc if no units given
        r1 = r1 * u.pc

    try:
        if r2.unit != u.pc:
            try:
                r2 = r2.to(u.pc)
            except u.UnitConversionError as e:
                raise e
    except AttributeError as e:
        # Assume units in pc if no units given
        r2 = r2 * u.pc

    if r1 == r2:
        return (r1 * np.sqrt(-1 * potential(r1) - G * mass_grad(r1))).value

    E = ((((r2 ** 2) * potential(r2)) - ((r1 ** 2) * potential(r1)))
         / ((r2 ** 2) - (r1 ** 2)))

#    return np.sqrt(2 * (((r2 ** -2) - (r1 ** -2)) ** -1)
#                   * (potential(r2) - potential(r1)))
    return (r1 * np.sqrt(2 * (E - potential(r1)))).value


@np.vectorize
def V_eff(r, l):
    """The effective potential.

    Calculates the effective potential at a point, given the angular
    angular momentum per unit mass.

    Parameters
    ----------
    r : float
        The point at which to calculate the effective potential.
    l : float
        The angular momentum per unit mass.

    Returns
    -------
    float
        The effective potential for a particle of unit mass with
        angular momentum `l` at point `r`

    """
    if r == 0. or r ** 2 == 0:
        return np.inf
    V_l = (l ** 2) / (2 * (r ** 2))
    return V_l + potential(r).value


def mass_grad(dist, interp=M_GRAD_INTERP):
    """Automates calculation of the mass gradient.

    Wrapper function to evaluate the derivative of the non-analytic
    mass fuction at a point.

    Parameters
    ----------
    dist : float
        The distance at which to evaluate the gradient of the spherical
        mass function.
    interp : func
        The mass gradient function to use. Should accept a distance
        from the source in pc, and return the mass enclosed, in
        log(Msun). Defaults to interpolations of the gradient of the
        mass distribution around Sgr A* found by
        Feldmeier-Krause et al. (2017).

    Returns
    -------
    float
        The estimated rate of change of the mass over the distance.

    """
    try:
        if dist.unit != u.pc:
            try:
                dist = dist.to(u.pc)
            except u.UnitConversionError as err:
                raise err
    except AttributeError as err:
        # Assume units in pc if no units given
        dist = dist * u.pc

    if dist == np.inf * u.pc:
        return 0. * u.Msun / u.pc

    m_grad = (interp(dist) * mass(dist) * np.log(10)).value
    return m_grad * u.Msun / u.pc


def potential_grad(dist):
    """Calculates the gradient of the potential at a point.

    Calculates the approximate gradient of the gravitational potential
    due to Sgr A* at the point indicated.

    Parameters
    ----------
    dist : float
        The distance from Sgr A* at which to calculate the potential.

    Returns
    -------
    float
        The potential gradient at the given distance in units of
        pc / yr^2

    Todo
    ----
    - Figure out is potential gradient is too high

    """
    if dist == 0:
        return np.inf * u.pc / (u.yr ** 2)

    return (-1. / dist) * (potential(dist) + G * mass_grad(dist))


@np.vectorize
def V_eff_grad(r, l):
    """The gradient of the effective potential.

    Calculates the gradient of the effective potential at a point,
    given the angular angular momentum per unit mass.

    Parameters
    ----------
    r : float
        The point at which to calculate the effective potential
        gradient.
    l : float
        The angular momentum per unit mass.

    Returns
    -------
    float
        The effective potential gradient for a particle of unit mass
        with angular momentum `l` at point `r`.

    """
    if r == 0. or r ** 2 == 0:
        return np.inf
    return (- (l ** 2) / (r ** 3)) + potential_grad(r).value


def orbit(r0, l_cons):
    """Generates orbits.

    Orbit generator that integrates orbits around Sgr A*. Assumes a
    spherical potential and that orbits are bound.

    Parameters
    ----------
    r0 : float
        The initial position. Specifically, the radial distance between
        Sgr A* and an (arbitrary) apside.
    l_cons : float
        The angular momentum per unit mass associated with the orbit.

    Returns
    -------
    r_pos : :obj:`numpy.ndarray`
        The radial position at each timestep. Covers one angular
        period.
    r_vel : :obj:`numpy.ndarray`
        The radial velocity at each timestep. Covers one angular
        period.
    ang_pos : :obj:`numpy.ndarray`
        The angular position at each timestep. Covers one angular
        period.
    ang_vel : :obj:`numpy.ndarray`
        The angular velocity at each timestep. Covers one angular
        period.

    """
    # keeping track of initial particle energy
    # necessary for some endpoint corrections
    E = V_eff(r0, l_cons)

    # sticking to 2D polar for initial integration since z = 0
    r0 *= u.pc
    r_pos = np.array([r0.value]) * u.pc
    r_vel = np.array([0.]) * u.pc / u.yr

    ang_pos = np.array([0.]) * u.rad

    l_cons *= (u.pc ** 2) / u.yr

    ang_v0 = l_cons / (r0 ** 2) * u.rad
    ang_vel = np.array([ang_v0.value]) * u.rad / u.yr

    while ang_pos[-1] < 2 * np.pi * u.rad:
        # radial portion first
        # first drift
        r_half = r_pos[-1] + 0.5 * TSTEP * r_vel[-1]

        # kick
        if E <= V_eff(r_half.value, l_cons.value):
            r_vel_new = TSTEP * ((l_cons ** 2) / (r_half ** 3)
                                 - potential_grad(r_half))
        else:
            r_vel_new = r_vel[-1] + TSTEP * ((l_cons ** 2) / (r_half ** 3)
                                             - potential_grad(r_half))

        # second drift
        r_new = r_half + 0.5 * TSTEP * r_vel_new
        r_pos = np.append(r_pos.value, r_new.value) * u.pc
        r_vel = np.append(r_vel.value, r_vel_new.value) * u.pc / u.yr

        # then angular
        ang_half = ang_pos[-1] + 0.5 * TSTEP * ang_vel[-1]
        ang_vel_new = l_cons * u.rad / (r_new ** 2)
        ang_new = ang_half + 0.5 * TSTEP * ang_vel_new
        ang_pos = np.append(ang_pos.value, ang_new.value) * u.rad
        ang_vel = np.append(ang_vel.value, ang_vel_new.value) * u.rad / u.yr

    return r_pos, r_vel, ang_pos, ang_vel


# =============================================================================
# Coordinate functions
# =============================================================================
def polar_to_cartesian(r_pos, r_vel, ang_pos, ang_vel):
    """Converts polar positions and velocities to cartesian.

    Polar to cartesian coordinate transform, where the polar
    coordinates given are assumed to be in the z = 0 plane of the
    cartesian system. Converts both positions and velocities.

    Parameters
    ----------
    r_pos : array
        Vector of radial positions.
    r_vel : array
        Vector of radial velocities. Should be the same shape as r_pos.
    ang_pos : array
        Vector of angular positions. Should be the same shape as r_pos.
    ang_vel : array
        Vector of angular velocities. Should be the same shape as
        r_pos.

    Returns
    -------
    pos : array, shape (3, len(r_pos))
        Matrix of 3D cartisian coordinates of original polar
        coordinates.
    vel : array, shape (3, len(r_vel))
        Matrix of 3D cartesian coordinates of original velocity
        coordinates.

    """
    pos = np.array([r_pos * np.cos(ang_pos),
                    r_pos * np.sin(ang_pos),
                    [0.] * len(r_pos)])

    vel = np.array([(r_vel * np.cos(ang_pos)
                    - r_pos * ang_vel * np.sin(ang_pos)
                    / u.rad).to(u.km / u.s),
                    (r_vel * np.sin(ang_pos)
                    + r_pos * ang_vel * np.cos(ang_pos)
                    / u.rad).to(u.km / u.s),
                    [0.] * len(r_vel)])

    return pos, vel


def orbit_rotator(pos, vel, aop, loan, inc):
    """Rotates orbits out of orbital plane.

    Rotates orbits out of orbital plane and into galactocentric
    coordinates.

    Parameters
    ----------
    pos : array, float, shape (3, n)
        Cartesian positions of `n` points in orbit
    vel : array, float, shape (3, n)
        Velocity of test particle at each point, where points
        correspond to positions in `pos`. Cartesian coordinates
    aop : float
        The argument of pericenter. The angle from the ascending node
        to pericenter.
    loan : float
        Longitude of the ascending node. The angle between the x axis
        and the axis at which the orbit intersects the xy-plane.
    inc : float
        Inclination. The inclination of vector normal to the orbital
        plane relative to the z axis.

    Returns
    -------
    pos : float
        The 3D orbit in galactocentric coordinates.
    vel : float
        The 3D velocities of the orbit in galactocentric coordinates

    """
    aop = np.radians(aop)
    loan = np.radians(loan)
    inc = np.radians(inc)

    c_aop, s_aop = np.cos(aop), np.sin(aop)
    c_loan, s_loan = np.cos(loan), np.sin(loan)
    c_inc, s_inc = np.cos(inc), np.sin(inc)

    r_aop = np.array([[c_aop, -s_aop, 0],
                      [s_aop, c_aop, 0],
                      [0, 0, 1]])
    r_inc = np.array([[1, 0, 0],
                      [0, c_inc, -s_inc],
                      [0, s_inc, c_inc]])
    r_loan = np.array([[c_loan, -s_loan, 0],
                       [s_loan, c_loan, 0],
                       [0, 0, 1]])

    T_mat = r_aop @ r_inc @ r_loan
    pos, vel = T_mat.T @ pos, T_mat.T @ vel
    return pos, vel


def sky_coords(pos, vel):
    """Transforms from galactocentric to sky coordinates.

    Transforms an orbit and it's associated orbital velocities from
    galactocentric coordinates to FK5 coordinates.

    Parameters
    ----------
    pos : array, float, shape (3, n)
        Cartesian positions of `n` points in orbit
    vel : array, float, shape (3, n)
        Velocity of test particle at each point, where points
        correspond to positions in `pos`. Cartesian coordinates

    """
    # take rotated reference frame to be galactocentric coordinates
    # then transform to FK5 (matching our data)
    c = Galactocentric(x=pos[0] * u.pc,
                       y=pos[1] * u.pc,
                       z=pos[2] * u.pc,
                       v_x=vel[0] * u.km / u.s,
                       v_y=vel[1] * u.km / u.s,
                       v_z=vel[2] * u.km / u.s,
                       galcen_distance=D_SGR_A,
                       galcen_coord=GAL_CENTER).transform_to(FK5)

    return c


# =============================================================================
# The model function
# =============================================================================
def model(theta, coords=False):
    """Model generator.

    Generates model orbits around Sgr A*, as seen from the FK5
    coordinate system.

    Parameters
    ----------
    theta : (aop, loan, inc, r_per, r_ap)

    """
    aop, loan, inc, r0, l_cons = theta
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            pos, vel = polar_to_cartesian(*orbit(r0, l_cons))
        except Warning:
            raise ValueError("orbits params: {0}, {1}".format(theta[-2], theta[-1]))
    pos, vel = orbit_rotator(pos, vel, aop, loan, inc)
    c = sky_coords(pos, vel)
    if coords:
        return c
    return np.array([c.ra.rad, c.dec.rad,
                     c.radial_velocity.value]).T


# =============================================================================
# Plotting functions
# =============================================================================
def plot_orbit(r0, l_cons):
    pos, _ = polar_to_cartesian(*orbit(r0, l_cons))

    plt.figure(figsize=FIGSIZE)
    plt.plot(pos[0, :], pos[1, :], 'b-', label="Gas Core")
    plt.plot([0], [0], 'ko', label='Sgr A*')

    plt.grid(True)
    plt.axes().set_aspect('equal', 'datalim')
    plt.xlabel('Offset [pc]')
    plt.ylabel('Offset [pc]')
    plt.legend()
    plt.show()


def plot_mass(r1, r2):
    """Plot the mass.

    Plot the central mass function from `r1` to `r2`.

    Parameters
    ----------
    r1 : float
        The minimum distance of the central mass function to plot.
    r2 : float
        The maximum distance of the central mass function to plot.

    """
    r_pos = np.linspace(r1, r2, num=100)
    mass_r = [mass(r).value for r in r_pos]

    plt.figure(figsize=FIGSIZE)
    plt.plot(r_pos, mass_r)
    plt.title("$M$ vs. $r$")
    plt.xlabel("$r [pc]$")
    plt.ylabel("$M(r) [M_{\\odot}]$")
    plt.grid()
    save_path = os.path.join(os.path.dirname(__file__), '..', 'out',
                             'mass_r.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    return mass_r


def plot_mass_grad(r1, r2):
    """Plot the mass gradient.

    Plot the gradient of the central mass function from `r1` to `r2`.

    Parameters
    ----------
    r1 : float
        The minimum distance of the central mass function gradient to
        plot.
    r2 : float
        The maximum distance of the central mass function gradient to
        plot.

    """
    r_pos = np.linspace(r1, r2, num=100)
    mass_grad_r = [mass_grad(r).value for r in r_pos]

    plt.figure(figsize=FIGSIZE)
    plt.plot(r_pos, mass_grad_r)
    plt.title("$dM/dr$ vs. $r$")
    plt.xlabel("$r [\\mathrm{pc}]$")
    plt.ylabel("$dM/dr [M_{\\odot} / \\mathrm{pc}]$")
    plt.grid()
    save_path = os.path.join(os.path.dirname(__file__), '..', 'out',
                             'mass_grad_r.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    return mass_grad_r


def plot_potential(r1, r2):
    """Plot the potential.

    Plot the potential from `r1` to `r2`.

    Parameters
    ----------
    r1 : float
        The minimum distance of the potential to plot.
    r2 : float
        The maximum distance of the potnetial to plot.

    """
    r_pos = np.linspace(r1, r2, num=100)
    potential_r = [potential(r).value for r in r_pos]

    plt.figure(figsize=FIGSIZE)
    plt.plot(r_pos, potential_r)
    plt.title("$\\Phi(r)$ vs. $r$")
    plt.xlabel("$r [\\mathrm{pc}]$")
    plt.ylabel("$\\Phi [\\mathrm{pc}^{2} / \\mathrm{yr}^{2}]$")
    plt.grid()
    save_path = os.path.join(os.path.dirname(__file__), '..', 'out',
                             'potential_r.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    return potential_r


def plot_potential_grad(r1, r2):
    """Plot the potential gradient.

    Plot the potential gradient from `r1` to `r2`.

    Parameters
    ----------
    r1 : float
        The minimum distance of the potential gradient to plot.
    r2 : float
        The maximum distance of the potnetial gradient to plot.

    """
    r_pos = np.linspace(r1, r2, num=100)
    potential_grad_r = [potential_grad(r).value for r in r_pos]

    plt.figure(figsize=FIGSIZE)
    plt.plot(r_pos, potential_grad_r)
    plt.title("$\\nabla\\Phi(r)$ vs. $r$")
    plt.xlabel("$r [\\mathrm{pc}]$")
    plt.ylabel("$\\nabla\\Phi [\\mathrm{pc} / \\mathrm{yr}^{2}]$")
    plt.grid()
    save_path = os.path.join(os.path.dirname(__file__), '..', 'out',
                             'potential_grad_r.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    return potential_grad_r


def plot_V_eff(r1, r2, l_cons, post=None):
    """Plot the effective potential.

    Plot the effective potential for a particle whose apsides are at
    either end of the radial range.

    Parameters
    ----------
    r1 : float
        The minimum distance of the potential to plot. Periapsis for
        test particle.
    r2 : float
        The maximum distance of the potnetial to plot. Apoapsis for
        test particle.

    """
    r_pos = np.linspace(r1, r2, num=100)
    V = V_eff(r_pos, l_cons)

    title = "$V_{\\mathrm{eff}}(r)$ vs. $r$"
    if post is not None:
        title += ", " + post

    plt.figure(figsize=FIGSIZE)
    plt.plot(r_pos, V)
    plt.title(title)
    plt.xlabel("$r [\\mathrm{pc}]$")
    plt.ylabel("$V_{\\mathrm{eff}} [\\mathrm{pc}^{2} / \\mathrm{yr}^{2}]$")
    plt.grid()
    save_path = os.path.join(os.path.dirname(__file__), '..', 'out',
                             'V_eff_r{0}.pdf'.format("_" + post if post
                                                     is not None else ""))
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    return V


def plot_V_eff_grad(r1, r2, l_cons, post=None):
    """Plot the effective potential.

    Plot the effective potential for a particle whose apsides are at
    either end of the radial range.

    Parameters
    ----------
    r1 : float
        The minimum distance of the potential to plot. Periapsis for
        test particle.
    r2 : float
        The maximum distance of the potnetial to plot. Apoapsis for
        test particle.

    """
    r_pos = np.linspace(r1, r2, num=100)
    V_grad = V_eff_grad(r_pos, l_cons)

    title = "$\\nabla V_{\\mathrm{eff}}(r)$ vs. $r$"
    if post is not None:
        title += ", " + post

    plt.figure(figsize=FIGSIZE)
    plt.plot(r_pos, V_grad)
    plt.title(title)
    plt.xlabel("$r [\\mathrm{pc}]$")
    plt.ylabel("$\\nabla V_{\\mathrm{eff}} "
               "[\\mathrm{pc} / \\mathrm{yr}^{2}]$")
    plt.grid()
    save_path = os.path.join(os.path.dirname(__file__), '..', 'out',
                             'V_eff_grad{0}.pdf'.format("_" + post if post
                                                        is not None else ""))
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    return V_grad


def plot_acceleration(r1, r2, l_cons):
    """Plot the acceleration.

    Plot the acceleration from `r1` to `r2`.

    Parameters
    ----------
    r1 : float
        The minimum distance of the potential to plot.
    r2 : float
        The maximum distance of the potnetial to plot.

    """
    r_pos = np.linspace(r1, r2, num=100)
    potential_grad_r = [potential_grad(r).value for r in r_pos]

    l_cons = l_cons * (u.pc ** 2) / u.yr
    a_l = ((l_cons ** 2) / (r_pos ** 3)).value

    a = a_l - potential_grad_r

    plt.figure(figsize=FIGSIZE)
    plt.plot(r_pos, a)
    plt.title("$a(r)$ vs. $r$")
    plt.xlabel("$r [\\mathrm{pc}]$")
    plt.ylabel("$a [\\mathrm{pc} / \\mathrm{yr}^{2}]$")
    plt.grid()
    plt.show()

    return a


def plot_velocity(r1, r2):
    """Plot the potential.

    Plot the potential from `r1` to `r2`.

    Parameters
    ----------
    r1 : float
        The minimum distance of the potential to plot.
    r2 : float
        The maximum distance of the potnetial to plot.

    """
    _, r_vel, ang_pos, ang_vel = orbit(r1, r2)

    plt.figure(figsize=FIGSIZE)
    plt.plot(ang_pos, r_vel)
    plt.title("$v_{r}$ vs. $\\theta$")
    plt.xlabel("$\\theta [\\mathrm{rad}]$")
    plt.ylabel("$v_{r} [\\mathrm{pc} / \\mathrm{yr}]$")
    plt.grid()
    plt.show()

    plt.figure(figsize=FIGSIZE)
    plt.plot(ang_pos, ang_vel)
    plt.title("$v_{\\theta}$ vs. $\\theta$")
    plt.xlabel("$\\theta [\\mathrm{rad}]$")
    plt.ylabel("$v_{\\theta} [\\mathrm{rad} / \\mathrm{yr}]$")
    plt.grid()
    plt.show()

    return r_vel, ang_vel


# =============================================================================
# Main function
# =============================================================================
def main():
    # set initial variables
#    r1 = .5
#    r2 = 2.
#
#    from scipy.optimize import brentq
#    l_min = (r1 * r2 * np.sqrt((2 * (potential(r2) - potential(r1)))
#                               / ((r2 ** 2) - (r1 ** 2)))).value
#    rmax = brentq(V_eff_grad, 8., 9., args=(l_min))

    # set up gridspace of apsides
#    rr1 = np.linspace(r1, rmax, num=n_pts)
#    rr2 = np.linspace(r1, rmax, num=n_pts)
#    rr1, rr2 = np.meshgrid(rr1, rr2, indexing='ij')

#    r_max_0 = brentq(V_eff_grad, 8., 10., args=(0.))
#    l_max = (r2 * rmax * np.sqrt((2 * (potential(rmax) - potential(r2)))
#                                 / ((rmax ** 2) - (r2 ** 2)))).value
#
#    r_max_min = brentq(V_eff_grad, 8., 10., args=(l_min))
#    r_max_max = brentq(V_eff_grad, 8., 10., args=(l_max))

#    V_0 = V_eff(r_max_0, 0.)
#    V_min = V_eff(r_max_min, l_min)
#    V_max = V_eff(r_max_max, l_max)
#
#    print("r_0 = {0}, V_0 = {1}, l = 0".format(r_max_0, V_0))
#    print("r_min = {0}, V_min = {1}, l = {2}".format(r_max_min, V_min, l_min))
#    print("r_max = {0}, V_max = {1}, l = {2}".format(r_max_max, V_max, l_max))

#    rtest = rmax
#    rmidtest = r2
#    rrtest = np.linspace(rmidtest, rtest, num=n_pts)
#    l_test = [(rmidtest * r * np.sqrt((2 * (potential(r)
#              - potential(rmidtest))) / ((r ** 2) - (rmidtest ** 2)))).value
#              for r in rrtest]

#    plt.figure(figsize=FIGSIZE)
#    plt.plot(rrtest, l_test)
#    plt.grid()
#    plt.show()

#    grad = np.gradient(l_test, rrtest)
#    interp = interp1d(rrtest[2:], grad[2:], kind='cubic',
#                      fill_value='extrapolate')
#    print(brentq(interp, 6., rtest))

#    rtest = rmax
#    rmidtest = 4.
#    rrtest = np.linspace(rmidtest, rtest, num=n_pts)
#    l_test = [(rmidtest * r * np.sqrt((2 * (potential(r)
#              - potential(rmidtest))) / ((r ** 2) - (rmidtest ** 2)))).value
#              for r in rrtest]

#    plt.figure(figsize=FIGSIZE)
#    plt.plot(rrtest, l_test)
#    plt.grid()
#    plt.show()

#    grad = np.gradient(l_test, rrtest)
#    interp = interp1d(rrtest[2:], grad[2:], kind='cubic', fill_value='extrapolate')
#    print(brentq(interp, 6., rtest))

    rmin = .5
    rmax = 10.
    r0 = .9413703833253498
    l_cons = .0001237035540755403

    plot_V_eff(rmin, rmax, l_cons)

    n_pts = 100
    rr = np.linspace(rmin, rmax, num=n_pts)

    plt.figure(figsize=FIGSIZE)
    plt.plot(rr, [V_eff(r, l_cons) for r in rr], label='V_eff')
    plt.hlines(V_eff(r0, l_cons), rmin, rmax, linestyles='dashed', label='V0')
    plt.grid()
    plt.legend()
    plt.show()

    plot_orbit(r0, l_cons)

#    plt.figure(figsize=FIGSIZE)
#    plt.plot(rr, [-potential_grad(r).value for r in rr], 'k-',
#             label='Potential Grad')
#    plt.plot(rr, [-V_eff_grad(r, l_min) for r in rr], 'r-',
#             label='V_eff_grad, lmin')
#    plt.plot(rr, [(l_min ** 2) / (r ** 3) for r in rr], 'r--', alpha=0.5,
#             label='$l_{min}^{2} / r^{3}$')
#    plt.plot(rr, [-V_eff_grad(r, l_max) for r in rr], 'b-',
#             label='V_eff_grad, lmax')
#    plt.plot(rr, [(l_max ** 2) / (r ** 3) for r in rr], 'b--', alpha=0.5,
#             label='$l_{max}^{2} / r^{3}$')
#    plt.grid()
#    plt.legend()
#    plt.show()

    # plot vc
#    plt.figure(figsize=FIGSIZE)
#    plt.plot(rr, [np.sqrt((G * mass(r)
#                          / (r * u.pc)).to(u.km ** 2 / u.s ** 2)).value
#                  for r in rr], 'k-',
#             label='$v_{c}$')
#    plt.title("$v_{c}$ vs. $r$")
#    plt.xlabel("$r$ [pc]")
#    plt.ylabel("$v_{c}$ [km / s]")
#    plt.grid()
#    plt.legend()
#    plt.show()
#
#    # plot shear
#    plt.figure(figsize=FIGSIZE)
#    plt.plot(rr, [(((r / (2 * mass(r))) * mass_grad(r))).value - .5
#                  for r in rr], 'k-',
#             label='$\\beta$')
#    plt.title("Shear")
#    plt.xlabel("$r$ [pc]")
#    plt.ylabel("$\\beta = \\frac{d\\ln{v_{c}}}{d\\ln{r}}$")
#    plt.grid()
#    plt.legend()
#    plt.show()

#    plot_potential_grad(r1, rmax)
#    plot_acceleration(r1, rmax, l_min)
#    plot_acceleration(r1, rmax, l_max)

#    plot_mass(r1, r20)
#    plot_mass_grad(r1, r20)
#    plot_potential(r1, r20)
#    potential_grad_r = plot_potential_grad(r1, r20)
#    V_max = plot_V_eff(r1, rmax, l_max, post='lmax')
#    V_grad_max = plot_V_eff_grad(r1, rmax, l_max, post='lmax')
#    V_min = plot_V_eff(r1, rmax, l_min, post='lmin')
#    V_grad_min = plot_V_eff_grad(r1, rmax, l_min, post='lmin')

#    plt.figure(figsize=FIGSIZE)
#    plt.plot(rr, [np.sqrt(2 * G * r * mass(r)).value for r in rr])
#    plt.title("Mass limits")
#    plt.ylabel("$\\sqrt{-2 G r_{0} M(r_{0})}$")
#    plt.grid()
#    plt.show()
#
#    plt.figure(figsize=FIGSIZE)
#    plt.plot(rr, [r * np.sqrt(-2 * potential(r)).value for r in rr])
#    plt.title("Potential limits")
#    plt.xlabel("$r_{0}$")
#    plt.ylabel("$r_{0} \\sqrt{-2\\Phi(r_{0})}$")
#    plt.grid()
#    plt.show()

#    l_range = np.linspace(l_min, l_max, num=n_pts)
#    rr, ll = np.meshgrid(rr, l_range, indexing='ij')
#
#    V_eff_r = V_eff(rr, ll)
#
#    fig = plt.figure(figsize=FIGSIZE)
#    ax = fig.gca(projection='3d')
#    ax.plot_surface(rr, ll, V_eff_r)
#    ax.set_xlabel('$r$')
#    ax.set_ylabel('$l$')
#    ax.set_zlabel('$V_{eff}$')
#    ax.set_zlim3d(top=0.)
#    plt.title("$V_{eff}$ vs. $r, l$")
#    save_path = os.path.join(os.path.dirname(__file__), '..',
#                             'out', 'V_eff.pdf')
#    plt.savefig(save_path, bbox_inches='tight')
#    plt.show()
#
#    V_eff_grad_r = V_eff_grad(rr, ll)
#
#    fig = plt.figure(figsize=FIGSIZE)
#    ax = fig.gca(projection='3d')
#    ax.plot_surface(rr, ll, V_eff_grad_r)
##    ax.plot_surface(rr, ll, np.zeros_like(V_eff_grad_r),
##                    color='orange')
#    ax.set_xlabel('$r$')
#    ax.set_ylabel('$l$')
#    ax.set_zlabel('$\\nabla V_{eff}$')
#    ax.set_xlim3d(left=6.)
#    ax.set_zlim3d(bottom=-1e-9, top=1e-9)
#    plt.title("$\\nabla V_{eff}$ vs. $r, l$")
#    save_path = os.path.join(os.path.dirname(__file__), '..', 'out',
#                             'V_eff_grad.pdf')
#    plt.savefig(save_path, bbox_inches='tight')
#    plt.show()

#    plot_orbit(r1, l_min)

#    r_pos, r_vel, ang_pos, ang_vel = orbit(r1, r2)
#
#    print("r_1 = {0:.4f}".format(r1))
#    print("r_2 = {0:.4f}".format(r2))
#    print("r_max = {0:.4f}".format(np.max(r_pos)))
#    print("r_min = {0:.4f}".format(np.min(r_pos)))

#    return V_min, V_grad_min, V_max, V_grad_max, bounds
    return


if __name__ == '__main__':
#    V_min, V_grad_min, V_max, V_grad_max, bounds = main()
    main()
