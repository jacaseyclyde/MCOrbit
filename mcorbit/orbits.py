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

from scipy.interpolate import interp1d

import astropy.units as u
from astropy.constants import G
from astropy.coordinates import Galactocentric, FK5, ICRS, Angle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

outpath = '../out/'

# galactic center
GC = ICRS(ra=Angle('17h45m40.0409s'), dec=Angle('-29:0:28.118 degrees'))

M_DAT = np.genfromtxt(os.path.join(os.path.dirname(__file__),
                                   '../dat/enclosed_mass_distribution.txt'))
M_DIST = Angle(M_DAT[:, 0], unit=u.arcsec).to(u.rad) * 8.0e3 * u.pc / u.rad
M_ENC = M_DAT[:, 1]  # 10 ** Mdat[:, 1] * u.Msun
M_GRAD = np.gradient(M_ENC, M_DIST.value) * u.Msun / u.pc

## uncomment to use point mass equal to mass enclosed at ~1pc
#M_ENC = len(M_ENC) * [M_ENC[175]]  # 1pc away from Sgr A*
#M_GRAD = np.zeros(len(M_ENC)) * u.Msun / u.pc
#
## uncomment to use point mass equal to mass enclosed at ~5pc
#M_ENC = len(M_ENC) * [M_ENC[240]]  # 5pc away from Sgr A*
#M_GRAD = np.zeros(len(M_ENC)) * u.Msun / u.pc

M_ENC_INTERP = interp1d(M_DIST.value, M_ENC,
                        kind='cubic', fill_value='extrapolate')
M_GRAD_INTERP = interp1d(M_DIST.value, M_GRAD,
                         kind='cubic', fill_value='extrapolate')

h = 500 * u.yr  # timestep

G = G.to((u.pc ** 3) / (u.Msun * u.yr ** 2))


def mass(dist):
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

    if dist == np.inf * u.pc:
        return np.inf * u.Msun

    mass_enc = M_ENC_INTERP(dist)
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
        if (dist.unit != u.pc):
            try:
                dist = dist.to(u.pc)
            except u.UnitConversionError as e:
                raise e
    except AttributeError as e:
        # Assume units in pc if no units given
        dist = dist * u.pc

    with warnings.catch_warnings(record=True):
        if dist == np.inf:
            return 0. * (u.pc ** 2) / (u.yr ** 2)
        return - G * mass(dist) / dist


def mass_grad(dist, data=M_GRAD):
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
    try:
        if (dist.unit != u.pc):
            try:
                dist = dist.to(u.pc)
            except u.UnitConversionError as e:
                raise e
    except AttributeError as e:
        # Assume units in pc if no units given
        dist = dist * u.pc

    if dist == np.inf * u.pc:
        return 0. * u.Msun / u.pc

    m_grad_dist = (M_GRAD_INTERP(dist) * mass(dist) * np.log(10)).value
    return m_grad_dist * u.Msun / u.pc


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
    if (dist == 0):
        return np.inf * u.pc / (u.yr ** 2)

    return (-1. / dist) * (potential(dist) + G * mass_grad(dist))


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

    E = ((((r2 ** 2) * potential(r2)) - ((r1 ** 2) * potential(r1)))
         / ((r2 ** 2) - (r1 ** 2)))

#    return np.sqrt(2 * (((r2 ** -2) - (r1 ** -2)) ** -1)
#                   * (potential(r2) - potential(r1)))
    return r1 * np.sqrt(2 * (E - potential(r1)))


def centrifugal_acceleration(dist, l_cons):
    """The centrifugal pseudo-acceleration.

    The acceleration in the radial direction due to a centrifugal
    pseudo-force.

    Parameters
    ----------
    dist : float
        The distance at which to calculate this acceleration, in units
        of parsecs
    l_cons : float
        The angular momentum per unit mass, in units of parsec^2 / year

    Returns
    -------
    float
        The centrifugal "acceleration", in units of parsecs / year^2

    """
    return (l_cons ** 2) / (dist ** 3)


def orbit(r1, r2, tstep):
    """Generates orbits.

    Orbit generator that integrates orbits around Sgr A*. Integrated
    orbits are necessary due to the presence of a distributed central
    mass function. The mass distribution is assumed to be spherical.

    Parameters
    ----------
    r1 : float
        The first apside. May be either periapsis or apoapsis. Units of
        parsecs.
    r2 : float
        The second apside. May be either periapsis or apoapsis. Units
        of parsecs.
    tstep : float
        The timestep to use, in years.

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
    # pylint: disable=E1101
    # force the order of apsides such that r1 = periapsis
    if (r1 > r2):
        tmp = r1
        r1 = r2
        r2 = tmp

    # sticking to 2D polar for initial integration since z = 0
    # TODO: Check if the issue in the integrated orbit is due to the
    # radial integrator
    r1 *= u.pc
    r2 *= u.pc
    r_pos = np.array([r1.value]) * u.pc
    r_vel = np.array([0.]) * u.pc / u.yr

    ang_pos = np.array([0.]) * u.rad

    l_cons = angular_momentum(r1, r2)

    ang_v0 = l_cons / (r1 ** 2) * u.rad
    ang_vel = np.array([ang_v0.value]) * u.rad / u.yr

    while ang_pos[-1] < 2 * np.pi * u.rad:
        # radial portion first
        # first drift
        r_half = r_pos[-1] + 0.5 * h * r_vel[-1]
#        r_half = round(r_half.value, 3) * u.pc

        # kick
        a_centrifugal = centrifugal_acceleration(r_half, l_cons)
        r_vel_new = r_vel[-1] + h * (a_centrifugal - potential_grad(r_half))
#        r_vel_new = round(r_vel_new.value, 3) * u.pc / u.yr

        # second drift
        r_new = r_half + 0.5 * h * r_vel_new
#        r_new = round(r_new.value, 3) * u.pc  # significant figures
        r_pos = np.append(r_pos.value, r_new.value) * u.pc
        r_vel = np.append(r_vel.value, r_vel_new.value) * u.pc / u.yr

        # then angular
        ang_half = ang_pos[-1] + 0.5 * h * ang_vel[-1]
        ang_vel_new = l_cons * u.rad / (r_new ** 2)
        ang_new = ang_half + 0.5 * h * ang_vel_new
        ang_pos = np.append(ang_pos.value, ang_new.value) * u.rad
        ang_vel = np.append(ang_vel.value, ang_vel_new.value) * u.rad / u.yr

        if (r_pos[-1] > 5 * r2):
            break

    return r_pos, r_vel, ang_pos, ang_vel


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
                       galcen_coord=GC).transform_to(FK5)

    return c


def model(p):
    # wrapper function to convert the coordinates to a numpy array of datapts
    c = sky(p)
    return np.array([c.ra.rad, c.dec.rad, c.radial_velocity.value]).T


def plot_orbit(r1, r2):
    pos, vel = polar_to_cartesian(*orbit(r1, r2, 500))

    plt.figure(figsize=figsize)
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

    plt.figure(figsize=figsize)
    plt.plot(r_pos, mass_r)
    plt.title("$M$ vs. $r$")
    plt.xlabel("$r [pc]$")
    plt.ylabel("$M(r) [M_{\\odot}]$")
    plt.grid()
    plt.show()

    return mass_r


def plot_mass_grad(r1, r2):
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
    mass_grad_r = [mass_grad(r).value for r in r_pos]

    plt.figure(figsize=figsize)
    plt.plot(r_pos, mass_grad_r)
    plt.title("$dM/dr$ vs. $r$")
    plt.xlabel("$r [\\mathrm{pc}]$")
    plt.ylabel("$dM/dr [M_{\\odot} / \\mathrm{pc}]$")
    plt.grid()
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

    plt.figure(figsize=figsize)
    plt.plot(r_pos, potential_r)
    plt.title("$\\Phi(r)$ vs. $r$")
    plt.xlabel("$r [\\mathrm{pc}]$")
    plt.ylabel("$\\Phi [\\mathrm{pc}^{2} / \\mathrm{yr}^{2}]$")
    plt.grid()
    plt.show()

    return potential_r


def plot_potential_grad(r1, r2):
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
    potential_grad_r = [potential_grad(r).value for r in r_pos]

    plt.figure(figsize=figsize)
    plt.plot(r_pos, potential_grad_r)
    plt.title("$\\nabla\\Phi(r)$ vs. $r$")
    plt.xlabel("$r [\\mathrm{pc}]$")
    plt.ylabel("$\\nabla\\Phi [\\mathrm{pc} / \\mathrm{yr}^{2}]$")
    plt.grid()
    plt.show()

    return potential_grad_r


@np.vectorize
def V_eff(r, r1, r2):
    l_cons = angular_momentum(r1, r2)
    V_l = ((l_cons ** 2) / (2 * (r ** 2))).value
    return V_l + potential(r).value


def plot_V_eff(r1, r2):
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

    plt.figure(figsize=figsize)
    plt.plot(r_pos, V_eff(r_pos, r1, r2))
    plt.title("$V_{\\mathrm{eff}}(r)$ vs. $r$")
    plt.xlabel("$r [\\mathrm{pc}]$")
    plt.ylabel("$V_{\\mathrm{eff}} [\\mathrm{pc}^{2} / \\mathrm{yr}^{2}]$")
    plt.grid()
    save_path = os.path.join(os.path.dirname(__file__), 'V_eff_r.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_acceleration(r1, r2):
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
    potential_grad_r = [potential_grad(r).value for r in r_pos]

    l_cons = angular_momentum(r1, r2)
    a_l = ((l_cons ** 2) / (r_pos ** 3)).value

    a = a_l - potential_grad_r

    plt.figure(figsize=figsize)
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
    r_pos, r_vel, ang_pos, ang_vel = orbit(r1, r2, 100)

    plt.figure(figsize=figsize)
    plt.plot(ang_pos, r_vel)
    plt.title("$v_{r}$ vs. $\\theta$")
    plt.xlabel("$\\theta [\\mathrm{rad}]$")
    plt.ylabel("$v_{r} [\\mathrm{pc} / \\mathrm{yr}]$")
    plt.grid()
    plt.show()

    plt.figure(figsize=figsize)
    plt.plot(ang_pos, ang_vel)
    plt.title("$v_{\\theta}$ vs. $\\theta$")
    plt.xlabel("$\\theta [\\mathrm{rad}]$")
    plt.ylabel("$v_{\\theta} [\\mathrm{rad} / \\mathrm{yr}]$")
    plt.grid()
    plt.show()

    return r_vel, ang_vel


if __name__ == '__main__':
    # set initial variables
    figsize = (12, 12)
    r1 = 1.
    r2 = 5.

    # set up gridspace of apsides
    n_pts = 20
    rr = np.linspace(r1, r2, num=n_pts)
    rr1 = np.linspace(r1, r2, num=n_pts)
    rr2 = np.linspace(r1, r2, num=n_pts)
    rr, rr1, rr2 = np.meshgrid(rr, rr1, rr2, indexing='ij')

#    mass_r = plot_mass(1., 5.)
#    mass_grad_r = plot_mass_grad(1., 5.)
#    potential_r = plot_potential(1., 5.)
#    potential_grad_r = plot_potential_grad(1., 5.)
    plot_V_eff(r1, r2)

    V_eff_r = V_eff(rr, rr1, rr2)

    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    ir2 = -1
    surf = ax.plot_surface(rr[:, :, ir2], rr1[:, :, ir2], V_eff_r[:, :, ir2])
    ax.set_xlabel('$r$')
    ax.set_ylabel('$r_{1}$')
    ax.set_zlabel('$V_{eff}$')
    plt.title("$V_e$ vs. $r, r_1, r_2 = {0:.3f}$".format(rr2[0, 0, ir2]))
    save_path = os.path.join(os.path.dirname(__file__), 'V_eff_Mr_r2.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    ir1 = 0
    surf = ax.plot_surface(rr[:, ir1, :], rr2[:, ir1, :], V_eff_r[:, ir1, :])
    ax.set_xlabel('$r$')
    ax.set_ylabel('$r_{2}$')
    ax.set_zlabel('$V_{eff}$')
    plt.title("$V_e$ vs. $r, r_2, r_1 = {0:.3f}$".format(rr2[0, ir1, 0]))
    save_path = os.path.join(os.path.dirname(__file__), 'V_eff_Mr_r1.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    # 4D plotting
#    fig = plt.figure(figsize=figsize)
#    ax = fig.gca(projection='3d')
#    ax.scatter(rr, rr1, rr2, s=V_eff_r)

#    r_vel, ang_vel = plot_velocity(r1, r2)
#    r_acc = plot_acceleration(r1, r2)

#    plot_orbit(r1, r2)

    r_pos, r_vel, ang_pos, ang_vel = orbit(r1, r2, 100)

    print("r_1 = {0:.4f}".format(r1))
    print("r_2 = {0:.4f}".format(r2))
    print("r_max = {0:.4f}".format(np.max(r_pos)))
    print("r_min = {0:.4f}".format(np.min(r_pos)))
