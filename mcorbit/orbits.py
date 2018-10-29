#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is MC Orbit, an MCMC application for fitting orbits near Sgr A*.
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
Menc = 10**Mdat[:, 1] * u.Msun

h = 500 * u.yr  # timestep

G = G.to((u.pc ** 3) / (u.Msun * u.yr ** 2))


def mass_func(dist):
    '''
    Takes in a distance from SgrA* and returns the enclosed mass. Based on data
    from Feldmeier-Krause et al. (2016), with interpolation

    dist = [pc]
    '''
    return np.interp(dist, Mdist, Menc) * u.Msun


def grav_potential(dist):
    """Definition of the potential in a region.

    Function that defines the gravitational potential at a given radius
    from Sgr A*

    Parameters
    ----------
    dist : float
        The distance from Sgr A* at which to compute the potential.

    Returns
    -------
    float
        The calculated potential

    Todo
    ----
    Check units on the returned potential

    """
    return -G * mass_func(dist) / dist


def potential_grad(dist):
    """Calculates the gradient of the potential at a point.

    Calculates the approximate gradient of the gravitational potential
    due to Sgr A* at the point indicated.

    Parameters
    ----------
    dist : float
        The distance from Sgr A* at which to calculate the potential
    h : float, optional (default = 0.001)
        The spacing to be used for calculating the potential

    """
    # use only the actual data on either side of the interpolated point
    sample_dists = np.array([np.max(Mdist[Mdist < dist].value),
                             dist.value,
                             np.min(Mdist[Mdist > dist].value)])

    # treating dM/dr as negligible at any given point
    return (- grav_potential(sample_dists).value / dist
            * u.pc ** 2 / (u.yr ** 2))


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


def orbit(r_per, r_ap, tstep):
    """Generates orbits.

    Takes in a peri/apoapsis and generates an integrated orbit around SgrA*.
    Returns 2 arrays of position and veolocity vectors.

    x0 = [pc], v0 = [km/s], tstep = [yr]
    """
    # pylint: disable=E1101
    # sticking to 2D polar for initial integration since z = 0
    r_per *= u.pc
    r_ap *= u.pc
    r_pos = np.array([r_per.value]) * u.pc
    r_vel = np.array([0.]) * u.pc / u.yr

    ang_pos = np.array([0.]) * u.rad

    if (r_per == r_ap):
        ang_v0 = np.sqrt(G * mass_func(r_per) / r_per**3)
    else:
        ang_v0 = (r_ap / r_per) * np.sqrt((2 * G
                                          / ((r_per ** 2) - (r_ap ** 2)))
                                          * ((mass_func(r_per) / r_per)
                                          - (mass_func(r_ap) / r_ap)))

    ang_v0 *= u.rad
    ang_vel = np.array([ang_v0.value]) * u.rad / u.yr

    l_cons = ang_v0 * (r_per ** 2)  # angular momentum per unit mass
    while ang_pos[-1] < 2 * np.pi * u.rad:
        # radial portion first
        r_half = r_pos[-1] + 0.5 * h * r_vel[-1]  # first drift
        r_vel_new = r_vel[-1] + h * ((l_cons ** 2 / u.rad ** 2) / r_half ** 3
                                     - potential_grad(r_half))  # kick
        r_new = r_half + 0.5 * h * r_vel_new  # second drift
        r_pos = np.append(r_pos.value, r_new.value) * u.pc
        r_vel = np.append(r_vel.value, r_vel_new.value) * u.pc / u.yr

        # then radial
        ang_half = ang_pos[-1] + 0.5 * h * ang_vel[-1]
        ang_vel_new = l_cons / (r_new ** 2)
        ang_new = ang_half + 0.5 * h * ang_vel_new
        ang_pos = np.append(ang_pos.value, ang_new.value) * u.rad
        ang_vel = np.append(ang_vel.value, ang_vel_new.value) * u.rad / u.yr

    pos = np.array([r_pos * np.cos(ang_pos),
                    r_pos * np.sin(ang_pos),
                    [0.] * len(r_pos)])

    vel = np.array([r_vel * np.cos(ang_pos)
                    - r_pos * ang_vel * np.sin(ang_pos) / u.rad,
                    r_vel * np.sin(ang_pos)
                    + r_pos * ang_vel * np.cos(ang_pos) / u.rad,
                    [0.] * len(r_vel)])

    return pos, vel


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

    orb_r, orb_v = orbit(r_per, r_ap, h)

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
    orbit(1., 1., 500 * u.yr)
