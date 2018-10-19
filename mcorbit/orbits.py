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

"""
Created on Fri Feb  9 16:08:27 2018

@author: jacaseyclyde

"""

# =============================================================================
# =============================================================================
# # Topmatter
# =============================================================================
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt


import astropy.units as u
from astropy.constants import G
from astropy.coordinates import Galactocentric, FK5, ICRS, Angle

np.set_printoptions(precision=5, threshold=np.inf)

outpath = '../out/'

# =============================================================================
# =============================================================================
# # Global Parameters
# =============================================================================
# =============================================================================

# =============================================================================
# Conversions
# =============================================================================
pcArcsec = 8e3 / 60 / 60 / 180 * np.pi  # [pc/arcsec] @ gal. center
kgMsun = 1.98855e+30  # [kg/Msun]
secYr = 60. * 60. * 24. * 365.  # [s/yr]
kmPc = 3.0857e+13  # [km/pc]
mKm = 1.e+3  # [m/km]


# =============================================================================
# Constants
# =============================================================================
gc = ICRS(ra=Angle('17h45m40.0409s'),
          dec=Angle('-29:0:28.118 degrees'))  # galactic center
# =============================================================================
# Mass Data
# =============================================================================
Mdat = np.genfromtxt(os.path.join(os.path.dirname(__file__),
                                  '../dat/enclosed_mass_distribution.txt'))
Mdist = Mdat[:, 0] * pcArcsec  # [pc]
Menc = Mdat[:, 1]  # [log(Msun)]

# =============================================================================
# Other
# =============================================================================
h = 500  # timestep
ttot = 1e5


# =============================================================================
# =============================================================================
# # Functions
# =============================================================================
# =============================================================================
def mass_func(dist):
    '''
    Takes in a distance from SgrA* and returns the enclosed mass. Based on data
    from Feldmeier-Krause et al. (2016), with interpolation

    dist = [pc]
    '''
    return 10**np.interp(dist, Mdist, Menc)  # [Msun]


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


def potential_grad(dist, h=0.001):
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
    sample_dists = np.array([dist - h, dist, dist + h])
    potentials = grav_potential(sample_dists)
    grads = np.gradient(potentials, sample_dists)
    return grads[1]


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


def orbit(r_per, r_ap, tstep, ttot):
    """Generates orbits.

    Takes in a peri/apoapsis and generates an integrated orbit around SgrA*.
    Returns 2 arrays of position and veolocity vectors.

    x0 = [pc], v0 = [km/s], tstep = [yr]
    """
    # pylint: disable=E1101
#    npoints = int(ttot / tstep)
#    pos = np.zeros((npoints, 3))
#    vel = np.zeros_like(pos)
#
#    x0 = np.array([r_per, 0., 0.]) * u.pc
#    v_per = np.sqrt((2. * G * mass_func(r_per) * r_ap)
#                    / (r_per * (r_per + r_ap)))
#    v0 = np.array([0., v_per, 0.]) * u.pc / u.yr
#
#    pos[0] = x0.value  # [pc]
#    vel[0] = v0.value  # [pc/yr]
#
#    posnorm = np.linalg.norm(pos[0])  # [pc]
#    a_old = - G * mass_func(posnorm) / posnorm**2  # [pc/yr^2]
#    a_old = a_old * pos[0] / posnorm
#
#    for i in range(npoints - 1):
#        pos[i+1] = pos[i] + vel[i] * tstep + 0.5 * a_old * tstep**2
#
#        posnorm = np.linalg.norm(pos[i+1])
#        a_new = - G * mass_func(posnorm) / posnorm**2
#        a_new = a_new * pos[i+1] / posnorm
#
#        vel[i+1] = vel[i] + 0.5 * (a_old + a_new) * tstep
#
#        a_old = a_new
#
#    return pos, (vel * u.pc / u.yr).to(u.km / u.s).value  # [pc], [km/s]
    # sticking to 2D polar for initial integration since z = 0
    r_pos = np.array([r_per])
    r_vel = np.array([0.])

    ang_pos = np.array([0.])
    ang_v0 = (r_ap / r_per) * np.sqrt((2 * G / ((r_per ** 2) - (r_ap ** 2)))
                                      * ((mass_func(r_per) / r_per)
                                      - (mass_func(r_ap) / r_ap)))
    ang_vel = np.array([ang_v0])

    l_cons = ang_v0 * (r_per ** 2)  # angular momentum per unit mass
    while ang_pos[-1] < 2 * np.pi:
        # radial portion first
        r_half = r_pos[-1] + 0.5 * h * r_vel[-1]  # first kick
        r_vel_new = r_vel[-1] - h * 8  # drift  # TODO: teach me how to gradie(nt)
        r_new = r_half + 0.5 * h * r_vel_new  # second kick
        r_pos = np.append(r_pos, r_new)

        # then radial
        ang_half = ang_pos[-1] + 0.5 * h * ang_vel[-1]
        ang_vel_new = l_cons / (r_new ** 2)
        ang_new = ang_half + 0.5 * h * ang_vel_new
        ang_pos = np.append(ang_pos, ang_new)

    pos = np.array([r_pos * np.cos(ang_pos),
                    r_pos * np.sin(ang_pos),
                    [0.] * len(r_pos)])

    vel = np.array([r_vel * np.cos(ang_pos) - r_pos * ang_vel * np.sin(ang_pos),
                    r_vel * np.sin(ang_pos) + r_pos * ang_vel * np.cos(ang_pos),
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

    orb_r, orb_v = orbit(r_per, r_ap, tstep, ttot)

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
    orb_r, orb_v = orbit(p[-2], p[-1], tstep, ttot)
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
