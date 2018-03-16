#!/usr/bin/env python2


'''
Created on Fri Feb  9 16:08:27 2018

@author: jacaseyclyde
'''

# =============================================================================
# =============================================================================
# # Topmatter
# =============================================================================
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

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
G = 6.67e-11 * kgMsun / (mKm * kmPc)**3 * secYr**2  # [pc^3 Msun^-1 yr^-2]

# =============================================================================
# Mass Data
# =============================================================================
Mdat = np.genfromtxt('../dat/enclosed_mass_distribution.txt')
Mdist = Mdat[:, 0] * pcArcsec  # [pc]
Menc = Mdat[:, 1]  # [log(Msun)]

# =============================================================================
# Other
# =============================================================================
tstep = 100
ttot = 100000


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


def orbit(x0, v0, tstep, ttot):
    '''
    Takes in an initial position and velocity vector and generates an
    integrated orbit around SgrA*. Returns 2 arrays of position and veolocity
    vectors.

    x0 = [pc], v0 = [km/s], tstep = [yr]
    '''
    npoints = int(ttot / tstep)
    pos = np.zeros((npoints, 3))
    vel = np.zeros_like(pos)

    pos[0] = x0  # [pc]
    vel[0] = v0 * secYr / kmPc  # [pc/yr]

    posnorm = np.linalg.norm(pos[0])  # [pc]
    a_old = - G * mass_func(posnorm) / posnorm**2  # [pc/yr^2]
    a_old = a_old * pos[0] / posnorm

    for i in range(npoints - 1):
        pos[i+1] = pos[i] + vel[i] * tstep + 0.5 * a_old * tstep**2

        posnorm = np.linalg.norm(pos[i+1])
        a_new = - G * mass_func(posnorm) / posnorm**2
        a_new = a_new * pos[i+1] / posnorm

        vel[i+1] = vel[i] + 0.5 * (a_old + a_new) * tstep

        a_old = a_new

    return pos, (vel / secYr * kmPc)  # [pc], [km/s]


def sky(p):
    """
    Takes in coordinates ~p in degrees~ and spits out f1 -- the constraint that
    the orbit must be elliptical and SgrA* lies at the focus
    """

    (aop, loan, inc, x0, v0) = p

    # convert from degrees to radians
    aop = aop * np.pi / 180.
    loan = loan * np.pi / 180.
    inc = inc * np.pi / 180.

    rot = rot_mat(aop, loan, inc)
    orb_r, orb_v = orbit(x0, v0, tstep, ttot)

    # Transform from Orbit plane to Sky plane
    # We can use the transpose of the rotation matrix instead of the inverse
    # because it's Hermition
    sky_R = np.matmul(rot.T, orb_r.T)
    sky_V = np.matmul(rot.T, orb_v.T)
    los_V = sky_V[-1]

    model = np.array([sky_R[0], sky_R[1], los_V])

    return model.T  # return transpose to get individual ellipse points


def plot_func(p):
    orb_r, orb_v = orbit(p[3], p[4], tstep, ttot)
    sky_xyv = sky(p)

    plt.figure(1)
    plt.plot(sky_xyv[:, 0], sky_xyv[:, 1], 'k-', label='Gas core')
    plt.plot([0], [0], 'g*', label='Sgr A*')

    plt.grid(True)
    plt.axes().set_aspect('equal', 'datalim')
    plt.title('Sky Plane')  # . p = {}'.format(p))
    plt.xlabel('Offset (pc)')
    plt.ylabel('Offset (pc)')
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
