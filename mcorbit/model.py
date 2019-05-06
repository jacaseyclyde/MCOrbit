#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This module implements the probability functions used by MCMC.
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
import numpy as np

from scipy.stats import multivariate_normal

import astropy.units as u

from mcorbit import orbits


def ln_like(data, model, cov):
    """Calculates the ln probability of a dataset generating from a model.

    Calculates the probability that the dataset associated with instances
    of this class could have been generated by an orbital model defined by
    the set of parameters `theta`, without any prior knowledge of the
    parameter space.

    Parameters
    ----------
    theta : :obj:`numpy.ndarray`
        An array of orbital parameters in the form::

            numpy.array([aop, loan, inc, r_per, r_ap])

        which define the orbital model to check probabilities for.

    Returns
    -------
    float
        The log likelihood that the associated dataset could have been
        generated by the given model.

    """
    # initializing a vector of probabilities for each point in data.
    # the basic idea here is that for each data point, it's probability
    # of being generated by the model is an integral over the entire
    # entire model, or in this case a sum of the discretized model
    # points. the log likelihood is then a sum of the natural logs of
    # all data points.
    prob = np.zeros(len(data))

    # first we sum the model prob for each data pt over all model pts
    # taking every other model point to improve computation time
    model = model[::2, :]
    for model_pt in model:
        prob += multivariate_normal.pdf(data, mean=model_pt, cov=cov,
                                        allow_singular=True)

    # normalize over the number of points in the model.
    prob /= len(model)

    # this suppresses a runtime warning we expect for log(0)
    with np.errstate(divide='ignore'):
        return np.sum(np.log(prob))


def ln_prior(theta, space):
    """The log likelihood of the priors.

    The log likelihood that the orbital model defined by `theta` is
    correct, based on our prior knowledge of the orbit and parameter space.

    Parameters
    ----------
    theta : :obj:`numpy.ndarray`
        An array of orbital parameters in the form::

            numpy.array([aop, loan, inc, r0, l_cons])

        which define the orbital model being checked.

    Returns
    -------
    float
        The log likelihood that the orbital model defined by `theta` is
        correct, given our prior knowledge of the region.

    Notes
    -----
    Currently, we are assuming a flat prior within the parameter space,
    barring certain dynamic exceptions towards the edges.

    """
    # first check that all parameters are within our parameter space
    prior = 1.
    for i in range(space.shape[0]):
        pmin = min(space[i])
        pmax = max(space[i])

        if theta[i] < pmin or theta[i] > pmax:
            return -np.inf, 0
        else:
            prior *= (1. / (pmax - pmin))

    # ensure periapsis is periapsis
    if theta[-2] > theta[-1]:
        return -np.inf, 0

    if theta[-1] == theta[-2]:
        l_cons = np.sqrt(orbits.mass(theta[-1]) * theta[-1])
    else:
        l_cons = (2 * theta[-2] * theta[-1]
                  * (theta[-1] * orbits.mass(theta[-2])
                  - theta[-2] * orbits.mass(theta[-1]))) / ((theta[-1] ** 2)
                                                            - (theta[-2] ** 2))
        if l_cons < 0.:
            return -np.inf, 0

        l_cons = np.sqrt(l_cons)

    V_eff_r = orbits.V_eff(np.linspace(theta[-2], theta[-1]), l_cons)
    if np.any(V_eff_r > orbits.V_eff(theta[-2], l_cons)):
        return -np.inf, l_cons

    V_p = orbits.V_eff(theta[-2], l_cons)
    V_a = orbits.V_eff(theta[-1], l_cons)
    # V_eff must be equal at orbit endpoints
    if V_p != V_a:
        return -np.inf, l_cons

    # check for stable orbit endpoints
    if (orbits.V_eff_grad(theta[-2], l_cons) == 0
        and orbits.V_eff(theta[-2] - 0.01, l_cons) < V_p):
        return -np.inf, l_cons

    if (orbits.V_eff_grad(theta[-1], l_cons) == 0
        and orbits.V_eff(theta[-1] + 0.01, l_cons) < V_a):
        return -np.inf, l_cons

    return np.log(prior), l_cons


def ln_prob(theta, data, space, cov, f, pos_ang):
    """Calculates P(data|model)

    Calculates the natural log of the Bayesian probability that the
    dataset could have been generated by the given model, defined by
    `theta`.

    Parameters
    ----------
    theta : (aop, loan, inc, r_per, r_ap)
        A tuple of orbital parameters which define the orbital
        model being evaluated.

    Returns
    -------
    lnprob : float
        The log probability that the associated dataset could have been
        generated by the model defined by `theta`, given prior knowledge of
        the region.
    lnprior : float
        The log of the prior for the given parameter space.

    """
    # first check the prior
    lnprior, l_cons = ln_prior(theta, space)
    if not np.isfinite(lnprior):
        return -np.inf, -np.inf
    c = orbits.model(theta, l_cons, coords=True)

    ra = c.ra.rad
    dec = c.dec.rad

    zero_ra = ra - orbits.GCRA.rad
    zero_dec = dec - orbits.GCDEC.rad

    theta = np.arctan2(zero_ra, zero_dec)
    theta = (theta + np.pi) * 180. / np.pi

    whereplus = np.where(theta < 270)
    whereminus = np.where(theta >= 270)

    # Tweak theta to match the astronomical norm (defined as east of north)
    theta[whereplus] = theta[whereplus] + 90
    theta[whereminus] = theta[whereminus] - 270

    model = np.array([ra, dec,
                     c.radial_velocity.value]).T

    wheretheta = np.where((theta >= pos_ang[0]) * (theta <= pos_ang[1]))[0]

    lnlike = ln_like(data, model[wheretheta], cov)
    if not np.isfinite(lnlike):
        return lnprior, -np.inf
    return lnprior + lnlike, lnprior


if __name__ == '__main__':
    # code profiling tasks
    # create theta
    r0 = .9413703833253498
    l_cons = .0001237035540755403
    theta = (34., 43., 37., r0, l_cons)

    # load data
    import os
    import astropy.units as u
    from spectral_cube import SpectralCube, LazyMask
    from astropy.coordinates import SkyCoord

    cube = SpectralCube.read(os.path.join(os.path.dirname(__file__),
                                          '..', 'dat', 'HNC3_2.fits'))

    # create mask to remove the NaN buffer around the image file later
    buffer_mask = LazyMask(lambda num: ~np.isnan(num), cube=cube)

    # mask out contents of maskfile as well as low intensity noise
    mask_cube = SpectralCube.read(os.path.join(os.path.dirname(__file__),
                                               '..', 'dat',
                                               'HNC3_2.mask.fits'))
    mask = (mask_cube == u.Quantity(1)) & (cube > 0.1 * u.Jy / u.beam)
    cube = cube.with_mask(mask)

    cube = cube.subcube_from_mask(buffer_mask)
    cube = cube.with_spectral_unit(u.km / u.s, velocity_convention='radio')

    m1 = cube.moment1()
    dd, rr = m1.spatial_coordinate_map
    c = SkyCoord(ra=rr, dec=dd, radial_velocity=m1, frame='fk5')
    c = c.ravel()

    # convert to numpy array and remove nan velocities
    data_pts = np.array([c.ra.rad,
                         c.dec.rad,
                         c.radial_velocity.value]).T

    # strip out anything that's not an actual data point
    nonnan = ~np.isnan(data_pts[:, 2])
    data = data_pts[nonnan]

    # covaraince matrix
    cov = np.cov(data, rowvar=False)

    ln_like(data, orbits.model(theta), cov)
