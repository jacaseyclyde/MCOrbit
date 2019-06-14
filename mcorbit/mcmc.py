#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Implementation of MCMC analysis.
# Copyright (C) 2017-2019  J. Andrew Casey-Clyde
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

"""Performs MCMC analysis.

Module that defines most of the MCMC code. Primarily implements methods
from [1]_emcee.

References
----------
.. [1] Foreman-Mackey, D., Hogg, D., Lang, D. and Goodman, J., "emcee:
    The MCMC Hammer" Publications of the Astronomical Society of the
    Pacific, vol. 125, pp. 306, 2013.

"""
import sys
import os

import numpy as np
import emcee

from sklearn.cluster import MeanShift


def fit_orbits(pool, lnlike, data, pspace, pos_ang_lim,
               nwalkers=500, nmax=10000, burn=1000,
               reset=True, mpi=False, outpath=None):
    """Uses MCMC to explore the parameter space specified by `priors`.

    Uses MCMC to fit orbits to the given `data`, exploring the parameter space
    specified by `priors`.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        An array of ppv datapoints. See the returns of :func:`ppv_pts` for more
        details on the formatting of this array
    pspace : :obj:`numpy.ndarray`
        A numpy array specifying the bounds of the parameter space. This should
        have the form::

            np.array([
                      [aop_min, aop_max],
                      [loan_min, loan_max],
                      [inc_min, inc_max],
                      [r_per_min, r_per_max],
                      [r_ap_min, r_ap_max]
                     ])

        where aop, loan, inc, r_per and r_ap are the parameters we are
        exploring (argument of pericenter, line of ascending nodes, inclination
        radius of pericenter, and radius of apocenter, respectively), and the
        _min and _max postfixes indicate, respectively, the minimum and maximum
        valuesfor each of these axes in parameter space
    nwalkers : int, optional
        The number of walkers to use for parameter space exploration. Default
        is 100.
    nmax : int, optional
        The maximum number of steps to run through. Default is 500.
    reset : bool, optional
        If True, the state of the probability space sampler will be reset, and
        sampling will start from scratch. If False, the sampler will load it's
        last recorded state, and continue sampling the space from there.
        Default is True.

    Returns
    -------
    samples : array[..., nwalkers, ndim]
        The positions of all walkers after sampling is complete.
    priors : array[..., nwalkers, ndim]
        The positions of all walkers at the start of sampling.
    all_samples : array[..., nwalkers, ndim, log_prob_samples, log_prob_priors]
        The positions of all walkers after sampling is complete, as well as the
        log probabilities of the samples and the priors.

    """
    # get the dimensionality of our parameter space
    ndim = pspace.shape[0]

    # Initialize the chain, which is uniformly distributed in parameter space
    pos_min = pspace[:, 0]
    pos_max = pspace[:, 1]
    prange = pos_max - pos_min
    pos = [pos_min + prange * np.random.rand(ndim) for i in range(nwalkers)]
    
    # separate data and emission weights
    data, weights = np.hsplit(data, [3])

    # Calculate a covariance matrix for fitting. The covariance for the
    # whole dataset is unreliable and unrealistic (it's calculation
    # assumes a single generating point near the mean, rather than the
    # generating function we are considering). Instead, we will use
    # clustering to calculate a more "typical" covariance at any given
    # point along the orbit.
    # To cluster our data properly, we first normalize each axis to
    # [-1, 1]. We use a MeanShift clustering algorithm for bandwidth
    # 0.125, which is roughly typical for the normalized data
    dmin = np.min(data, axis=0)
    dscale = 1. / (np.max(data, axis=0) - dmin)
    data = (data - dmin) * dscale * 2 - 1
    ms = MeanShift(bandwidth=.125).fit(data)

    # we then take our covariance to be the mean cov of the clusters
    cov = np.mean([np.cov(data[ms.labels_ == k], rowvar=False)
                   for k in np.unique(ms.labels_)], axis=0)

    lprobscale = 0.5 * (3 * np.log(2 * np.pi) + np.log(np.linalg.det(cov)))
    
    # Set up backe end for walker position saving
    # note that this requires h5py and emcee 3.x
    backend = emcee.backends.HDFBackend(os.path.join(outpath, 'chain.h5'))
    if reset:
        # starts simulation over
        backend.reset(nwalkers, ndim)

    with pool as pool:
        if mpi and not pool.is_master():
            pool.wait()
            sys.exit(0)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike,
                                        args=[data, weights, lprobscale,
                                              pspace, cov, pos_ang_lim,
                                              dmin, dscale],
                                        pool=pool, backend=backend)

        # initial burn-in. this appears to be necessary to avoid
        # initial NaN issues
#        pos = sampler.run_mcmc(pos, burn, progress=True)
#        sampler.reset()

        # full run
        # this also includes the burn in, which we will discard later
        # discard based on autocorrelation times
        old_tau = np.inf

        for sample in sampler.sample(pos, iterations=nmax, progress=True):
            if sampler.iteration % 100:
                continue

            # check convergence
            tau = sampler.get_autocorr_time(tol=0)

            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau

            with open(os.path.join(outpath, 'acor.csv'), mode='a') as f:
                np.savetxt(f, tau.reshape(1, len(tau)), delimiter=',')

    return sampler
