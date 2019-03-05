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
from emcee.autocorr import AutocorrError


def fit_orbits(pool, lnlike, data, pspace, nwalkers=500, nmax=10000, burn=1000,
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
#    prange[-1] = .5 * prange[-1]
#    prange[-2] = .5 * prange[-2]
    pos = [pos_min + prange * np.random.rand(ndim) for i in range(nwalkers)]
    cov = np.cov(data, rowvar=False)

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
                                        args=[data, pspace, cov], pool=pool,
                                        backend=backend)

        # initial burn-in. this appears to be necessary to avoid
        # initial NaN issues
#        pos = sampler.run_mcmc(pos, burn, progress=True)
#        sampler.reset()

        # full run
        # this also includes the burn in, which we will discard later
        # discard based on autocorrelation times
        old_tau = np.inf
        autocorr = np.array([])

        for sample in sampler.sample(pos, iterations=nmax, progress=True):
            if sampler.iteration % 100:
                continue

            # check convergence
            tau = sampler.get_autocorr_time(tol=0)
            if not mpi:
                print(tau)
            autocorr = np.append(autocorr, np.mean(tau))

            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
            np.savetxt(os.path.join(outpath, 'acor.csv'), tau, delimiter=',')

        try:
            tau = sampler.get_autocorr_time()
        except Exception:
            tau = sampler.get_autocorr_time(tol=0)

        print("Mean acceptance fraction: {0:.3f}"
              .format(np.mean(sampler.acceptance_fraction)))

        print("Mean autocorrelation time: {0:.3f} steps"
              .format(np.mean(tau)))

        burnin = int(2 * np.nanmax(tau))
        thin = int(0.5 * np.nanmin(tau))
        samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
        log_prob_samples = sampler.get_log_prob(discard=burnin,
                                                flat=True, thin=thin)
        log_prior_samples = sampler.get_blobs(discard=burnin,
                                              flat=True, thin=thin)

        print("tau: {0}".format(tau))
        print("burn-in: {0}".format(burnin))
        print("thin: {0}".format(thin))
        print("flat chain shape: {0}".format(samples.shape))
        print("flat log prob shape: {0}".format(log_prob_samples.shape))
        print("flat log prior shape: {0}".format(log_prior_samples.shape))

        all_samples = np.concatenate((samples, log_prob_samples[:, None],
                                      log_prior_samples[:, None]), axis=1)

    return all_samples, autocorr
