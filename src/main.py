#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Developed by Julio S Rodriguez (@BlueEarOtter) for the purposes of fitting
orbits to the minispiral of Sgr A West without proper motion data. There are
still issues that need to be worked out.

Modified by J. Andrew Casey-Clyde (@jacaseyclyde) For the purpose of fitting
orbits to the circumnuclear disk
'''

# =============================================================================
# =============================================================================
# # Topmatter
# =============================================================================
# =============================================================================

import orbits
import model

import os
import datetime
import warnings

import numpy as np

import matplotlib.pyplot as plt
import corner

import emcee
from emcee.autocorr import AutocorrError

from multiprocessing import Pool
from multiprocessing import cpu_count

from astropy.io import fits

# Ignores stuff
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

np.set_printoptions(precision=5, threshold=np.inf)

datafile = '../dat/CLF-Sim.csv'
outpath = '../out/'

stamp = ''  #'{:%Y%m%d%H%M%S}/'.format(datetime.datetime.now())


# =============================================================================
# =============================================================================
# # Functions
# =============================================================================
# =============================================================================
def corner_plot(walkers, prange, filename):
    fig = corner.corner(walkers, labels=["$aop$", "$loan$", "$inc$", "$a$",
                                         "$e$"],
                        range=prange)
    fig.set_size_inches(12, 12)

    plt.savefig(outpath + stamp + filename, bbox_inches='tight')
    plt.show()


def orbital_fitting(data, priors, nwalkers=100, nmax=500, reset=True):
    ndim = priors.shape[0]

    # Initialize the chain, which is uniformly distributed in parameter space
    print("initializing parameter space")
    pos_min = priors[:, 0]
    pos_max = priors[:, 1]
    psize = pos_max - pos_min
    pos = [pos_min + psize * np.random.rand(ndim) for i in range(nwalkers)]

    # save positions of the priors to return with all data
    pos_priors = pos

    # Set up backend so we can save chain in case of catastrophe
    # note that this requires h5py and emcee 3.0.0 on github
    filename = 'chain.h5'
    backend = emcee.backends.HDFBackend(filename)
    if reset:
        # starts simulation over
        backend.reset(nwalkers, ndim)

    m = model.Model(data)

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, m.ln_prob, pool=pool,
                                        backend=backend)

        ncpu = cpu_count()
        print("Running MCMC on {0} CPUs".format(ncpu))
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

    try:
        tau = sampler.get_autocorr_time()
    except AutocorrError as e:
        print(e)
        tau = sampler.get_autocorr_time(tol=0)

    print(tau)

    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True,
                                            thin=thin)
    log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)

    print("burn-in: {0}".format(burnin))
    print("thin: {0}".format(thin))
    print("flat chain shape: {0}".format(samples.shape))
    print("flat log prob shape: {0}".format(log_prob_samples.shape))
    print("flat log prior shape: {0}".format(np.shape(log_prior_samples)))

    # let's plot the results
    # using a try catch because as of testing, log_prior_samples is a NoneType
    # object, and I'm not sure why
    try:
        all_samples = np.concatenate((samples, log_prob_samples[:, None],
                                      log_prior_samples[:, None]), axis=1)
    except TypeError as e:
        print(e)
        all_samples = np.concatenate((samples, log_prob_samples[:, None]),
                                      axis=1)

    return samples, pos_priors, all_samples


def main():
    # create output folder
    try:
        os.makedirs(outpath + stamp)
    except FileExistsError:
        pass

    # load data
    my_data = np.genfromtxt(datafile, delimiter=',')

    X = my_data[:, 0]
    Y = my_data[:, 1]
    V = my_data[:, 2]
    Xerr = my_data[:, 3]
    Yerr = my_data[:, 4]
    Verr = my_data[:, 5]
    Verr[Verr == 0] = 4e-2

    data = (np.array([X, Y, V]).T)

    # set up priors
    priors = np.array([[55., 65.], [130., 140.], [295., 305.],
                       [.45, .55], [200., 300.]])

    samples, pos_priors, all_samples = orbital_fitting(data, priors,
                                                       nwalkers=10, nmax=50,
                                                       reset=True)

    # Visualize the fit
    print('plotting priors')
    corner_plot(pos_priors, priors, 'priors.pdf')
    print('plotting results')
    corner_plot(samples, priors, 'results.pdf')

    # analyze the walker data
    aop, loan, inc, a, e = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                               zip(*np.percentile(samples, [16, 50, 84],
                                                  axis=0)))
    aop = aop[0]
    loan = loan[0]
    inc = inc[0]
    a = a[0]
    e = e[0]
    pbest = np.array([aop, loan, inc, a, e])

    print(pbest)

    orbits.plot_func(pbest)

    # bit of cleanup
    if not os.listdir(outpath):
        os.rmdir(outpath)
    return samples


if __name__ == '__main__':
    samples = main()
