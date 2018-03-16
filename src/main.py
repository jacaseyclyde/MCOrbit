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

import os
import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee
from multiprocessing import Pool
from multiprocessing import cpu_count
import datetime
import warnings

import orbits
import model

# Ignores stuff
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

np.set_printoptions(precision=5, threshold=np.inf)

datafile = '../dat/CLF-Sim.csv'
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
tspace = 1000
ttest = np.linspace(0, 2 * np.pi, tspace)

# =============================================================================
# =============================================================================
# # Main Program
# =============================================================================
# =============================================================================

# initialize stamp
stamp = '{:%Y%m%d%H%M%S}/'.format(datetime.datetime.now())

# create output folder
os.makedirs(outpath + stamp)

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
cov = np.cov(data.T)
H = np.linalg.inv(cov)

# =============================================================================
# MC MC
# =============================================================================

# Now, let's setup some parameters that define the MCMC
ndim = 5
nwalkers = 100
n_max = 1000

priors = np.array([[0., 0., 0., .1, .5], [np.pi, np.pi, np.pi, 2, 1.]])
prange = np.ndarray.tolist(priors.T)

# Initialize the chain
# Choice 1: chain uniformly distributed in the range of the parameters
print("initializing parameters")
pos_min = priors[0, :]
pos_max = priors[1, :]
psize = pos_max - pos_min
pos = [pos_min + psize*np.random.rand(ndim) for i in range(nwalkers)]

# Visualize the initialization
fig = corner.corner(pos, labels=["$aop$", "$loan$", "$inc$", "$a$", "$e$"],
                    range=prange)
fig.set_size_inches(10, 10)


plt.savefig(outpath + stamp + 'priors.pdf', bbox_inches='tight')
plt.show()

# Set up backend so we can save chain in case of catastrophe
# note that this requires h5py and emcee 3.0.0 on github
filename = 'chain.h5'
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)  # uncomment to start simulation from scratch

m = model.Model(data)

with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, m.ln_prob, pool=pool,
                                    backend=backend)

    ncpu = cpu_count()
    print("Running MCMC on {0} CPUs".format(ncpu))
    old_tau = np.inf
    for sample in sampler.sample(pos, iterations=n_max, progress=True):
        if sampler.iteration % 100:
            continue

        # check convergence
        tau = sampler.get_autocorr_time(tol=0)
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau

sampler.get_autocorr_time()  # insert tau here if not autocorrelated
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))
print("flat log prior shape: {0}".format(np.shape(log_prior_samples)))


# let's plot the results
all_samples = np.concatenate((
    samples, log_prob_samples[:, None]  # , log_prior_samples[:, None]
), axis=1)

fig = corner.corner(samples, labels=["$aop$", "$loan$", "$inc$", "$a$", "$e$"],
                    range=prange)
fig.set_size_inches(10, 10)
plt.savefig(outpath + stamp + 'results_{0}.pdf'.format(nwalkers),
            bbox_inches='tight')
plt.show()

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
