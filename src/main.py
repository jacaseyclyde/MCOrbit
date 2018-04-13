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
import warnings

from multiprocessing import Pool
from multiprocessing import cpu_count

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Galactocentric

from spectral_cube import SpectralCube, LazyMask

import matplotlib.pyplot as plt
import corner
import aplpy

import emcee
from emcee.autocorr import AutocorrError

# Ignores stuff
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

np.set_printoptions(precision=5, threshold=np.inf)

datafile = '../dat/CLF-Sim.csv'
stamp = ''
outpath = '../out/' + stamp


# =============================================================================
# =============================================================================
# # Functions
# =============================================================================
# =============================================================================
def notnan(x):
    return ~np.isnan(x)


def import_data(cubefile=None, maskfile=None):
    HNC_cube = SpectralCube.read('../dat/{0}'.format(cubefile))

    # create mask to remove the NaN buffer around the image file later
    buffer_mask = LazyMask(notnan, cube=HNC_cube)

    # mask out contents of maskfile as well as low intensity noise
    if maskfile is not None:
        mask_cube = SpectralCube.read('../dat/{0}'.format(maskfile))
        mask = (mask_cube == u.Quantity(1)) & (HNC_cube > 0.1 * u.Jy / u.beam)

    else:
        mask = HNC_cube > 0.1 * u.Jy / u.beam

    HNC_cube = HNC_cube.with_mask(mask)

    HNC_cube = HNC_cube.subcube_from_mask(buffer_mask)
    return HNC_cube.with_spectral_unit(u.km / u.s,
                                       velocity_convention='radio')


def convert_points(cube):
    # import the coordinate for Sgr A* in FK5 (matching our data)
    galcen = SkyCoord(Galactocentric.galcen_coord).fk5

    # get the moment 1 map and positions, then convert to an array of ppv data
    m1 = cube.moment1()
    dd, rr = m1.spatial_coordinate_map
    c = SkyCoord(ra=rr, dec=dd, radial_velocity=m1, frame='fk5')
    c = c.ravel()


def plot_moment(cube, prefix, moment):
    # calculate moments
    m = cube.moment(order=moment).hdu  # integrated intensity

    z_unit = ''
    if moment == 0:
        z_unit = 'Flux (Jy/beam)'
    elif moment == 1:
        z_unit = '$v_r (km/s)$'
    elif moment == 2:
        z_unit = '$v_r (km^{2}/s^{2})$'
    else:
        print('Please choose from moment 0, 1, or 2')
        return

    f = aplpy.FITSFigure(m)
    f.show_colorscale()
    f.add_colorbar()
    f.colorbar.set_axis_label_text(z_unit)
    f.save(outpath + '{0}_moment_{1}.png'.format(prefix, moment))


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
    HNC3_2_cube = import_data(cubefile='HNC3_2.fits', maskfile=None)
    masked_HNC3_2_cube = import_data(cubefile='HNC3_2.fits',
                                     maskfile='HNC3_2.mask.fits')

    # plot the first 3 moments of each cube
    plot_moment(HNC3_2_cube, 'HNC3_2', moment=0)
    plot_moment(HNC3_2_cube, 'HNC3_2', moment=1)
    plot_moment(HNC3_2_cube, 'HNC3_2', moment=2)
    plot_moment(masked_HNC3_2_cube, 'HNC3_2_masked', moment=0)
    plot_moment(masked_HNC3_2_cube, 'HNC3_2_masked', moment=1)
    plot_moment(masked_HNC3_2_cube, 'HNC3_2_masked', moment=2)

    X = my_data[:, 0]
    Y = my_data[:, 1]
    V = my_data[:, 2]
    Xerr = my_data[:, 3]
    Yerr = my_data[:, 4]
    Verr = my_data[:, 5]
    Verr[Verr == 0] = 4e-2

    data = (np.array([X, Y, V]).T)

    # set up priors and do MCMC
    priors = np.array([[55., 65.], [130., 140.], [295., 305.],
                       [.45, .55], [200., 300.]])

    samples, pos_priors, all_samples = orbital_fitting(data, priors,
                                                       nwalkers=100, nmax=50,
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

    # print the best parameters found and plot the fit
    print(pbest)
    orbits.plot_func(pbest)

    # bit of cleanup
    if not os.listdir(outpath):
        os.rmdir(outpath)
    return samples


# if __name__ == '__main__':
    # samples = main()
