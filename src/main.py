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
import sys
import warnings
import time

from schwimmbad import MPIPool

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord, FK5, ICRS, Angle

from spectral_cube import SpectralCube, LazyMask
from spectral_cube.utils import VarianceWarning

import matplotlib as mpl
import matplotlib.pyplot as plt
import corner
import aplpy

import emcee
from emcee.autocorr import AutocorrError



# Ignores stuff
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
warnings.filterwarnings('ignore', 'Cube is a Stokes cube, ')
warnings.filterwarnings('ignore', category=VarianceWarning)

np.set_printoptions(precision=5, threshold=np.inf)

stamp = ''
outpath = '../out/' + stamp

figsize = (10, 10)
filetype = 'pdf'


# =============================================================================
# =============================================================================
# # Functions
# =============================================================================
# =============================================================================
def notnan(x):
    return ~np.isnan(x)


# =============================================================================
# Data functions
# =============================================================================
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


def ppv_pts(cube):
    # get the moment 1 map and positions, then convert to an array of ppv data
    m1 = cube.moment1()
    dd, rr = m1.spatial_coordinate_map
    c = SkyCoord(ra=rr, dec=dd, radial_velocity=m1, frame='fk5')
    c = c.ravel()

    # convert to numpy array and remove nan velocities
    data_pts = np.array([c.ra.rad, c.dec.rad, c.radial_velocity.value]).T
    data_pts = data_pts[notnan(data_pts[:, 2])]

    return data_pts


# =============================================================================
# Plot functions
# =============================================================================
def _ra_labeler(dec, pos):
    ang = Angle(dec, unit=u.deg)
    h = int(ang.hms.h)
    m = abs(int(ang.hms.m))
    s = abs(round(ang.hms.s, 2))

    if pos == 0:
        return "${0}h\,{1}m\,{2}s$".format(h, m, s)
    else:
        return "${0}s$".format(s)


def _dec_labeler(dec, pos):
    ang = Angle(dec, unit=u.deg)
    d = int(ang.dms.d)
    m = abs(int(ang.dms.m))
    s = abs(round(ang.dms.s, 2))

    if pos == 0 or (m == 0. and s == 0.):
        return "${0}\degree\,{1}'\,{2}''$".format(d, m, s)
    elif s == 0.:
        return "${0}'\,{1}''$".format(m, s)
    else:
        return "${0}''$".format(s)


def _model_plot(img, model, bounds, p, xflag, yflag):
    # start the basics of the plot
    f = plt.figure(figsize=figsize)
    plt.imshow(img, origin='lower', cmap='jet', figure=f, aspect='auto',
               extent=bounds)

    # Add the model
    plt.plot(model[0], model[1], 'k--',
             label='Gas core '
             '($\omega = {0:.2f}, \Omega = {1:.2f}, i = {2:.2f}$)'
             .format(p[0], p[1], p[2]))
    plt.plot(model[0][0], model[1][0], 'r*', label='Model Start')

    # all graphs need a color bar, but the label differs so we add it later
    cbar = plt.colorbar()

    ax = plt.gca()

    if xflag == 'ra' and yflag == 'dec':
        # add Sgr A*
        gc = ICRS(ra=Angle('17h45m40.0409s'),
                  dec=Angle('-29:0:28.118 degrees')).transform_to(FK5)
        sgr_ra = gc.ra.to(u.deg).value
        sgr_dec = gc.dec.to(u.deg).value
        plt.plot(sgr_ra, sgr_dec, label='Sgr A*',
                 c='black', marker='o', ms=5, linestyle='None')

        # add scale bar
        scale = (((.5 * u.pc) / (8. * u.kpc)) * u.rad).to(u.deg).value
        plt.plot((266.40 + scale, 266.40), (-29.024, -29.024), 'k-')
        plt.text(266.40 + 0.5 * scale, -29.024, '0.5 pc',
                 horizontalalignment='center', verticalalignment='bottom')

        cbar.set_label('Integrated Flux $(\mathrm{Hz}\,'
                       '\mathrm{Jy}/\mathrm{beam})$')
    else:
        cbar.set_label('Integrated Flux $(\degree\,'
                       '\mathrm{Jy}/\mathrm{beam})$')

    if xflag == 'ra':
        plt.xlabel('RA (J2000)')

        ra_locations = [(Angle('17h45m36.00s')
                        + i * Angle('0h0m2.00s')).to(u.deg).value
                        for i in range(4)]
        ra_locator = mpl.ticker.FixedLocator(ra_locations)
        ax.xaxis.set_major_locator(ra_locator)

        ra_labeler = mpl.ticker.FuncFormatter(_ra_labeler)
        ax.xaxis.set_major_formatter(ra_labeler)
    else:
        plt.xlabel('Dec (J2000)')

        dec_locations = [(Angle('-28:59:40.0 degrees')
                          - i * Angle('0:0:20.0 degrees')).value
                         for i in range(6)]
        dec_locator = mpl.ticker.FixedLocator(dec_locations)
        ax.xaxis.set_major_locator(dec_locator)

        dec_labeler = mpl.ticker.FuncFormatter(_dec_labeler)
        ax.xaxis.set_major_formatter(dec_labeler)

    if yflag == 'dec':
        plt.ylabel('Dec (J2000)')

        dec_locations = [(Angle('-28:59:40.0 degrees')
                          - i * Angle('0:0:20.0 degrees')).value
                         for i in range(6)]
        dec_locator = mpl.ticker.FixedLocator(dec_locations)
        ax.yaxis.set_major_locator(dec_locator)

        dec_labeler = mpl.ticker.FuncFormatter(_dec_labeler)
        ax.yaxis.set_major_formatter(dec_labeler)
    else:
        plt.ylabel('$v_{r}\,(\mathrm{km} / \mathrm{s})$')

    plt.legend()

    return f


def plot_model(cube, prefix, p):
    vmin = cube.spectral_axis.min().value
    vmax = cube.spectral_axis.max().value
    ra_min, ra_max = cube.longitude_extrema.value
    dec_min, dec_max = cube.latitude_extrema.value

    ra_dec = cube.with_spectral_unit(u.Hz, velocity_convention='radio')
    ra_dec = ra_dec.moment0(axis=0).array
    ra_v = cube.moment0(axis=1).array
    dec_v = cube.moment0(axis=2).array

    # get the model for the given parameters
    c = orbits.sky(p)
    ra = c.ra.to(u.deg).value
    dec = c.dec.to(u.deg).value
    vel = c.radial_velocity.value

    # plot dec vs ra
    f = _model_plot(ra_dec, [ra, dec], [ra_max, ra_min, dec_min, dec_max], p,
                    'ra', 'dec')
    f.savefig(outpath + '{0}_model_ra_dec.{1}'.format(prefix, filetype))

    # plot velocity vs ra
    f = _model_plot(ra_v, [ra, vel], [ra_max, ra_min, vmin, vmax], p,
                    'ra', 'vel')
    f.savefig(outpath + '{0}_model_ra_v.{1}'.format(prefix, filetype))

    # plot dec vs v
    f = _model_plot(dec_v, [dec, vel], [dec_min, dec_max, vmin, vmax], p,
                    'dec', 'vel')
    f.savefig(outpath + '{0}_model_dec_v.{1}'.format(prefix, filetype))


def plot_moment(cube, prefix, moment):
    m = cube.moment(order=moment).hdu
    filename = '{0}_moment_{1}.{2}'

    z_unit = ''
    if moment == 0:
        z_unit = 'Integrated Flux $(\mathrm{Hz}\,\mathrm{Jy}/\mathrm{beam})$'
        cube = cube.with_spectral_unit(u.Hz, velocity_convention='radio')
    elif moment == 1:
        z_unit = '$v_{r} (\mathrm{km}/\mathrm{s})$'
    elif moment == 2:
        z_unit = '$\sigma_{v_{r}}^{2} (\mathrm{km}^{2}/\mathrm{s}^{2})$'
    else:
        # TODO: Use a try except instead
        print('Please choose from moment 0, 1, or 2')
        return

    # plot data
    f = aplpy.FITSFigure(m,  figsize=figsize)
    f.set_xaxis_coord_type('longitude')
    f.set_yaxis_coord_type('latitude')

    # add Sgr A*
    gc = ICRS(ra=Angle('17h45m40.0409s'),
              dec=Angle('-29:0:28.118 degrees')).transform_to(FK5)
    ra = gc.ra.value
    dec = gc.dec.value
    f.show_markers(ra, dec, layer='sgra', label='Sgr A*',
                   edgecolor='black', facecolor='black', marker='o', s=10,)

    plt.legend()

    # add meta information
    f.ticks.show()
    f.add_beam(linestyle='solid', facecolor='white',
               edgecolor='black', linewidth=1)
    f.add_scalebar(((.5 * u.pc) / (8. * u.kpc)) * u.rad)
    f.scalebar.set_label('0.5 pc')

    f.show_colorscale()
    f.add_colorbar()
    f.colorbar.set_axis_label_text(z_unit)

    f.save(outpath + filename.format(prefix, moment, filetype))


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
    pos_min = priors[:, 0]
    pos_max = priors[:, 1]
    psize = pos_max - pos_min
    pos = [pos_min + psize * np.random.rand(ndim) for i in range(nwalkers)]

    # save positions of the priors to return with all data
    pos_priors = pos


    m = model.Model(data)

    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit()

        # Set up backend so we can save chain in case of catastrophe
        # note that this requires h5py and emcee 3.0.0 on github
        filename = 'chain.h5'
        backend = emcee.backends.HDFBackend(filename)
        if reset:
            # starts simulation over
            backend.reset(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, m.ln_prob, pool=pool,
                                        backend=backend)

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

    burnin = int(2 * np.nanmax(tau))
    thin = int(0.5 * np.nanmin(tau))
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
    # Grab the initial time for total runtime statistics
    t0 = time.time()

    # create output folder
    try:
        os.makedirs(outpath + stamp)
    except FileExistsError:
        pass

    # load data
#    HNC3_2_cube = import_data(cubefile='HNC3_2.fits', maskfile=None)
    masked_HNC3_2_cube = import_data(cubefile='HNC3_2.fits',
                                     maskfile='HNC3_2.mask.fits')

    # plot the first 3 moments of each cube
#    plot_moment(HNC3_2_cube, 'HNC3_2', moment=0)
#    plot_moment(HNC3_2_cube, 'HNC3_2', moment=1)
#    plot_moment(HNC3_2_cube, 'HNC3_2', moment=2)
#
#    plot_moment(masked_HNC3_2_cube, 'HNC3_2_masked', moment=0)
#    plot_moment(masked_HNC3_2_cube, 'HNC3_2_masked', moment=1)
#    plot_moment(masked_HNC3_2_cube, 'HNC3_2_masked', moment=2)

    data = ppv_pts(masked_HNC3_2_cube)

    # set up priors and do MCMC
    priors = np.array([[55., 65.], [130., 140.], [295., 305.],
                       [0., 1.5], [1.5, 4.]])

    samples, pos_priors, all_samples = orbital_fitting(data, priors,
                                                       nwalkers=100, nmax=5,
                                                       reset=True)

    # Visualize the fit
    print('plotting priors')
    corner_plot(pos_priors, priors, 'priors.pdf')
    print('plotting results')
    corner_plot(samples, priors, 'results.pdf')

    # analyze the walker data
    aop, loan, inc, r_per, r_ap = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                      zip(*np.percentile(samples, [16, 50, 84],
                                                         axis=0)))
#    aop = aop[0]
#    loan = loan[0]
#    inc = inc[0]
#    r_per = r_per[0]
#    r_ap = r_ap[0]
    pbest = np.array([aop[0], loan[0], inc[0], r_per[0], r_ap[0]])

    # print the best parameters found and plot the fit
    print("Best Fit")
    print("aop: {0}, loan: {1}, inc: {2}, r_per: {3}, r_ap: {4}".format(pbest))
#    plot_model(masked_HNC3_2_cube, 'HNC3_2_masked', pbest)
    t1 = time.time()
    print("Runtime: {0}".format(t1 - t0))

    # bit of cleanup
    if not os.listdir(outpath):
        os.rmdir(outpath)


if __name__ == '__main__':
    main()
#    pass
