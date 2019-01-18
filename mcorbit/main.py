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

"""This is MC Orbit, an MCMC application for fitting orbits near Sgr A*.

MC Orbit is a program designed for fititng the orbits of gas clouds near
Sagittarius A*, the supermassive black hole (SMBH) at the center of our galaxy,
the Milky Way. This project relies on several large data files, which in the
interest of saving space in the distribution of this application, are not
included in the program repository. For quesitons about the data, please
contact PI Elisabeth A.C. Mills.

"""
# pylint: disable=C0413
import os
import sys
import warnings
import argparse
# import time

# Set up warning filters for things that don't really matter to us
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
warnings.filterwarnings('ignore', 'Cube is a Stokes cube, ')
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="The mpl_toolkits.axes_grid module "
                        "was deprecated in version 2.1")
warnings.filterwarnings("ignore", category=FutureWarning)

import schwimmbad  # noqa

import numpy as np  # noqa

import astropy.units as u  # noqa
from astropy.coordinates import SkyCoord, FK5, ICRS, Angle  # noqa

from spectral_cube import SpectralCube, LazyMask  # noqa

import matplotlib as mpl  # noqa
mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa
import corner  # noqa
import aplpy  # noqa

import emcee  # noqa
from emcee.autocorr import AutocorrError  # noqa

from mcorbit import orbits  # noqa
from mcorbit.model import Model  # noqa

np.set_printoptions(precision=5, threshold=np.inf)

STAMP = ''
OUTPATH = '../out/'

FIGSIZE = (10, 10)
FILETYPE = 'pdf'


# =============================================================================
# =============================================================================
# # Function definitions
# =============================================================================
# =============================================================================

def _notnan(num):
    """
    Inverse of :func:`numpy.isnan`
    """
    return ~np.isnan(num)


# =============================================================================
# Data fandling functions
# =============================================================================

def import_data(cubefile, maskfile=None):
    """Pipeline function that processes spectral data.

    Imports spectral cube data and filters out background noise. Noise filter
    is set at 0.1 Jy / beam. If a maskfile is specified, this is also applied
    to the imported data.

    Parameters
    ----------
    cubefile : str
        Name of the spectral cube file to load
    maskfile : str, optional
        Name of the mask to apply. By default, no mask will be applied, but
        background noise will still be filtered.

    Returns
    -------
    :obj:`spectral_cube.SpectralCube`
        A spectral cube with "spectral" units of km/s (the recessional
        velocity).

    """
    # pylint: disable=E1101
    cube = SpectralCube.read('../dat/{0}'.format(cubefile))

    # create mask to remove the NaN buffer around the image file later
    buffer_mask = LazyMask(_notnan, cube=cube)

    # mask out contents of maskfile as well as low intensity noise
    if maskfile is not None:
        mask_cube = SpectralCube.read('../dat/{0}'.format(maskfile))
        mask = (mask_cube == u.Quantity(1)) & (cube > 0.1 * u.Jy / u.beam)

    else:
        mask = cube > 0.1 * u.Jy / u.beam

    cube = cube.with_mask(mask)

    cube = cube.subcube_from_mask(buffer_mask)
    return cube.with_spectral_unit(u.km / u.s, velocity_convention='radio')


def ppv_pts(cube):
    """Converts spectral cube data to ppv datapoints.

    This function converts the data in a spectral cube into individual
    datapoints of position and recessional velocity, with each pixel of the
    flattened image (i.e., each pixel of the moment 1 map) corresponding to a
    single data point. Pixels without a recessoinal velocity (i.e., `NaN`
    valued velocities) are not considered datapoints

    Parameters
    ----------
    cube : :obj:`spectral_cube.SpectralCube`
        Spectral cube to be converted

    Returns
    -------
    :obj:`numpy.ndarray`
        A numpy array of ppv data points.

        Each data point is itself a list of the form [ra, dec, vel], where ra
        and dec are the right ascension and declination, respectively, in
        radians, while vel is the recessional velocity in units of km/s, and is
        based on the moment 1 map of the original data cube, which is itself an
        intensity weighted average of the gas velocity at each sky position.

    """
    # pylint: disable=C0103
    # get the moment 1 map and positions, then convert to an array of ppv data
    moment1 = cube.moment1()
    dd, rr = moment1.spatial_coordinate_map
    c = SkyCoord(ra=rr, dec=dd, radial_velocity=moment1, frame='fk5')
    c = c.ravel()

    # convert to numpy array and remove nan velocities
    data_pts = np.array([c.ra.rad,
                         c.dec.rad,
                         c.radial_velocity.value]).T

    # strip out anything that's not an actual data point
    data_pts = data_pts[_notnan(data_pts[:, 2])]

    return data_pts


# =============================================================================
# Plot functions
# =============================================================================
def _ra_labeler(dec, pos):
    """
    Generates the right ascension labels for plots.

    """
    # pylint: disable=E1101, W1401
    # pylint: disable=anomalous-backslash-in-string
    ang = Angle(dec, unit=u.deg)
    hour = int(ang.hms.h)
    minute = abs(int(ang.hms.m))
    sec = abs(round(ang.hms.s, 2))

    if pos == 0:
        return "${0}h\,{1}m\,{2}s$".format(hour, minute, sec)

    return "${0}s$".format(sec)


def _dec_labeler(dec, pos):
    """
    Generates the declination labels for plots.
    """
    # pylint: disable=E1101, W1401
    ang = Angle(dec, unit=u.deg)
    deg = int(ang.dms.d)
    minute = abs(int(ang.dms.m))
    sec = abs(round(ang.dms.s, 2))

    if pos == 0 or (minute == 0. and sec == 0.):
        return "${0}\degree\,{1}'\,{2}''$".format(deg, minute, sec)
    elif sec == 0.:
        return "${0}'\,{1}''$".format(minute, sec)

    return "${0}''$".format(sec)


def _model_plot(img, mdl, bounds, params, flags):
    """
    This is where the bulk of the plotting for the models actually occurs
    """
    # pylint: disable=E1101, W1401, C0103
    # start the basics of the plot
    fig = plt.figure(figsize=FIGSIZE)
    plt.imshow(img, origin='lower', cmap='jet', figure=fig, aspect='auto',
               extent=bounds)

    # Add the model
    plt.plot(mdl[0], mdl[1], 'k--',
             label='Gas core '
             '($\omega = {0:.2f}, \Omega = {1:.2f}, i = {2:.2f}$)'
             .format(params[0], params[1], params[2]))
    plt.plot(mdl[0][0], mdl[1][0], 'r*', label='Model Start')

    # all graphs need a color bar, but the label differs so we add it later
    cbar = plt.colorbar()

    axes = plt.gca()

    if flags[0] == 'ra' and flags[1] == 'dec':
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

    # Set the labels for the x and y axes based on the flags passed
    # Note (JACC): I'm sure this can be done in a much better way, w/o flags
    # XXX: Refactor this
    if flags[0] == 'ra':
        plt.xlabel('RA (J2000)')

        ra_locations = [(Angle('17h45m36.00s')
                         + i * Angle('0h0m2.00s')).to(u.deg).value
                        for i in range(4)]
        axes.xaxis.set_major_locator(mpl.ticker.FixedLocator(ra_locations))

        axes.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_ra_labeler))
    else:
        plt.xlabel('Dec (J2000)')

        dec_locations = [(Angle('-28:59:40.0 degrees')
                          - i * Angle('0:0:20.0 degrees')).value
                         for i in range(6)]
        axes.xaxis.set_major_locator(mpl.ticker.FixedLocator(dec_locations))

        axes.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_dec_labeler))

    if flags[1] == 'dec':
        plt.ylabel('Dec (J2000)')

        dec_locations = [(Angle('-28:59:40.0 degrees')
                          - i * Angle('0:0:20.0 degrees')).value
                         for i in range(6)]
        axes.yaxis.set_major_locator(mpl.ticker.FixedLocator(dec_locations))

        axes.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_dec_labeler))
    else:
        plt.ylabel('$v_{r}\,(\mathrm{km} / \mathrm{s})$')

    plt.legend()

    return fig


def plot_model(cube, params, prefix):
    """Plots the model in 3 graphs depicting each plane of the ppv space

    Plots an orbit model (as defined by `params`) over the data, showing all 3
    planes of the position-position-velocity space. The data in each plane is
    shown as a moment 0 map of the integrated intensity in each data column
    (integrated along the axis perpendicular to the axes shown in each plot).

    Parameters
    ----------
    cube : :obj:`spectral_cube.SpectralCube`
        The spectral cube to use for the model plot.
    params : :obj:`numpy.ndarray`
        The parameters to use for the orbital model that is being plotted.
    prefix : str
        Prefix to use when saving files.

    """
    # pylint: disable=E1101, C0103
    vmin = cube.spectral_axis.min().value
    vmax = cube.spectral_axis.max().value
    ra_min, ra_max = cube.longitude_extrema.value
    dec_min, dec_max = cube.latitude_extrema.value

    ra_dec = cube.with_spectral_unit(u.Hz, velocity_convention='radio')

    # get the model for the given parameters
    c = orbits.sky(params)
    ra = c.ra.to(u.deg).value
    dec = c.dec.to(u.deg).value
    vel = c.radial_velocity.value

    # plot dec vs ra
    f = _model_plot(ra_dec.moment0(axis=0).array, [ra, dec],
                    [ra_max, ra_min, dec_min, dec_max], params, ['ra', 'dec'])
    f.savefig(OUTPATH + '{0}_model_ra_dec.{1}'.format(prefix, FILETYPE))

    # plot velocity vs ra
    f = _model_plot(cube.moment0(axis=1).array, [ra, vel],
                    [ra_max, ra_min, vmin, vmax], params, ['ra', 'vel'])
    f.savefig(OUTPATH + '{0}_model_ra_v.{1}'.format(prefix, FILETYPE))

    # plot dec vs v
    f = _model_plot(cube.moment0(axis=2).array, [dec, vel],
                    [dec_min, dec_max, vmin, vmax], params, ['dec', 'vel'])
    f.savefig(OUTPATH + '{0}_model_dec_v.{1}'.format(prefix, FILETYPE))


def plot_moment(cube, moment, prefix):
    """Saves an image of the given `moment` for a spectral cube

    Create a plot of the given `moment` for a spectral cube. Plots are then
    saved to the path specified by OUTPATH, with names generated based on the
    `prefix` given, as well as the `moment`.

    Parameters
    ----------
    cube : :obj:`spectral_cube.SpectralCube`
        The spectral cube to plot moments for
    moment : int
        The moment to plot
    prefix : str
        Prefix to use for filename

    """
    # TODO: Refactor function to plot all 3 moments in a single call
    # pylint: disable=E1101, W1401, C0103
    m = cube.moment(order=moment).hdu
    filename = '{0}_moment_{1}.{2}'

    # XXX: Throw an error instead of printing
    z_unit = ''
    if moment == 0:
        z_unit = "Integrated Flux $(\\mathrm{Hz}\\,\\mathrm{Jy}/"
        "\\mathrm{beam})$"
        cube = cube.with_spectral_unit(u.Hz, velocity_convention='radio')
    elif moment == 1:
        z_unit = "$v_{r} (\\mathrm{km}/\\mathrm{s})$"
    elif moment == 2:
        z_unit = "$\\sigma_{v_{r}}^{2} (\\mathrm{km}^{2}/\\mathrm{s}^{2})$"
    else:
        print("Please choose from moment 0, 1, or 2")
        return

    # plot data
    f = aplpy.FITSFigure(m, figsize=FIGSIZE)
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

    f.save(OUTPATH + filename.format(prefix, moment, FILETYPE))


def corner_plot(walkers, prange, filename):
    """Wrapper function for creating and saving graphs of the parameter space.

    Creates and saves a corner plot of the parameter space we are doing MCMC
    in. This is really just a wrapper function for :func:`corner.corner` that
    also saves the final graph.

    Parameters
    ----------
    walkers : :obj:`numpy.ndarray`
        Contains the positions of all the walkers which are used to explore the
        parameter space. Each walker is a :obj:`numpy.ndarray` whose values
        specify the walker's position in parameter space.
    prange : array_like
        The bounds of the parameter space.
    filename : str
        Name of the file to save the plot to.

    """
    # TODO: update the docstring entry for walkers with their structure
    # TODO: Fix labels
    fig = corner.corner(walkers,
                        labels=["$aop$", "$loan$", "$inc$", "$a$", "$e$"],
                        range=prange)
    fig.set_size_inches(12, 12)

    plt.savefig(OUTPATH + STAMP + filename, bbox_inches='tight')
    plt.show()


# =============================================================================
# MCMC functions
# =============================================================================

def orbital_fitting(pool, data, pspace, nwalkers=100, nmax=500, reset=True,
                    mpi=False):
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
    # pylint: disable=C0103
    # TODO: refactor this code
    # JACC Note: it might just be worth it to rebuild this entire section from
    # the ground up
    ndim = pspace.shape[0]

    # Initialize the chain, which is uniformly distributed in parameter space
    pos_min = pspace[:, 0]
    pos_max = pspace[:, 1]
    prange = pos_max - pos_min
    pos = [pos_min + prange * np.random.rand(ndim) for i in range(nwalkers)]

#    # save positions of the priors to return with all data
#    priors = pos
    with pool as pool:
        m = Model(data, pspace)

        if mpi and not pool.is_master():
            pool.wait()
            sys.exit(0)

        # Set up backend so we can save chain in case of catastrophe
        # note that this requires h5py and emcee 3.0.0 on github
        filename = 'chain.h5'
        backend = emcee.backends.HDFBackend(filename)
        if reset:
            # starts simulation over
            backend.reset(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, m.ln_prob, pool=pool,
                                        backend=backend)

        # burn-in for 100 steps
        state = sampler.run_mcmc(pos, 100)

        # full run
        sampler.run_mcmc(state, nmax)

        samples = sampler.get_chain(flat=True)

#        old_tau = np.inf
#        for sample in sampler.sample(pos, iterations=nmax, progress=True):
#            if sampler.iteration % 100:
#                continue
#
#            # check convergence
#            tau = sampler.get_autocorr_time(tol=0)
#            converged = np.all(tau * 100 < sampler.iteration)
#            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
#            if converged:
#                break
#            old_tau = tau
#
#    try:
#        tau = sampler.get_autocorr_time()
#    except AutocorrError as e:
#        print(e)
#        tau = sampler.get_autocorr_time(tol=0)
#
#    print(tau)

#    burnin = int(2 * np.nanmax(tau))
#    thin = int(0.5 * np.nanmin(tau))
#    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
#    log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True,
#                                            thin=thin)
#    log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)
#
#    print("burn-in: {0}".format(burnin))
#    print("thin: {0}".format(thin))
#    print("flat chain shape: {0}".format(samples.shape))
#    print("flat log prob shape: {0}".format(log_prob_samples.shape))
#    print("flat log prior shape: {0}".format(np.shape(log_prior_samples)))
#
#    # let's plot the results
#    # using a try catch because as of testing, log_prior_samples is a NoneType
#    # object, and I'm not sure why
#    # XXX: needs fixing so that this just isn't an issue anymore
#    try:
#        all_samples = np.concatenate((samples, log_prob_samples[:, None],
#                                      log_prior_samples[:, None]), axis=1)
#    except TypeError as e:
#        print(e)
#        all_samples = np.concatenate((samples, log_prob_samples[:, None]),
#                                     axis=1)

    return samples


# =============================================================================
# Main program
# =============================================================================

def main(pool, args):
    """The main function of MC Orbit. Used to start all sampling

    This is the main function for MC Orbit. It carries out what is effectivley
    the entire study, including applying masks, initializing plots, and
    starting the actual MCMC process.

    """
    # Grab the initial time for total runtime statistics
    # t0 = time.time()

    # create output folder
    try:
        os.makedirs(OUTPATH + STAMP)
    except FileExistsError:
        pass

    # load data
    hnc3_2_cube = import_data(cubefile='HNC3_2.fits', maskfile=None)
    masked_hnc3_2_cube = import_data(cubefile='HNC3_2.fits',
                                     maskfile='HNC3_2.mask.fits')

    # plot the first 3 moments of each cube
#    plot_moment(hnc3_2_cube, moment=0, prefix='HNC3_2')
#    plot_moment(hnc3_2_cube, moment=1, prefix='HNC3_2')
#    plot_moment(hnc3_2_cube, moment=2, prefix='HNC3_2')
#
#    plot_moment(masked_hnc3_2_cube, moment=0, prefix='HNC3_2_masked')
#    plot_moment(masked_hnc3_2_cube, moment=1, prefix='HNC3_2_masked')
#    plot_moment(masked_hnc3_2_cube, moment=2, prefix='HNC3_2_masked')

    data = ppv_pts(masked_hnc3_2_cube)

    # set up priors and do MCMC
    pspace = np.array([[55., 65.], [130., 140.], [295., 305.],
                       [0., 1.5], [1.5, 4.]])

    samples = orbital_fitting(pool, data, pspace, nwalkers=32,
                              nmax=1000, reset=True, mpi=args.mpi)

    plt.hist(samples[:, 0], 100, color='k', histtype='step')
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$p(\theta_1)$")
    plt.gca().set_yticks([]);
#
#    # Visualize the fit
#    print('plotting priors')
#    corner_plot(pos_priors, priors, 'priors.pdf')
#    print('plotting results')
#    corner_plot(samples, priors, 'results.pdf')
#
#    # analyze the walker data
#    aop, loan, inc, r_per, r_ap = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#                                      zip(*np.percentile(samples, [16, 50, 84],
#                                                         axis=0)))
#    aop = aop[0]
#    loan = loan[0]
#    inc = inc[0]
#    r_per = r_per[0]
#    r_ap = r_ap[0]
#    pbest = np.array([aop[0], loan[0], inc[0], r_per[0], r_ap[0]])

    # print the best parameters found and plot the fit
#    print("Best Fit")
#    print("aop: {0}, loan: {1}, inc: {2}, r_per: {3}, r_ap: {4}".format(pbest))
#    plot_model(masked_hnc3_2_cube, 'HNC3_2_masked', pbest)
#    t1 = time.time()
#    print("Runtime: {0}".format(t1 - t0))

    # bit of cleanup
    if not os.listdir(OUTPATH):
        os.rmdir(OUTPATH)

    return masked_hnc3_2_cube


if __name__ == '__main__':
    # Parse command line arguments
    PARSER = argparse.ArgumentParser()

    # Add command line flags
    PARSER.add_argument('-d', '--data_path', action='store',
                        type=str, dest='DATA_PATH', default="~/data/")
    GROUP = PARSER.add_mutually_exclusive_group()
    GROUP.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes "
                       "(uses multiprocessing).")
    GROUP.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    args = PARSER.parse_args()

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    main(pool, args)
#    pass
