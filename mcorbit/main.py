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
import gc

import logging
import warnings

import argparse
from pathlib import Path

# Set up warning filters for things that don't really matter to us
#warnings.filterwarnings('ignore', 'The iteration is not making good progress')
warnings.filterwarnings('ignore', 'Cube is a Stokes cube, ')
#warnings.filterwarnings("ignore", message="numpy.dtype size changed")
#warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
#warnings.filterwarnings("ignore", message="The mpl_toolkits.axes_grid module "
#                        "was deprecated in version 2.1")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

import schwimmbad
import emcee

import numpy as np

from scipy.optimize import brentq

import astropy.units as u  # noqa
from astropy.coordinates import SkyCoord, FK5, ICRS, Angle

from spectral_cube import SpectralCube, LazyMask

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import corner
import aplpy

from mcorbit import orbits
from mcorbit import mcmc


np.set_printoptions(precision=2, threshold=np.inf)

STAMP = ""  # datetime.datetime.now().strftime('%Y%m%d%H%M%S')
OUTPATH = os.path.join(os.path.dirname(__file__), '..', 'out')

FIGSIZE = (10, 10)
FILETYPE = 'pdf'

GAL_CENTER = ICRS(ra=Angle('17h45m40.0409s'),
                  dec=Angle('-29:0:28.118 degrees'))
GCRA = GAL_CENTER.transform_to(FK5).ra.value
GCDEC = GAL_CENTER.transform_to(FK5).dec.value
D_SGR_A = 8.127 * u.kpc


# %%
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


# %%
# =============================================================================
# Data fandling functions
# =============================================================================
def import_data(cubefile, maskfile=None, clean=True):
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
    logging.info("Loading {0}".format(cubefile))
    cube = SpectralCube.read(cubefile)

    buffer_mask = LazyMask(lambda x: ~np.isnan(x), cube=cube)

    if maskfile is not None or clean:
        # few times rms for thresholding
        rms = 4. * np.sqrt(np.nanmean(np.square(cube.hdu.data)))

    # mask out contents of maskfile as well as low intensity noise
    if maskfile is not None:
        mask_cube = SpectralCube.read(maskfile)
        mask = (mask_cube == u.Quantity(1)) & (cube > rms * u.Jy / u.beam)
        cube = cube.with_mask(mask)

    elif clean:
        mask = cube > rms * u.Jy / u.beam
        cube = cube.with_mask(mask)

    cube = cube.subcube_from_mask(buffer_mask)
    return cube.with_spectral_unit(u.km / u.s, velocity_convention='radio')


# %%
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
    m1 = cube.moment1()
    dd, rr = m1.spatial_coordinate_map
    c = SkyCoord(ra=rr, dec=dd, radial_velocity=m1, frame='fk5')
    c = c.ravel()

    # convert to numpy array and remove nan velocities
    data_pts = np.array([c.ra.rad,
                         c.dec.rad,
                         c.radial_velocity.value]).T

    # strip out anything that's not an actual data point
    data_pts = data_pts[_notnan(data_pts[:, 2])]

    return data_pts


# %%
# =============================================================================
# Plot functions
# =============================================================================
def plot_model(cube, prefix, params, theta_min, theta_max, label=None):
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
    try:
        os.makedirs(os.path.join(OUTPATH, STAMP, 'HNC3_2_fit'))
    except FileExistsError:
        pass

    # pylint: disable=E1101, C0103
    ra_min, ra_max = cube.longitude_extrema.value
    dec_min, dec_max = cube.latitude_extrema.value

    ra_dec = cube.with_spectral_unit(u.Hz, velocity_convention='radio')

    aop, loan, inc, r_p, r_a = params
    if r_p == r_a:
        ones = np.ones(360)
        orbit = (ones * r_p * u.pc,
                 ones * 0. * u.pc / u.yr,
                 np.linspace(0., 2. * np.pi, 360) * u.rad,
                 np.sqrt((orbits.mass(r_p) / (r_p ** 3))) * u.rad / u.yr)
        pos, vel = orbits.polar_to_cartesian(*orbit)

        pos, vel = orbits.orbit_rotator(pos, vel, aop, loan, inc)
        c = orbits.sky_coords(pos, vel)
    else:
        # get the model for the given parameters
        l_cons = (2 * r_p * r_a * (r_a * orbits.mass(r_p)
                  - r_p * orbits.mass(r_a))) / ((r_a ** 2) - (r_p ** 2))
        l_cons = np.sqrt(l_cons)
        c = orbits.model(params, l_cons, coords=True)

    ra = c.ra.to(u.deg).value
    dec = c.dec.to(u.deg).value
    vel = c.radial_velocity.value

    m0 = ra_dec.moment0(axis=0)

    # plot data
    f = aplpy.FITSFigure(m0.hdu, figsize=FIGSIZE)
    f.set_xaxis_coord_type('longitude')
    f.set_yaxis_coord_type('latitude')

    x, y = f.world2pixel(ra, dec)

    sgrastar_x, sgrastar_y = f.world2pixel(GCRA, GCDEC)

    zero_x = x - sgrastar_x
    zero_y = y - sgrastar_y

    theta = np.arctan2(zero_y, zero_x)
    theta = (theta + np.pi) * 180. / np.pi

    whereplus = np.where(theta < 270)
    whereminus = np.where(theta >= 270)

    # Tweak theta to match the astronomical norm (defined as east of north)
    theta[whereplus] = theta[whereplus] + 90
    theta[whereminus] = theta[whereminus] - 270

    wheretheta = np.where((theta >= theta_min) * (theta <= theta_max))[0]

    model = np.array([ra.tolist(), dec.tolist()]).T[wheretheta]
    theta, model = np.transpose(sorted(zip(theta[wheretheta], model)))
    ra = [point[0] for point in model]
    dec = [point[1] for point in model]

    # add Sgr A*
    f.show_markers(GCRA, GCDEC, layer='sgra', label='Sgr A*',
                   edgecolor='black', facecolor='black', marker='o', s=10)

    f.show_lines([np.array([ra, dec])],
                 layer='model', color='black', linestyle='dashed',
                 label=label)

    plt.legend()

    # add meta information
    f.ticks.show()
    f.add_beam(linestyle='solid', facecolor='white',
               edgecolor='black', linewidth=1)
    f.add_scalebar(((.5 * u.pc) / D_SGR_A * u.rad))
    f.scalebar.set_label('0.5 pc')

    f.show_colorscale()
    f.add_colorbar()
    f.colorbar.set_axis_label_text('Integrated Flux $(\\mathrm{Hz}\\,'
                                   '\\mathrm{Jy}/\\mathrm{beam})$')

    f.savefig(os.path.join(OUTPATH, STAMP, 'HNC3_2_fit', '{0}_model.{1}'
                           .format(prefix, FILETYPE)))
    f.close()
    return


# %%
def pa_model(params, f, theta_min, theta_max):
    aop, loan, inc, r_p, r_a = params
    if r_p == r_a:
        ones = np.ones(360)
        orbit = (ones * r_p * u.pc,
                 ones * 0. * u.pc / u.yr,
                 np.linspace(0., 2. * np.pi, 360) * u.rad,
                 np.sqrt((orbits.mass(r_p) / (r_p ** 3))) * u.rad / u.yr)
        pos, vel = orbits.polar_to_cartesian(*orbit)

        pos, vel = orbits.orbit_rotator(pos, vel, aop, loan, inc)
        vel *= -1
        c = orbits.sky_coords(pos, vel)
    else:
        l_cons = (2 * r_p * r_a * (r_a * orbits.mass(r_p)
                  - r_p * orbits.mass(r_a))) / ((r_a ** 2) - (r_p ** 2))
        l_cons = np.sqrt(l_cons)
        c = orbits.model(params, l_cons, coords=True)

    ra = c.ra.to(u.deg).value
    dec = c.dec.to(u.deg).value
    vel = c.radial_velocity.value

    x, y = f.world2pixel(ra, dec)

    sgrastar_x, sgrastar_y = f.world2pixel(GCRA, GCDEC)

    zero_x = x - sgrastar_x
    zero_y = y - sgrastar_y

    theta = np.arctan2(zero_y, zero_x)
    theta = (theta + np.pi) * 180. / np.pi

    whereplus = np.where(theta < 270)
    whereminus = np.where(theta >= 270)

    # Tweak theta to match the astronomical norm (defined as east of north)
    theta[whereplus] = theta[whereplus] + 90
    theta[whereminus] = theta[whereminus] - 270

    wheretheta = np.where((theta >= theta_min) * (theta <= theta_max))[0]

    return np.transpose(sorted(zip(theta[wheretheta], vel[wheretheta])))


# %%
def pa_transform(cube):
    cube = cube.with_spectral_unit(u.Hz, velocity_convention='radio')
    m0 = cube.moment0()

    # position angle plot
    xx, yy = np.meshgrid(np.arange(0, m0.shape[1]), np.arange(0, m0.shape[0]),
                         indexing='xy')

    # convenient functions for finding the location of Sgr A*
    f = aplpy.FITSFigure(m0.hdu, figsize=FIGSIZE)
    f.set_xaxis_coord_type('longitude')
    f.set_yaxis_coord_type('latitude')

    # get pixel coords of Sgr A* before plotting
    sgrastar_x, sgrastar_y = f.world2pixel(GCRA, GCDEC)

    zero_x = xx - sgrastar_x  # center the pixel indices on the black hole
    zero_y = yy - sgrastar_y

    theta = np.arctan2(zero_y, zero_x)  # define a new coordinate theta
    theta = (theta + np.pi) * 180.0 / np.pi

    whereplus = np.where(theta < 270)
    whereminus = np.where(theta >= 270)

    # Tweak theta to match the astronomical norm (defined as east of north)
    theta[whereplus] = theta[whereplus] + 90
    theta[whereminus] = theta[whereminus] - 270

    # set up position angle
    pos_ang = np.zeros((360, cube.shape[0]))
    for i in np.arange(360):
        wheretheta = np.where((theta > i) * (theta < i+1))
        wherex = xx[wheretheta]
        wherey = yy[wheretheta]
        pos_ang[i, :] = np.nansum(cube.hdu.data[:, wherey, wherex], axis=1)

    return np.rot90(pos_ang), f


# %%
def pa_plot(pos_ang, vlim, model=None, title=None, prefix=None,
            label=None):
    try:
        os.makedirs(os.path.join(OUTPATH, STAMP, 'HNC3_2_fit'))
    except FileExistsError:
        pass

#    logging.info("Plotting velocity vs position angle for {0}".format(prefix))

    plt.figure(figsize=(12, 4))
    plt.imshow(pos_ang,
               extent=[0, 360, vlim[0], vlim[1]],
               aspect='auto', cmap='jet', origin='upper')

    if model is not None:
        plt.plot(model[0], model[1], label=label, color='white', ls='--', lw=1)
        plt.legend()

    plt.xlabel('Position Angle East of North [Degrees]')
    plt.ylabel('$v_{r}$ [km/s]')
    plt.title(title)

#    plt.colorbar().set_label('Integrated Flux $(\\mathrm{Hz}\\,'
#                             '\\mathrm{Jy}/\\mathrm{beam})$')
    plt.savefig(os.path.join(OUTPATH, STAMP, 'HNC3_2_fit', '{0}_pa.{1}'
                             .format(prefix, FILETYPE)), bbox_inches='tight')


# %%
def plot_moment(cube, moment, prefix, title):
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
    try:
        os.makedirs(os.path.join(OUTPATH, STAMP, prefix))
    except FileExistsError:
        pass

    logging.info("Plotting moment {0} for {1}".format(moment, prefix))
    # only make file if it doesnt already exist
    filename = os.path.join(OUTPATH, STAMP, prefix, 'moment_{0}.{1}'
                            .format(moment, FILETYPE))
#    with Path(filename) as file:
#        if file.exists():
#            return

    z_lbl = ''
    if moment == 0:
        z_lbl = "Integrated Flux $(\\mathrm{Hz}\\," \
                "\\mathrm{Jy}/\\mathrm{beam})$"
        cube = cube.with_spectral_unit(u.Hz, velocity_convention='radio')
    elif moment == 1:
        z_lbl = "$v_{r} (\\mathrm{km}/\\mathrm{s})$"
    elif moment == 2:
        z_lbl = "$\\sigma_{v_{r}}^{2} (\\mathrm{km}^{2}/\\mathrm{s}^{2})$"
    else:
        print("Please choose from moment 0, 1, or 2")
        return

    # plot data
    f = aplpy.FITSFigure(cube.moment(order=moment).hdu, figsize=FIGSIZE)
    f.set_xaxis_coord_type('longitude')
    f.set_yaxis_coord_type('latitude')

    # add Sgr A*
    gc = ICRS(ra=Angle('17h45m40.0409s'),
              dec=Angle('-29:0:28.118 degrees')).transform_to(FK5)
    ra = gc.ra.value
    dec = gc.dec.value
    f.show_markers(ra, dec, layer='sgra', label='Sgr A*',
                   edgecolor='black', facecolor='black', marker='o', s=10)

    plt.legend()

    # add meta information
    f.ticks.show()
    f.add_beam(linestyle='solid', facecolor='white',
               edgecolor='black', linewidth=1)
    f.add_scalebar(((.5 * u.pc) / D_SGR_A * u.rad))
    f.scalebar.set_label('0.5 pc')

    f.show_colorscale()
    f.add_colorbar()
    f.colorbar.set_axis_label_text(z_lbl)

    f.set_title(title)

    f.save(filename)
    return


# %%
def corner_plot(samples, prange, fit, args):
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
    fig = corner.corner(samples,
                        labels=["$\\omega$", "$\\Omega$",
                                "$i$", "$r_p$", "$r_a$"],
                        quantiles=[.0014, .0227, .1587, .8414, .9773, .9987],
                        truths=fit)

    fig.set_size_inches(12, 12)

    plt.savefig(os.path.join(OUTPATH, STAMP, 'corner.pdf'
                             .format(args.WALKERS, args.NMAX)),
                bbox_inches='tight')


# %%
def plot_acor(acor):
    n = 100 * np.arange(1, len(acor) + 1)

    plt.figure(figsize=FIGSIZE)
    plt.plot(n, n / 100.0, "--k")
    plt.plot(n, acor)
    plt.xlim(0, n.max())
    plt.ylim(0, acor.max() + 0.1*(acor.max() - acor.min()))
    plt.xlabel("number of steps")
    plt.ylabel(r"mean $\hat{\tau}$")
    plt.savefig(os.path.join(OUTPATH, STAMP, 'acor.pdf'))


# %%
# =============================================================================
# Main program
# =============================================================================
def main(pool, args):
    """The main function of MC Orbit. Used to start all sampling

    This is the main function for MC Orbit. It carries out what is effectivley
    the entire study, including applying masks, initializing plots, and
    starting the actual MCMC process.

    """
    global STAMP
    STAMP = args.OUT

    # create output folder
    try:
        os.makedirs(os.path.join(OUTPATH, STAMP))
    except FileExistsError:
        pass

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=logging.INFO)
    logging.getLogger("astropy").setLevel(logging.WARNING)
#    logging.getLogger("aplpy.core").setLevel(logging.WARNING)
    logging.info("Starting analysis.")

    if args.PLOT:
        # load, plot, and garbage collect all tracers
        # HNC 3-2
        hnc3_2 = import_data(cubefile=os.path.join(os.path.dirname(__file__),
                                                   '..', 'dat',
                                                   'HNC3_2.fits'), clean=False)
        vmin = hnc3_2.spectral_axis.min().value
        vmax = hnc3_2.spectral_axis.max().value

        plot_moment(hnc3_2, moment=0, prefix='HNC3_2',
                    title="Integrated emission, HNC 3-2")
        pa_plot(pa_transform(hnc3_2)[0],
                [vmin, vmax], prefix='HNC3_2', title="HNC 3-2")

        # HCN 3-2
        hcn3_2 = import_data(cubefile=os.path.join(os.path.dirname(__file__),
                                                   '..', 'dat',
                                                   'HCN3_2.fits'), clean=False)
        logging.info("Reprojecting HNC 3-2.")
        hcn3_2 = hcn3_2.spectral_interpolate(hnc3_2.spectral_axis)
        hcn3_2 = hcn3_2.reproject(hnc3_2.header)

        plot_moment(hcn3_2, moment=0, prefix='HCN3_2',
                    title="Integrated emission, HCN 3-2")
        pa_plot(pa_transform(hcn3_2[0]),
                [vmin, vmax], prefix='HCN3_2', title="HCN 3-2")

        del hcn3_2
        gc.collect()

        # HCN 4-3
        hcn4_3 = import_data(cubefile=os.path.join(os.path.dirname(__file__),
                                                   '..', 'dat',
                                                   'HCN4_3.fits'), clean=False)
        logging.info("Reprojecting HNC 4-3.")
        hcn4_3 = hcn4_3.spectral_interpolate(hnc3_2.spectral_axis)
        hcn4_3 = hcn4_3.reproject(hnc3_2.header)

        plot_moment(hcn4_3, moment=0, prefix='HCN4_3',
                    title="Integrated emission, HCN 4-3")
        pa_plot(pa_transform(hcn4_3)[0], [vmin, vmax],
                prefix='HCN4_3', title="HCN 4-3")

        del hcn4_3
        gc.collect()

        # HCO+ 3-2
        hco3_2 = import_data(cubefile=os.path.join(os.path.dirname(__file__),
                                                   '..', 'dat',
                                                   'HCO+3_2.fits'),
                             clean=False)
        logging.info("Reprojecting HCO+ 3-2.")
        hco3_2 = hco3_2.spectral_interpolate(hnc3_2.spectral_axis)
        hco3_2 = hco3_2.reproject(hnc3_2.header)

        plot_moment(hco3_2, moment=0, prefix='HCO+3_2',
                    title="Integrated emission, HCO+ 3-2")
        pa_plot(pa_transform(hco3_2)[0], [vmin, vmax], prefix='HCO+3_2',
                title="HCO\\textsuperscript{+} 3-2")

        del hco3_2
        gc.collect()

        # SO 56-45
        so56_45 = import_data(cubefile=os.path.join(os.path.dirname(__file__),
                                                    '..', 'dat',
                                                    'SO56_45.fits'),
                              clean=False)
        logging.info("Reprojecting SO 56-45.")
        so56_45 = so56_45.spectral_interpolate(hnc3_2.spectral_axis)
        so56_45 = so56_45.reproject(hnc3_2.header)

        plot_moment(so56_45, moment=0, prefix='SO56_45',
                    title="Integrated emission, SO 56-45")
        pa_plot(pa_transform(so56_45)[0], [vmin, vmax], prefix='SO56_45', title="SO 56-45")

        del so56_45
        del hnc3_2
        gc.collect()

        # repeat, removing rms noise
        # HNC 3-2
        hnc3_2 = import_data(cubefile=os.path.join(os.path.dirname(__file__),
                                                   '..', 'dat',
                                                   'HNC3_2.fits'), clean=True)
        vmin = hnc3_2.spectral_axis.min().value
        vmax = hnc3_2.spectral_axis.max().value

        plot_moment(hnc3_2, moment=0, prefix='HNC3_2_rms',
                    title="Integrated emission, HNC 3-2, no rms")
        pa_plot(pa_transform(hnc3_2)[0], [vmin, vmax], prefix='HNC3_2_rms',
                title="HNC 3-2, no rms")

        # HCN 3-2
        hcn3_2 = import_data(cubefile=os.path.join(os.path.dirname(__file__),
                                                   '..', 'dat',
                                                   'HCN3_2.fits'), clean=True)
        logging.info("Reprojecting HNC 3-2.")
        hcn3_2 = hcn3_2.spectral_interpolate(hnc3_2.spectral_axis)
        hcn3_2 = hcn3_2.reproject(hnc3_2.header)

        plot_moment(hcn3_2, moment=0, prefix='HCN3_2_rms',
                    title="Integrated emission, HCN 3-2, no rms")
        pa_plot(pa_transform(hcn3_2)[0], [vmin, vmax], prefix='HCN3_2_rms',
                title="HCN 3-2, no rms")

        del hcn3_2
        gc.collect()

        # HCN 4-3
        hcn4_3 = import_data(cubefile=os.path.join(os.path.dirname(__file__),
                                                   '..', 'dat',
                                                   'HCN4_3.fits'), clean=True)
        logging.info("Reprojecting HNC 4-3.")
        hcn4_3 = hcn4_3.spectral_interpolate(hnc3_2.spectral_axis)
        hcn4_3 = hcn4_3.reproject(hnc3_2.header)

        plot_moment(hcn4_3, moment=0, prefix='HCN4_3_rms',
                    title="Integrated emission, HCN 4-3, no rms")
        pa_plot(pa_transform(hcn4_3)[0], [vmin, vmax], prefix='HCN4_3_rms',
                title="HCN 4-3, no rms")

        del hcn4_3
        gc.collect()

        # HCO+ 3-2
        hco3_2 = import_data(cubefile=os.path.join(os.path.dirname(__file__),
                                                   '..', 'dat',
                                                   'HCO+3_2.fits'), clean=True)
        logging.info("Reprojecting HCO+ 3-2.")
        hco3_2 = hco3_2.spectral_interpolate(hnc3_2.spectral_axis)
        hco3_2 = hco3_2.reproject(hnc3_2.header)

        plot_moment(hco3_2, moment=0, prefix='HCO+3_2_rms',
                    title="Integrated emission, HCO+ 3-2, no rms")
        pa_plot(pa_transform(hco3_2)[0], [vmin, vmax], prefix='HCO+3_2_rms',
                title="HCO\\textsuperscript{+} 3-2, no rms")

        del hco3_2
        gc.collect()

        # SO 56-45
        so56_45 = import_data(cubefile=os.path.join(os.path.dirname(__file__),
                                                    '..', 'dat',
                                                    'SO56_45.fits'),
                              clean=True)
        logging.info("Reprojecting SO 56-45.")
        so56_45 = so56_45.spectral_interpolate(hnc3_2.spectral_axis)
        so56_45 = so56_45.reproject(hnc3_2.header)

        plot_moment(so56_45, moment=0, prefix='SO56_45_rms',
                    title="Integrated emission, SO 56-45, no rms")
        pa_plot(pa_transform(so56_45)[0], [vmin, vmax], prefix='SO56_45_rms',
                title="SO 56-45, no rms")

        del so56_45
        gc.collect()

        # plot the next 2 moments of HNC 3-2 without rms
        plot_moment(hnc3_2, moment=1, prefix='HNC3_2_rms',
                    title="Line of sight velocity map, HNC 3-2")
        plot_moment(hnc3_2, moment=2, prefix='HNC3_2_rms',
                    title="Line of sight velocity variance map, HNC 3-2")

        del hnc3_2
        gc.collect()

    logging.info("Applying mask.")
    hnc3_2 = import_data(cubefile=os.path.join(os.path.dirname(__file__),
                                               '..', 'dat', 'HNC3_2.fits'),
                         maskfile=os.path.join(os.path.dirname(__file__),
                                               '..', 'dat',
                                               'HNC3_2.mask.south.fits'))
    vmin = hnc3_2.spectral_axis.min().value
    vmax = hnc3_2.spectral_axis.max().value
    logging.info("Mask complete.")

    try:
        pos_ang = np.load('pos_ang.south.npy')

        cube = hnc3_2.with_spectral_unit(u.Hz, velocity_convention='radio')
        m0 = cube.moment0()

        # position angle plot
        xx, yy = np.meshgrid(np.arange(0, m0.shape[1]),
                             np.arange(0, m0.shape[0]),
                             indexing='xy')

        # convenient functions for finding the location of Sgr A*
        f = aplpy.FITSFigure(m0.hdu, figsize=FIGSIZE)
        f.set_xaxis_coord_type('longitude')
        f.set_yaxis_coord_type('latitude')
    except FileNotFoundError:
        logging.info("Making position angles")
        pos_ang, f = pa_transform(hnc3_2)
        np.save('pos_ang.south.npy', pos_ang)

    wheredata = np.where(pos_ang.any(axis=0))
    min_pos_ang = np.min(wheredata)
    max_pos_ang = np.max(wheredata)

    if args.PLOT:
        # plot masked moments of HNC 3-2 and position angle plot
        plot_moment(hnc3_2, moment=0, prefix='HNC3_2_masked',
                    title="Integrated emission, HNC 3-2, masked")
        plot_moment(hnc3_2, moment=1, prefix='HNC3_2_masked',
                    title="Line of sight velocity map, HNC 3-2, masked")
        plot_moment(hnc3_2, moment=2, prefix='HNC3_2_masked',
                    title="Line of sight velocity variance map, "
                          "HNC 3-2, masked")
        pa_plot(pos_ang, [vmin, vmax], prefix='HNC3_2_masked',
                title="HNC 3-2, masked")

        # Plot Martin 2012 model
        martin_model = (90., 90., 30., 1.6, 1.6)
        plot_model(hnc3_2, 'HNC3_2_Martin', min_pos_ang, max_pos_ang,
                   params=martin_model,
                   label="Martin et al. 2012")
        martin_pa = pa_model(martin_model, f, min_pos_ang, max_pos_ang)
        pa_plot(pos_ang, [vmin, vmax], model=martin_pa, prefix='HNC3_2_Martin',
                label="Martin et al. 2012")

#    plot_moment(hnc3_2_masked, moment=1, prefix='HNC3_2_masked')
#    plot_moment(hnc3_2_masked, moment=2, prefix='HNC3_2_masked')

    if args.SAMPLE or args.VEFF or args.CORNER:
        logging.info("Preparing data.")
        data = ppv_pts(hnc3_2)
        logging.info("Data preparation complete.")

        if args.SUB != 1.:
            n_pts = len(data)
            ind = np.random.choice(range(n_pts), size=int(args.SUB * n_pts),
                                   replace=False)
            data = data[ind]

        # find the lower bounds on the peri and apoapses using apparent sep
        m1 = hnc3_2.moment1()
        dd, rr = m1.spatial_coordinate_map
        c = SkyCoord(ra=rr, dec=dd, radial_velocity=m1, frame='fk5')
        c = c.ravel()

        offset = np.array([((c.ra - GAL_CENTER.ra).rad * D_SGR_A).to(u.pc),
                           ((c.dec - GAL_CENTER.dec).rad * D_SGR_A).to(u.pc),
                           c.radial_velocity.to(u.pc / u.yr).value]).T

        offset = offset[_notnan(offset[:, 2])]

        # use lower bounds on peri/apoapsis to set lower bound on angular
        # momentum
        r_p_lb = np.min(np.sqrt(offset[:, 0] ** 2 + offset[:, 1] ** 2))
        r_a_lb = np.max(np.sqrt(offset[:, 0] ** 2 + offset[:, 1] ** 2))

        # our upper bound on the radius is determined by the position of
        # the furthest local maximum
        r_p_ub = 2. * r_a_lb
        r_a_ub = 10.  # * r_p_ub

    if args.VEFF:
        lmin = (r_p_lb * r_a_lb * np.sqrt((2 * (orbits.potential(r_a_lb)
                                                - orbits.potential(r_p_lb)))
                / ((r_a_lb ** 2) - (r_p_lb ** 2))))
        lmax = (r_p_ub * r_a_ub * np.sqrt((2 * (orbits.potential(r_a_ub)
                                                - orbits.potential(r_p_ub)))
                / ((r_a_ub ** 2) - (r_p_ub ** 2))))

        rr, ll = np.meshgrid(np.linspace(r_p_lb, 10., num=100),
                             np.linspace(lmin, lmax, num=100), indexing='ij')

        V_eff_r = orbits.V_eff(rr, ll)
        V_eff_r[V_eff_r > 0.] = np.nan

        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.gca(projection='3d')
        ax.plot_surface(rr, ll, V_eff_r)
        ax.set_xlabel('$r$')
        ax.set_ylabel('$l$')
        ax.set_zlabel('$V_{eff}$')
#        ax.set_zlim3d(top=0.)
        ax.set_zbound(upper=0.)
        plt.title("$V_{eff}$ vs. $r, l$")
        save_path = os.path.join(os.path.dirname(__file__), '..',
                                 'out', 'V_eff.pdf')
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        print(V_eff_r.shape)

    if args.SAMPLE or args.CORNER:
        # set up priors and do MCMC. angular momentum bounds are based on
        # the maximum radius
        p_aop = [-60., 60.]  # argument of periapsis
        p_loan = [60., 240.]  # longitude of ascending node
        p_inc = [60., 240.]  # inclination
        p_rp = [r_p_lb, r_p_ub]  # starting radial distance
        p_ra = [r_a_lb, r_a_ub]  # ang. mom.
        pspace = np.array([p_aop,
                           p_loan,
                           p_inc,
                           p_rp,
                           p_ra], dtype=np.float64)

    if args.SAMPLE:
        from mcorbit.model import ln_prob
        np.savetxt(os.path.join(OUTPATH, STAMP, 'pspace.csv'), pspace)

        logging.info("Sampling probability space.")
        sampler = mcmc.fit_orbits(pool, ln_prob, data, pspace,
                                  [min_pos_ang, max_pos_ang],
                                  nwalkers=args.WALKERS, nmax=args.NMAX,
                                  burn=args.BURN, reset=False, mpi=args.MPI,
                                  outpath=os.path.join(OUTPATH, STAMP))
    elif args.PLOT or args.CORNER:
        logging.info("Loading MCMC data")
        pspace = np.loadtxt(os.path.join(OUTPATH, STAMP, 'pspace.csv'))
        sampler = emcee.backends.HDFBackend(os.path.join(OUTPATH,
                                                         STAMP, 'chain.h5'))
        logging.info("MCMC data loaded")

    if args.PLOT or args.CORNER:
        logging.info("Analyzing MCMC data")
        tau = sampler.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        thin = int(.5 * np.min(tau))
        samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)

        # analyze the walker data
        aop, loan, inc, r_per, r_ap = map(lambda v: (v[1], v[2]-v[1],
                                                     v[1]-v[0]),
                                          zip(*np.quantile(samples,
                                                             [.1587, .5, .8414],
                                                             axis=0)))
        pbest = (aop[0],
                 loan[0],
                 inc[0],
                 r_per[0],
                 r_ap[0])
        # print the best parameters found and plot the fit
        logging.info("Best Fit: aop: {0}, loan: {1}, inc: {2}, "
                     "r_per: {3}, r_ap: {4}".format(*pbest))

        corner_plot(samples, pspace, pbest, args)

        np.savetxt(os.path.join(OUTPATH, STAMP, 'pbest.csv'), pbest)

        print("aop: {0:.2f} + {1:.2f} - {2:.2f}".format(*aop))
        print("loan: {0:.2f} + {1:.2f} - {2:.2f}".format(*loan))
        print("inc: {0:.2f} + {1:.2f} - {2:.2f}".format(*inc))
        print("r_per: {0:.2f} + {1:.2f} - {2:.2f}".format(*r_per))
        print("r_ap: {0:.2f} + {1:.2f} - {2:.2f}".format(*r_ap))

        ptest = (35., 135., 35. + 180., 2., 4.5)  # (30, 135, 215, 2, 4.5)
        label = 'Best Fit ($\\omega = {0:.2f}, \\Omega = {1:.2f}, ' \
                'i = {2:.2f}, r_p = {3:.2f}, r_a = {4:.2f}$)'.format(*pbest)
        prefix = 'HNC3_2_fit_{0}_{1}_{2}_{3}_{4}'.format(*pbest)
        plot_model(hnc3_2, prefix, pbest,
                   min_pos_ang, max_pos_ang, label=label)
        model = pa_model(pbest, f, min_pos_ang, max_pos_ang)
        pa_plot(pos_ang, [vmin, vmax], model=model, prefix=prefix, label=label)
#        from tqdm import tqdm
#        aop_range = np.linspace(-40, 70, num=5)
#        loan_range = np.linspace(130, 220, num=5)
#        inc_range = np.linspace(190, 230, num=5)
#        r_p_range = np.linspace(r_p_lb, r_p_ub, num=5)
#        r_a_range = np.linspace(r_a_lb, r_a_ub, num=5)
#        with tqdm(total=len(aop_range) * len(loan_range)
#                  * len(inc_range) * len(r_p_range) * len(r_a_range)) as pbar:
#            for aop in aop_range:
#                for loan in loan_range:
#                    for inc in inc_range:
#                        for r_p in r_p_range:
#                            for r_a in r_a_range:
#                                ptest = (aop, loan, inc, r_p, r_a)
#
#                                label = 'Best Fit ($\\omega = {0:.2f}, ' \
#                                        '\\Omega = {1:.2f}, ' \
#                                        'i = {2:.2f}, ' \
#                                        'r_p = {3:.2f}, r_a = {4:.2f}$)' \
#                                        .format(*ptest)
#
#                                prefix = 'HNC3_2_fit_{0}_{1}_{2}_{3}_{4}' \
#                                         .format(int(aop), int(loan),
#                                                 int(inc), r_p, r_a)
#                                plot_model(hnc3_2,
#                                           prefix, ptest,
#                                           min_pos_ang, max_pos_ang,
#                                           label=label)
#                                model = pa_model(ptest, f,
#                                                 min_pos_ang, max_pos_ang)
#                                pa_plot(pos_ang, [vmin, vmax], model=model,
#                                        prefix=prefix,
#                                        label=label)
#                                pbar.update(1)
        logging.info("Analysis complete")

    # bit of cleanup
    if not os.listdir(OUTPATH):
        os.rmdir(OUTPATH)


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 10})
    # Parse command line arguments
    PARSER = argparse.ArgumentParser()

    # Add command line flags
    PARSER.add_argument('-d', '--data_path', action='store',
                        type=str, dest='DATA_PATH', default="~/data/")
    PARSER.add_argument('--nmax', dest='NMAX', action='store',
                        help='maximum number of iterations',
                        default=1000, type=int)
    PARSER.add_argument('-w', '--walkers', dest='WALKERS', action='store',
                        help='number of walkers to use',
                        default=100, type=int)
    PARSER.add_argument('-b', '--burn', dest='BURN', action='store',
                        help='number of initial burn-in iterations',
                        default=1000, type=int)
    PARSER.add_argument('-s', '--sub', dest='SUB', action='store',
                        help='fraction of data points to use (random sample)',
                        default=1., type=float)
    PARSER.add_argument('-o', '--out', dest='OUT', action='store',
                        help='specific out path',
                        default='', type=str)
    PARSER.add_argument('--sample', dest='SAMPLE',
                        action='store_true', default=False)
    PARSER.add_argument('--plot', dest='PLOT',
                        action='store_true', default=False)
    PARSER.add_argument('--Veff', dest='VEFF',
                        action='store_true', default=False)
    PARSER.add_argument('--corner', dest='CORNER', action='store_true',
                        default=False)

    GROUP = PARSER.add_mutually_exclusive_group()
    GROUP.add_argument("--ncores", dest="NCORES", default=1,
                       type=int, help="Number of processes "
                       "(uses multiprocessing).")
    GROUP.add_argument("--mpi", dest="MPI", default=False,
                       action="store_true", help="Run with MPI.")
    args = PARSER.parse_args()

    pool = schwimmbad.choose_pool(mpi=args.MPI, processes=args.NCORES)

    main(pool, args)
