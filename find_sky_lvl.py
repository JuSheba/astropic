import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import LogNorm

import sky_background as skyb
from zscale import zscale

# менять для смены фильтра только это
FILTER = "z"

# from https://www.legacysurvey.org/dr9/description/:
ZERO_POINTS = {
    "g": 22.5,
    "r": 22.5,
    "z": 22.5,
}
# if known, optional:
ABSORP_COEFS = {
    "g": 0.0,
    "r": 0.0,
    "z": 0.0
}

GAL_NAME = 'EON_56.401_-7.472'


def show(data):
    yLen, xLen = data.shape
    fig, ax = plt.subplots()
    vmin = max(0.001, np.mean(data) - 3 * np.std(data))
    vmax = max(0.001, np.mean(data) + 3 * np.std(data))
    vmin, vmax = zscale(data)
    im = ax.imshow(data, cmap=plt.cm.bone, origin='lower',
                   norm=LogNorm(vmin=vmin, vmax=vmax), )
    ax.set_title("Show")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('value', rotation=-90, va="bottom")
    plt.show()


def get_gal_data(gal_name):
    """
    Find and open fits file in directory.

    :param gal_name: Object name with this format EON_{RA}_{DEC}
    :type gal_name: string
    :return: Data with surface brightness  as linear fluxes in units of nanomaggies
    :rtype: numpy 3d arraynanomaggies
    :rtype: numpy 3d array
    """
    data_dir_path = pl.Path(f'{gal_name}')
    gal_path = data_dir_path / f'{gal_name}.fits'
    gal_fits = fits.open(gal_path)[0]
    gal_data = gal_fits.data
    return gal_data


def get_galdata_fltrs(gal_data):
    """
    Divide the array into filters.

    :param gal_data: Data with surface brightness  as linear fluxes
                     in units of nanomaggies
    :type gal_data: numpy 3d array
    :return: Data with surface brightness for each filter
    :rtype: numpy 2d array
    """
    gal_data_g = gal_data[0, :, :]
    gal_data_r = gal_data[1, :, :]
    gal_data_z = gal_data[2, :, :]
    return gal_data_g, gal_data_r, gal_data_z


def get_keys_filters(gal_data_g, gal_data_r, gal_data_z):
    keys_filters = {
        "g": gal_data_g,
        "r": gal_data_r,
        "z": gal_data_z
    }
    return keys_filters


def remove_negative_val(gal_data):
    gal_data[gal_data < 0.001] = 0.001
    return gal_data


def get_interpol_sky_bg(gal_data):
    """
    Interpolate of the sky background in the sky_background module and saving to file.

    :param gal_data: Data with surface brightness for one filter
    :type gal_data: numpy 2d array
    :return: Data with sky background
    :rtype: ???
    """
    sky_back = skyb.getSkyBackground(gal_data, sample=10, k=(1, 1))
    plt.matshow(sky_back)

    skyb.saveSky(sky_back, name="skyBack")
    sky_back = skyb.loadSky(name="skyBack")
    return sky_back


def remove_sky_bg(gal_data, sky_back):
    """
    Remove interpolated sky background from galaxy data.

    :param gal_data: Data with surface brightness for one filter
    :type gal_data: numpy 2d array
    :param sky_back: Data with sky background
    :type sky_back: ???
    :return: Data with surface brightness for one filter minus the sky sky_background
    :rtype: numpy 2d array
    """
    gal_data = gal_data - sky_back
    return gal_data


def convert_magnitudes(gal_data, zero_points, photo_filter):
    """
    Convert from linear fluxes to magnitudes. See the section Photometry at the link:
    https://www.legacysurvey.org/dr9/description/#id9

    :param gal_data: Data with surface brightness for one filter
    :type gal_data: numpy 2d array
    :param zero_points: The reference point for the {photo_filter} key
    :type zero_points: dict
    :param photo_filter: Filter name
    :type photo_filter: str
    :return: Data in magnitudes for one filter
    :rtype: numpy 2d array
    """
    gal_data = zero_points[photo_filter] - 2.5 * np.log10(gal_data)
    return gal_data


def minus_absorption(gal_data, absorp_coefs, photo_filter, zenith_angle):
    """
    Minus the absorption by the atmosphere.

    :param gal_data: Data in magnitudes for one filter
    :type gal_data: numpy 2d array
    :param absorp_coefs: Absorption coefficient for the {photo_filter} key
    :type absorp_coefs: dict
    :param photo_filter: Filter name
    :type photo_filter: str
    :return: Data in magnitudes for one filter
    :rtype: numpy 2d array
    """
    gal_data = gal_data - absorp_coefs[photo_filter] / np.cos(zenith_angle * np.pi / 180)
    return gal_data


def remove_nans(gal_data, zero_points, photo_filter):
    """
    Remove nan and inf+, inf- values.

    :param gal_data: Data in magnitudes for one filter
    :type gal_data: numpy 2d array
    :param zero_points: The reference point for the {photo_filter} key
    :type zero_points: dict
    :param photo_filter: Filter name
    :type photo_filter: str
    :return: Data in magnitudes for one filter
    :rtype: numpy 2d array
    """
    gal_data = np.nan_to_num(gal_data, copy=False, nan=zero_points[photo_filter],
                             posinf=None, neginf=None)
    return gal_data


def show_isophots(data, v=(22, 16.5), limits=(16.5, 22), count=15):
    fig, ax = plt.subplots()
    vmin, vmax = v
    im = ax.imshow(data, cmap=plt.cm.bone, origin='lower',
                   norm=LogNorm(vmin=vmin, vmax=vmax), )
    ax.set_title("Show Isophots")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('value', rotation=-90, va="bottom")
    ax.contour(data, levels=np.linspace(
        limits[0], limits[1], count), colors='black', alpha=0.5)
    plt.show()


def main():
    gal_data = get_gal_data(GAL_NAME)
    gal_data_g, gal_data_r, gal_data_z = get_galdata_fltrs(gal_data)
    keys_filters = get_keys_filters(gal_data_g, gal_data_r, gal_data_z)

    gal_data_filter = keys_filters[FILTER]

    gal_data_filter = remove_negative_val(gal_data_filter)
    sky_back = get_interpol_sky_bg(gal_data_filter)
    gal_data_filter = remove_sky_bg(gal_data_filter, sky_back)
    show(gal_data_filter)

    gal_data_filter = convert_magnitudes(gal_data_filter, ZERO_POINTS, FILTER)
    gal_data_filter = minus_absorption(gal_data_filter, ABSORP_COEFS, FILTER, zenith_angle=0)
    gal_data_filter = remove_nans(gal_data_filter, ZERO_POINTS, FILTER)
    show_isophots(gal_data_filter)


if __name__ == "__main__":
    main()
