import numpy as np
import utm

def to_lonlat(xy):
    """
    Convert UTM coordinates to longitude and latitude.

    Parameters
    ----------
    xy: array_like
        The cartesian coordinates, shape: (n_points, 2), where
        the last dimension is (x, y).

    Returns
    -------
    lonlat: numpy.ndarray
        The longitude and latitude coordinates, shape: (n_points, 2), where
        the last dimension is (lon, lat).

    :group: utils

    """
    from foxes.config.config import config
    utmz, utml = config.utm_zone
    lat, lon = utm.to_latlon(
        xy[:, 0],
        xy[:, 1],
        utmz,
        utml,
    )
    return np.stack((lon, lat), axis=-1)

def from_lonlat(lonlat):
    """
    Convert longitude and latitude to UTM coordinates.

    Parameters
    ----------
    lonlat: array_like
        The longitude and latitude coordinates, shape: (n_points, 2), where
        the last dimension is (lon, lat).

    Returns
    -------
    xy: numpy.ndarray
        The cartesian coordinates, shape: (n_points, 2), where
        the last dimension is (x, y).
    
    :group: utils

    """
    from foxes.config.config import config
    utmz, utml = config.utm_zone
    x, y, __, __ = utm.from_latlon(
        lonlat[:, 1],
        lonlat[:, 0],
        utmz,
        utml,
    )
    return np.stack((x, y), axis=-1)

def get_utm_zone(lonlat):
    """
    Get the UTM zone for given latitude and longitude.

    Parameters
    ----------
    lonlat: array_like
        The latitude and longitude coordinates, shape: (n_points, 2), where
        the last dimension is (lat, lon).

    Returns
    -------
    utm_zone: tuple
        The UTM zone as (zone_number, zone_letter).

    :group: utils

    """
    lat = lonlat[:, 1]
    lon = lonlat[:, 0]
    return utm.from_latlon(lat, lon)[2:4]
