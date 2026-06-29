import numpy as np
import argparse
from glob import glob, has_magic

from .dict import Dict
from .geom2d import AreaUnion, AreaIntersection, ClosedPolygon

try:
    import geopandas as gpd

    IMPORT_GPD_OK = True
except ImportError:
    gpd = None
    IMPORT_GPD_OK = False

try:
    import utm

    IMPORT_UTM_OK = True
except ImportError:
    utm = None
    IMPORT_UTM_OK = False


def check_import_gpd():
    """
    Checks if library import worked,
    raises error otherwise.
    """
    if not IMPORT_GPD_OK:
        print("\n\nFailed to import geopandas. Please install, either via pip:\n")
        print("  pip install geopandas\n")
        print("or via conda:\n")
        print("  conda install -c conda-forge geopandas\n")
        raise ImportError("Failed to import geopandas")


def check_import_utm():
    """
    Checks if library import worked,
    raises error otherwise.
    """
    if not IMPORT_UTM_OK:
        print("\n\nFailed to import utm. Please install, either via pip:\n")
        print("  pip install utm\n")
        print("or via conda:\n")
        print("  conda install -c conda-forge utm\n")
        raise ImportError("Failed to import utm")


def read_shp(fname, **kwargs):
    """
    Read a shapefile file

    Parameters
    ----------
    fname: str
        Path to the .shp file
    kwargs: dict, optional
        Additional parameters for geopandas.read_file()

    Returns
    -------
    data: geopandas.GeoDataFrame
        The data frame in WSG84

    :group: utils

    """
    check_import_gpd()
    gpdf = gpd.read_file(fname, **kwargs)
    return gpdf.to_crs("EPSG:4326")  # Convert to WGS84


def shp2csv(ifile, ofile, in_kwargs={}, out_kwargs={}, verbosity=1):
    """
    Read shapefile file, write csv file

    Parameters
    ----------
    iname: str
        Path to the input .shp file
    oname: str
        Path to the output .csv file
    in_kwargs: dict
        Additional parameters for geopandas.read_file()
    out_kwargs: dict
        Additional parameters for geopandas to_csv()
    verbosity: int
        The verbosity level, 0 = silent

    :group: utils

    """
    if verbosity > 0:
        print("Reading file", ifile)

    gpdf = read_shp(ifile, **in_kwargs)

    if verbosity > 0:
        print("Writing file", ofile)

    gpdf.to_csv(ofile, **out_kwargs)

    return gpdf


def _extract_poly_coords(geom):
    """
    Helper function for shapefile reading
    """
    if geom.geom_type == "Polygon":
        exterior_coords = geom.exterior.coords[:]
        interior_coords = []
        for interior in geom.interiors:
            interior_coords.append(interior.coords[:])
    elif geom.geom_type == "MultiPolygon":
        exterior_coords = []
        interior_coords = []
        for part in geom.geoms:
            epe, epi = _extract_poly_coords(part)  # Recursive call
            exterior_coords.append(epe)
            interior_coords.append(epi)
    else:
        raise ValueError("Unhandled geometry type: " + repr(geom.type))
    return exterior_coords, interior_coords


def read_shp_polygons(
    fname,
    names=None,
    name_col="Name",
    geom_col="geometry",
    to_utm=True,
    ret_utm_zone=False,
    **kwargs,
):
    """
    Reads the polygon points from a shp file.

    Parameters
    ----------
    fname: str
        Path to the .shp file
    names: list: of str, optinal
        The names of the polygons to be extracted. All by
        default
    name_col: int
        Column that contains the area names
    geom_col: str
        The geometry column
    to_utm: bool or str, optional
        Convert to UTM coordinates. If str, then UTM zone
        plus letter, e.g. "32U"
    ret_utm_zone: bool
        Return UTM zone plus letter as str
    kwargs: dict, optional
        Additional parameters for geopandas.read_shp()

    Returns
    -------
    point_dict_exterior: dict
        Dict with list of array of points. Key: area name,
        Value: list:np.ndarray, shape of latter: (n_points, 2)
    point_dict_interior: dict
        Dict with list of array of points. Key: area name,
        Value: list:np.ndarray, shape of latter: (n_points, 2)
    utm_zone_str: str, optional
        The utem zone plus letter as str, e.g. "32U"

    :group: utils

    """

    pdf = read_shp(fname, **kwargs)
    pnames = list(pdf[name_col])

    utmz = None
    utml = None
    apply_utm = False
    if isinstance(to_utm, str) or to_utm:
        apply_utm = True
        check_import_utm()
        utmz = int(to_utm[:-1]) if isinstance(to_utm, str) else None
        utml = to_utm[-1] if isinstance(to_utm, str) else None

    exterior = Dict()
    interior = Dict()
    names = pnames if names is None else names
    for name in names:
        if name == name:  # exclude nan values
            if name not in pnames:
                raise KeyError(
                    f"Name '{name}' not found in file '{fname}'. Names: {pnames}"
                )

            a = pdf.loc[pnames.index(name), geom_col]
            epe, epi = _extract_poly_coords(a)

            def _to_utm(poly):
                nonlocal utmz, utml
                utm_poly = np.zeros_like(poly)
                utm_poly[:, 0], utm_poly[:, 1], utmz, utml = utm.from_latlon(
                    poly[:, 1],
                    poly[:, 0],
                    force_zone_number=utmz,
                    force_zone_letter=utml,
                )
                return utm_poly

            def _to_numpy(data):
                if not len(data):
                    return []
                if isinstance(data[0], tuple):
                    out = np.array(data, dtype=np.float64)
                    return _to_utm(out) if apply_utm else out
                return [_to_numpy(d) for d in data]

            exterior[name] = _to_numpy(epe)
            interior[name] = _to_numpy(epi)

    if ret_utm_zone:
        return exterior, interior, f"{utmz}{utml}"
    else:
        return exterior, interior


def shp2geom2d(*args, combine_mode="union", ret_utm_zone=False, **kwargs):
    """
    Read shapefile into geom2d geometry

    Parameters
    ----------
    args: tuple, optional
        Arguments for read_shp_polygons(). If the first argument is
        a glob pattern, all matched ``.shp`` files are read and
        combined into a union geometry
    combine_mode: str
        The combination mode for multiple areas. Options:
        ``"union"`` (default), ``"intersection"``
    ret_utm_zone: bool
        Return UTM zone plus letter as str
    kwargs: dict, optional
        Keyword arguments for read_shp_polygons()

    Returns
    -------
    geom: foxes.tools.geom2D.AreaGeometry
        The geometry object
    utm_zone_str: str, optional
        The utem zone plus letter as str, e.g. "32U"

    :group: utils

    """

    if combine_mode not in ["union", "intersection"]:
        raise ValueError(
            f"Invalid combine_mode '{combine_mode}', expected 'union' or 'intersection'"
        )

    def _combine(gs, mode):
        gs = [g for g in gs if g is not None]
        if not len(gs):
            return None
        if len(gs) == 1:
            return gs[0]
        return AreaUnion(gs) if mode == "union" else AreaIntersection(gs)

    if len(args) and isinstance(args[0], str) and has_magic(args[0]):
        if ret_utm_zone:
            raise ValueError("ret_utm_zone=True is not supported for glob patterns")

        fnames = sorted(glob(args[0]))
        if not len(fnames):
            raise FileNotFoundError(f"No files matched glob pattern '{args[0]}'")

        geoms = []
        for f in fnames:
            g = shp2geom2d(
                f,
                *args[1:],
                combine_mode=combine_mode,
                ret_utm_zone=False,
                **kwargs,
            )
            if g is not None:
                geoms.append(g)

        if not len(geoms):
            raise ValueError(f"No geometries created for glob pattern '{args[0]}'")

        return _combine(geoms, combine_mode)

    exint = read_shp_polygons(*args, ret_utm_zone=ret_utm_zone, **kwargs)

    def _create_geom(data, mode):
        if not len(data):
            return None
        if isinstance(data, dict):
            gs = [_create_geom(g, mode) for g in data.values()]
            return _combine(gs, mode)
        if isinstance(data, np.ndarray) and len(data.shape) == 2:
            return ClosedPolygon(data)
        gs = [_create_geom(g, mode) for g in data]
        return _combine(gs, mode)

    gext = _create_geom(exint[0], combine_mode)
    # Keep interior rings combined as union before subtraction.
    gint = _create_geom(exint[1], "union")
    geom = gext - gint if gint is not None else gext

    if ret_utm_zone:
        return geom, exint[2]
    else:
        return geom


if __name__ == "__main__":
    # define arguments and options:
    parser = argparse.ArgumentParser()
    parser.add_argument("shp_file", help="The input .shp file")
    parser.add_argument("-n", "--names", help="Area names", default=None, nargs="+")
    parser.add_argument(
        "--no_utm", help="switch off conversion to UTM", action="store_true"
    )
    args = parser.parse_args()

    g = shp2geom2d(args.shp_file, to_utm=not args.no_utm, names=args.names)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    g.add_to_figure(ax)
    plt.show()
    plt.close(fig)
