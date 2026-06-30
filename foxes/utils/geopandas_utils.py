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


def _extract_utm(to_utm):
    """
    Helper function for UTM zone parsing
    """
    utmz = None
    utml = None
    apply_utm = False
    if isinstance(to_utm, str) or to_utm:
        utmz = int(to_utm[:-1]) if isinstance(to_utm, str) else None
        utml = to_utm[-1] if isinstance(to_utm, str) else None
        apply_utm = True
    if apply_utm:
        if (utmz is not None and utml is None) or (utmz is None and utml is not None):
            raise ValueError(
                f"Invalid UTM zone specification '{to_utm}', "
                "must be either both zone number and letter or neither, "
                f"got: zone number = {utmz}, zone letter = {utml}"
            )
    return apply_utm, utmz, utml


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
    names: list of str, optional
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
        The UTM zone plus letter as str, e.g. "32U"

    :group: utils

    """

    pdf = read_shp(fname, **kwargs)
    pnames = list(pdf[name_col])
    apply_utm, utmz, utml = _extract_utm(to_utm)
    if apply_utm:
        check_import_utm()

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
        utm_zone = f"{utmz}{utml}" if utmz is not None and utml is not None else None
        return exterior, interior, utm_zone
    else:
        return exterior, interior


def shp2geom2d(
    shp_files,
    *args,
    combine_mode="union",
    to_utm=True,
    ret_utm_zone=False,
    **kwargs,
):
    """
    Read shapefile into geom2d geometry

    Parameters
    ----------
    shp_files: str
        Path to a ``.shp`` file or glob pattern matching multiple
        ``.shp`` files
    args: tuple, optional
        Additional positional arguments for read_shp_polygons()
    combine_mode: str
        The combination mode for multiple areas. Options:
        ``"union"`` (default), ``"intersection"``
    to_utm: bool or str, optional
        Convert to UTM coordinates. If str, then UTM zone
        plus letter, e.g. "32U"
    ret_utm_zone: bool
        Return UTM zone plus letter as str
    kwargs: dict, optional
        Keyword arguments for read_shp_polygons()

    Returns
    -------
    geom: foxes.tools.geom2D.AreaGeometry
        The geometry object
    utm_zone_str: str, optional
        The UTM zone plus letter as str, e.g. "32U"

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

    def _expand_files(path_spec):
        s = str(path_spec)
        return glob(s) if has_magic(s) else [s]

    if isinstance(shp_files, (list, tuple)):
        fnames = []
        for f in shp_files:
            fnames.extend(_expand_files(f))
    else:
        fnames = _expand_files(shp_files)

    if not len(fnames):
        raise FileNotFoundError(f"No files matched '{shp_files}'")

    # case one area only:
    if len(fnames) == 1:
        if ret_utm_zone:
            data, utm_zone = read_shp_polygons(
                fnames[0], *args, to_utm=to_utm, ret_utm_zone=True, **kwargs
            )
            data = [data]
        else:
            data = [
                read_shp_polygons(
                    fnames[0], *args, to_utm=to_utm, ret_utm_zone=False, **kwargs
                )
            ]
            utm_zone = None

    # case multiple areas:
    else:
        apply_utm, utmz, utml = _extract_utm(to_utm)
        utm_zone = False
        if apply_utm and utmz is not None and utml is not None:
            utm_zone = f"{utmz}{utml}"

        data = []
        for f in fnames:
            data.append(
                read_shp_polygons(
                    f, *args, to_utm=utm_zone, ret_utm_zone=False, **kwargs
                )
            )

        # auto determine UTM zone and apply:
        if to_utm == True and utm_zone == False:
            pts = []
            for d in data:
                if d[0] is not None:
                    pts += list(d[0].values())
                if d[1] is not None:
                    pts += list(d[1].values())
            pts = [p for p in pts if len(p) > 0]
            assert len(pts) > 0, "No points found for UTM zone scouting"
            pts = np.concatenate(pts, axis=0)

            check_import_utm()
            __, __, utmz, utml = utm.from_latlon(
                pts[:, 1],
                pts[:, 0],
                force_zone_number=utmz,
                force_zone_letter=utml,
            )
            utm_zone = f"{utmz}{utml}"
            del pts

            for d in data:
                if d[0] is not None:
                    for k in d[0].keys():
                        if len(d[0][k]) > 0:
                            d[0][k] = np.array(
                                utm.from_latlon(
                                    d[0][k][:, 1],
                                    d[0][k][:, 0],
                                    force_zone_number=utmz,
                                    force_zone_letter=utml,
                                )[:2]
                            ).T
                if d[1] is not None:
                    for k in d[1].keys():
                        if len(d[1][k]) > 0:
                            d[1][k] = np.array(
                                utm.from_latlon(
                                    d[1][k][:, 1],
                                    d[1][k][:, 0],
                                    force_zone_number=utmz,
                                    force_zone_letter=utml,
                                )[:2]
                            ).T

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

    gext = _create_geom([d[0] for d in data], combine_mode)
    # Keep interior rings combined as union before subtraction.
    gint = _create_geom([d[1] for d in data], "union")
    geom = gext - gint if gint is not None else gext

    if ret_utm_zone:
        return geom, utm_zone
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
