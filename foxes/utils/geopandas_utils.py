import numpy as np
import argparse
from glob import glob, has_magic
from typing import Any

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


def check_import_gpd() -> None:
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


def check_import_utm() -> None:
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


def read_shp(fname: str, **kwargs: Any) -> Any:
    """
    Read a shapefile file

    Parameters
    ----------
    fname
        Path to the .shp file
    kwargs
        Additional parameters for geopandas.read_file()

    Returns
    -------
    data
        The data frame in WSG84

    """
    check_import_gpd()
    gpdf = gpd.read_file(fname, **kwargs)
    return gpdf.to_crs("EPSG:4326")  # Convert to WGS84


def shp2csv(
    ifile: str,
    ofile: str,
    in_kwargs: dict[str, Any] | None = None,
    out_kwargs: dict[str, Any] | None = None,
    verbosity: int = 1,
) -> Any:
    """
    Read shapefile file, write csv file

    Parameters
    ----------
    ifile
        Path to the input .shp file
    ofile
        Path to the output .csv file
    in_kwargs
        Additional parameters for geopandas.read_file()
    out_kwargs
        Additional parameters for geopandas to_csv()
    verbosity
        The verbosity level, 0 = silent

    """
    if verbosity > 0:
        print("Reading file", ifile)

    gpdf = read_shp(ifile, **({} if in_kwargs is None else in_kwargs))

    if verbosity > 0:
        print("Writing file", ofile)

    gpdf.to_csv(ofile, **({} if out_kwargs is None else out_kwargs))

    return gpdf


def _ring_to_2d(coords: Any) -> np.ndarray:
    """
    Converts a single ring's coordinate sequence into a (N, 2) float array.

    Shapefiles may store Z (and M) coordinates. Matplotlib and the 2D geometry
    code expect planar (x, y) points only, so any extra coordinate columns are
    dropped here.

    Parameters
    ----------
    coords
        Sequence of coordinate tuples of a single ring

    Returns
    -------
    ring
        Array of shape (N, 2), possibly (0, 2) if empty

    """
    arr = np.asarray(coords, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2D ring coordinate array, got shape {arr.shape}. "
            "This usually indicates a nesting bug in the geometry extraction."
        )
    if arr.shape[1] < 2:
        raise ValueError(f"Ring has fewer than 2 coordinate columns: {arr.shape}")
    return np.ascontiguousarray(arr[:, :2])


def _extract_poly_coords(geom: Any) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Helper function for shapefile reading.

    Extracts exterior and interior rings from a (Multi)Polygon geometry.
    Both return values are always *flat* lists of (N, 2) arrays, regardless
    of the nesting depth of the input geometry. This is essential so that no
    part of a MultiPolygon gets lost or mis-sliced downstream.

    Parameters
    ----------
    geom
        A shapely Polygon, MultiPolygon or GeometryCollection

    Returns
    -------
    exterior_coords
        Flat list of exterior ring arrays, each of shape (N, 2)
    interior_coords
        Flat list of interior ring (hole) arrays, each of shape (N, 2)

    """
    exterior_coords: list[np.ndarray] = []
    interior_coords: list[np.ndarray] = []

    if geom is None or getattr(geom, "is_empty", False):
        return exterior_coords, interior_coords

    gtype = geom.geom_type

    if gtype == "Polygon":
        ext = _ring_to_2d(geom.exterior.coords[:])
        if len(ext):
            exterior_coords.append(ext)
        for interior in geom.interiors:
            ring = _ring_to_2d(interior.coords[:])
            if len(ring):
                interior_coords.append(ring)

    elif gtype in ("MultiPolygon", "GeometryCollection"):
        for part in geom.geoms:
            epe, epi = _extract_poly_coords(part)  # recursive call
            exterior_coords.extend(epe)  # extend, NOT append
            interior_coords.extend(epi)  # extend, NOT append

    else:
        raise ValueError(f"Unhandled geometry type: {gtype!r}")

    return exterior_coords, interior_coords


def _extract_utm(to_utm: bool | str) -> tuple[bool, int | None, str | None]:
    """
    Helper function for UTM zone parsing

    Parameters
    ----------
    to_utm
        Convert to UTM coordinates. If str, then UTM zone
        plus letter, e.g. "32U"

    Returns
    -------
    apply_utm
        Flag for UTM conversion
    utmz
        The forced UTM zone number, or None
    utml
        The forced UTM zone letter, or None

    """
    utmz: int | None = None
    utml: str | None = None
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


def _find_utm_zone(
    rings: list[np.ndarray],
    utmz: int | None = None,
    utml: str | None = None,
) -> tuple[int, str]:
    """
    Determines a single UTM zone for a whole set of rings.

    The zone is derived from the centroid of the bounding box of all points,
    so that geometries spanning a zone boundary are handled consistently
    instead of the zone being fixed by whichever ring happens to come first.

    Parameters
    ----------
    rings
        Flat list of (N, 2) lon/lat arrays
    utmz
        Forced UTM zone number, or None
    utml
        Forced UTM zone letter, or None

    Returns
    -------
    zone_number
        The UTM zone number
    zone_letter
        The UTM zone letter

    """
    if utmz is not None and utml is not None:
        return utmz, utml

    check_import_utm()

    non_empty = [r for r in rings if len(r)]
    if not len(non_empty):
        raise ValueError("No points found for UTM zone scouting")
    pts = np.concatenate(non_empty, axis=0)

    lon = 0.5 * (pts[:, 0].min() + pts[:, 0].max())
    lat = 0.5 * (pts[:, 1].min() + pts[:, 1].max())

    __, __, zone_number, zone_letter = utm.from_latlon(
        lat, lon, force_zone_number=utmz, force_zone_letter=utml
    )
    return int(zone_number), str(zone_letter)


def _rings_to_utm(rings: list[np.ndarray], utmz: int, utml: str) -> list[np.ndarray]:
    """
    Converts a flat list of lon/lat rings into UTM coordinates.

    Parameters
    ----------
    rings
        Flat list of (N, 2) lon/lat arrays
    utmz
        The UTM zone number
    utml
        The UTM zone letter

    Returns
    -------
    out
        Flat list of (N, 2) UTM easting/northing arrays

    """
    check_import_utm()

    out: list[np.ndarray] = []
    for ring in rings:
        if not len(ring):
            continue
        e, n, __, __ = utm.from_latlon(
            ring[:, 1],
            ring[:, 0],
            force_zone_number=utmz,
            force_zone_letter=utml,
        )
        out.append(np.stack([np.asarray(e), np.asarray(n)], axis=-1))
    return out


def read_shp_polygons(
    fname: str,
    names: Any = None,
    name_col: str = "Name",
    geom_col: str = "geometry",
    to_utm: bool | str = True,
    ret_utm_zone: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Reads the polygon points from a shp file.

    All rings of all matching features are collected. In particular,
    MultiPolygons are fully expanded and multiple rows sharing the same
    name are merged, so no sub-area is silently dropped.

    Parameters
    ----------
    fname
        Path to the .shp file
    names
        The names of the polygons to be extracted. All by
        default
    name_col
        Column that contains the area names
    geom_col
        The geometry column
    to_utm
        Convert to UTM coordinates. If str, then UTM zone
        plus letter, e.g. "32U"
    ret_utm_zone
        Return UTM zone plus letter as str
    kwargs
        Additional parameters for geopandas.read_shp()

    Returns
    -------
    point_dict_exterior
        Mapping from area names to lists of (N, 2) exterior ring arrays
    point_dict_interior
        Mapping from area names to lists of (N, 2) interior ring arrays
    utm_zone_str
        The UTM zone plus letter as str, e.g. "32U"

    """
    pdf = read_shp(fname, **kwargs)

    if name_col in pdf.columns:
        pnames = list(pdf[name_col])
    else:
        # fall back to positional names if the column is missing
        pnames = [f"area_{i}" for i in range(len(pdf))]

    apply_utm, force_z, force_l = _extract_utm(to_utm)
    if apply_utm:
        check_import_utm()

    # select the requested names, preserving order and skipping nan:
    if names is None:
        sel_names = []
        for n in pnames:
            if n == n and n not in sel_names:  # n == n excludes nan
                sel_names.append(n)
    else:
        sel_names = [n for n in names if n == n]
        for n in sel_names:
            if n not in pnames:
                raise KeyError(
                    f"Name '{n}' not found in file '{fname}'. Names: {pnames}"
                )

    # collect all rings per name, over ALL matching rows:
    raw_ext: dict[Any, list[np.ndarray]] = {}
    raw_int: dict[Any, list[np.ndarray]] = {}
    for name in sel_names:
        ext_rings: list[np.ndarray] = []
        int_rings: list[np.ndarray] = []
        for i, pn in enumerate(pnames):
            if pn != name:
                continue
            geom = pdf.iloc[i][geom_col]
            epe, epi = _extract_poly_coords(geom)
            ext_rings.extend(epe)
            int_rings.extend(epi)
        raw_ext[name] = ext_rings
        raw_int[name] = int_rings

    exterior: Any = Dict()
    interior: Any = Dict()
    utm_zone: str | None = None

    if apply_utm:
        # determine one common UTM zone for the whole file:
        all_rings: list[np.ndarray] = []
        for rings in raw_ext.values():
            all_rings.extend(rings)
        for rings in raw_int.values():
            all_rings.extend(rings)

        zone_number, zone_letter = _find_utm_zone(all_rings, force_z, force_l)
        utm_zone = f"{zone_number}{zone_letter}"

        for name in sel_names:
            exterior[name] = _rings_to_utm(raw_ext[name], zone_number, zone_letter)
            interior[name] = _rings_to_utm(raw_int[name], zone_number, zone_letter)
    else:
        for name in sel_names:
            exterior[name] = raw_ext[name]
            interior[name] = raw_int[name]

    if ret_utm_zone:
        return exterior, interior, utm_zone
    else:
        return exterior, interior


def shp2geom2d(
    shp_files: Any,
    *args: Any,
    combine_mode: str = "union",
    to_utm: bool | str = True,
    ret_utm_zone: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Read shapefile into geom2d geometry

    Parameters
    ----------
    shp_files
        Path to a ``.shp`` file or glob pattern matching multiple
        ``.shp`` files
    args
        Additional positional arguments for read_shp_polygons()
    combine_mode
        The combination mode for multiple areas. Options:
        ``"union"`` (default), ``"intersection"``
    to_utm
        Convert to UTM coordinates. If str, then UTM zone
        plus letter, e.g. "32U"
    ret_utm_zone
        Return UTM zone plus letter as str
    kwargs
        Keyword arguments for read_shp_polygons()

    Returns
    -------
    geom
        The geometry object
    utm_zone_str
        The UTM zone plus letter as str, e.g. "32U"

    """
    if "to_utm" in kwargs:
        to_utm = kwargs.pop("to_utm")
    if "ret_utm_zone" in kwargs:
        ret_utm_zone = kwargs.pop("ret_utm_zone")

    if combine_mode not in ["union", "intersection"]:
        raise ValueError(
            f"Invalid combine_mode '{combine_mode}', expected 'union' or 'intersection'"
        )

    def _expand_files(path_spec: Any) -> list[str]:
        s = str(path_spec)
        return sorted(glob(s)) if has_magic(s) else [s]

    if isinstance(shp_files, (list, tuple)):
        fnames: list[str] = []
        for f in shp_files:
            fnames.extend(_expand_files(f))
    else:
        fnames = _expand_files(shp_files)

    if not len(fnames):
        raise FileNotFoundError(f"No files matched '{shp_files}'")

    apply_utm, force_z, force_l = _extract_utm(to_utm)

    # first pass: read everything in lon/lat, so that a single UTM zone
    # can be determined across all files consistently
    raw: list[tuple[Any, Any]] = []
    for f in fnames:
        read_kwargs = dict(kwargs)
        read_kwargs["to_utm"] = False
        read_kwargs["ret_utm_zone"] = False
        raw.append(read_shp_polygons(f, *args, **read_kwargs))

    utm_zone: str | None = None
    data: list[tuple[Any, Any]]

    if apply_utm:
        all_rings: list[np.ndarray] = []
        for ext, int_ in raw:
            for rings in ext.values():
                all_rings.extend(rings)
            for rings in int_.values():
                all_rings.extend(rings)

        zone_number, zone_letter = _find_utm_zone(all_rings, force_z, force_l)
        utm_zone = f"{zone_number}{zone_letter}"

        data = []
        for ext, int_ in raw:
            ext_utm: Any = Dict()
            int_utm: Any = Dict()
            for k, v in ext.items():
                ext_utm[k] = _rings_to_utm(v, zone_number, zone_letter)
            for k, v in int_.items():
                int_utm[k] = _rings_to_utm(v, zone_number, zone_letter)
            data.append((ext_utm, int_utm))
    else:
        data = raw

    def _combine(gs: list[Any], mode: str) -> Any:
        gs = [g for g in gs if g is not None]
        if not len(gs):
            return None
        if len(gs) == 1:
            return gs[0]
        return AreaUnion(gs) if mode == "union" else AreaIntersection(gs)

    def _create_geom(obj: Any, mode: str) -> Any:
        """
        Recursively builds a geometry from nested dicts / lists / arrays.
        """
        if obj is None:
            return None
        if isinstance(obj, np.ndarray):
            if obj.ndim == 2 and obj.shape[0] >= 3:
                return ClosedPolygon(obj)
            return None
        if isinstance(obj, dict):
            return _combine([_create_geom(g, mode) for g in obj.values()], mode)
        if isinstance(obj, (list, tuple)):
            if not len(obj):
                return None
            return _combine([_create_geom(g, mode) for g in obj], mode)
        raise ValueError(f"Cannot build geometry from type {type(obj).__name__}")

    # exterior rings of one area are always a union of its parts:
    # the combine_mode applies across areas/files, not within a MultiPolygon.
    gs_ext: list[Any] = []
    for ext, __ in data:
        for rings in ext.values():
            g = _create_geom(rings, "union")
            if g is not None:
                gs_ext.append(g)
    gext = _combine(gs_ext, combine_mode)

    # interior rings (holes) are always unioned before subtraction:
    gs_int: list[Any] = []
    for __, int_ in data:
        for rings in int_.values():
            g = _create_geom(rings, "union")
            if g is not None:
                gs_int.append(g)
    gint = _combine(gs_int, "union")

    if gext is None:
        raise ValueError(f"No exterior polygons found in {fnames}")

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
