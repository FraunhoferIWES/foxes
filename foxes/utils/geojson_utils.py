import json
from pathlib import Path

import numpy as np

from foxes.utils.geom2d import AreaUnion, ClosedPolygon
from foxes.utils.utm_utils import from_lonlat


def area_contains_point(area, point):
    """
    Checks if a point lies in an area geometry.

    Parameters
    ----------
    area: object
        Area-like object exposing contains_point(point) or
        points_inside(points)
    point: array_like
        The point coordinates, shape: (2,)

    Returns
    -------
    inside: bool
        True if the point is inside area

    :group: utils

    """
    if hasattr(area, "contains_point"):
        return bool(area.contains_point(point))
    if hasattr(area, "points_inside"):
        return bool(area.points_inside(np.asarray(point)[None, :])[0])
    raise TypeError(
        "Expected area with 'contains_point' or 'points_inside', "
        f"got '{type(area).__name__}'"
    )


def geojson_geometry_to_area(geometry):
    """
    Converts one GeoJSON geometry object into an AreaGeometry.

    Parameters
    ----------
    geometry: dict
        A GeoJSON geometry dictionary

    Returns
    -------
    area: foxes.utils.geom2d.AreaGeometry or None
        The area geometry or None for empty polygon coordinate arrays

    :group: utils

    """
    gtype = geometry.get("type", None)

    def _polygon_with_holes(rings):
        if not rings:
            return None

        ext = np.asarray(rings[0], dtype=np.float64)
        if ext.ndim != 2 or ext.shape[1] < 2:
            raise ValueError("Invalid polygon ring in GeoJSON")
        geom = ClosedPolygon(from_lonlat(ext[:, :2]))

        for ring in rings[1:]:
            hole = np.asarray(ring, dtype=np.float64)
            if hole.ndim != 2 or hole.shape[1] < 2:
                raise ValueError("Invalid polygon hole ring in GeoJSON")
            geom = geom - ClosedPolygon(from_lonlat(hole[:, :2]))

        return geom

    if gtype == "Polygon":
        return _polygon_with_holes(geometry.get("coordinates", []))

    if gtype == "MultiPolygon":
        geoms = [_polygon_with_holes(rings) for rings in geometry.get("coordinates", [])]
        geoms = [g for g in geoms if g is not None]
        if not geoms:
            return None
        if len(geoms) == 1:
            return geoms[0]
        return AreaUnion(geoms)

    raise ValueError(
        "Unsupported GeoJSON geometry type "
        f"'{gtype}'. Only Polygon and MultiPolygon are supported"
    )


def load_areas_from_geojson(geojson_path, name_key="name"):
    """
    Loads area geometries from a GeoJSON file path.

    Parameters
    ----------
    geojson_path: str or pathlib.Path
        Path to a GeoJSON file

    name_key: str or list of str
        Preferred feature property key(s) for area names

    Returns
    -------
    area_map: dict
        Mapping from unique resolved area names (str) to
        `foxes.utils.geom2d.AreaGeometry` objects.
        Missing, empty, or duplicate names are replaced with
        default names of the form ``area_XXX``.

    :group: utils

    """
    geojson_path = Path(geojson_path)
    with geojson_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return load_areas_from_geojson_data(
        data,
        source_name=geojson_path,
        name_key=name_key,
    )


def load_areas_from_geojson_data(
    data,
    source_name="GeoJSON data",
    name_key="name",
):
    """
    Loads area geometries from a GeoJSON dictionary.

    Parameters
    ----------
    data: dict
        GeoJSON dictionary
    source_name: str
        Data source label used in error messages

    name_key: str or list of str
        Preferred feature property key(s) for area names

    Returns
    -------
    area_map: dict
        Mapping from unique resolved area names (str) to
        `foxes.utils.geom2d.AreaGeometry` objects.
        Missing, empty, or duplicate names are replaced with
        default names of the form ``area_XXX``.

    :group: utils

    """
    if not isinstance(data, dict):
        raise TypeError(
            "Expected GeoJSON data as dict, "
            f"got '{type(data).__name__}'"
        )

    def _feature_name(feature):
        props = feature.get("properties", {}) or {}
        keys = []
        if isinstance(name_key, str) and len(name_key):
            keys.append(name_key)
        elif isinstance(name_key, (tuple, list)):
            keys += [k for k in name_key if isinstance(k, str) and len(k)]

        for key in keys:
            if key in props and props[key] is not None:
                return str(props[key])
        return None

    dtype = data.get("type", None)
    if dtype == "FeatureCollection":
        features = data.get("features", [])
        geometries = [f.get("geometry", None) for f in features]
        names = [_feature_name(f) for f in features]
    elif dtype == "Feature":
        geometries = [data.get("geometry", None)]
        names = [_feature_name(data)]
    else:
        geometries = [data]
        names = [None]

    areas = []
    out_names = []
    for geom, name in zip(geometries, names):
        if geom is None:
            continue
        area = geojson_geometry_to_area(geom)
        if area is not None:
            areas.append(area)
            out_names.append(name)

    if not len(areas):
        raise ValueError(
            f"GeoJSON source '{source_name}' does not contain Polygon or MultiPolygon geometries"
        )

    used_names = set()
    norm_names = []
    for i, name in enumerate(out_names):
        if not isinstance(name, str) or not len(name.strip()) or name in used_names:
            name = f"area_{i:03d}"
        norm_names.append(name)
        used_names.add(name)

    return {name: area for name, area in zip(norm_names, areas)}


def normalize_areas_input(areas, geojson_name_key="name"):
    """
    Normalizes area input and resolves unique area names.

    Parameters
    ----------
    areas: list or str or pathlib.Path or dict
        Accepted area input forms:
        - list of AreaGeometry objects
        - list of (name, AreaGeometry) tuples
        - dict mapping names to AreaGeometry objects
        - path to GeoJSON file
        - GeoJSON dictionary
    geojson_name_key: str or list of str
        Preferred GeoJSON feature property key(s) used
        to read area names from GeoJSON inputs.

    Returns
    -------
    area_map: dict
        Mapping from unique resolved area names (str) to
        `foxes.utils.geom2d.AreaGeometry` objects.
        Missing, empty, or duplicate names are replaced with default
        names of the form ``area_XXX``.

    :group: utils

    """
    area_names = None

    if isinstance(areas, (str, Path)):
        return load_areas_from_geojson(
            areas,
            name_key=geojson_name_key,
        )
    elif isinstance(areas, dict):
        if "type" in areas:
            return load_areas_from_geojson_data(
                areas,
                name_key=geojson_name_key,
            )
        else:
            area_names = list(areas.keys())
            areas = list(areas.values())
    else:
        named = [
            isinstance(a, (tuple, list)) and len(a) == 2 and isinstance(a[0], str)
            for a in areas
        ]
        if all(named):
            area_names = [a[0] for a in areas]
            areas = [a[1] for a in areas]

    if area_names is None:
        area_names = [None] * len(areas)

    used_names = set()
    norm_names = []
    for i, name in enumerate(area_names):
        if not isinstance(name, str) or not len(name.strip()) or name in used_names:
            name = f"area_{i:03d}"
        norm_names.append(name)
        used_names.add(name)

    return {name: area for name, area in zip(norm_names, areas)}
