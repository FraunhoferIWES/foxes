import json

import pytest
import numpy as np

import foxes
import foxes.constants as FC
from foxes.config import config
from foxes.utils import load_areas_from_geojson_data, to_lonlat
from foxes.utils.geom2d import ClosedPolygon


def _build_farm():
    farm = foxes.WindFarm()
    farm.add_turbine(foxes.Turbine([0.0, 0.0]), verbosity=0)
    farm.add_turbine(foxes.Turbine([10.0, 0.0]), verbosity=0)
    farm.add_turbine(foxes.Turbine([20.0, 0.0]), verbosity=0)
    return farm


def _build_dense_farm(n_turbines=100):
    farm = foxes.WindFarm()
    for i in range(n_turbines):
        farm.add_turbine(foxes.Turbine([float(i), float(i % 10)]), verbosity=0)
    return farm


def _build_multiple_farms_output(farm):
    return foxes.output.MultipleFarmsOutput(farm, None)


def _geojson_polygon_from_xy(xy_points):
    if not config.utm_zone_set:
        config.set_utm_zone(31, "N")
    lonlat = to_lonlat(np.asarray(xy_points, dtype=np.float64))
    return lonlat.tolist()


def test_map_turbines_to_areas_with_area_list():
    farm = _build_farm()

    areas = [
        ClosedPolygon(np.array([[-1.0, -1.0], [15.0, -1.0], [15.0, 1.0], [-1.0, 1.0]])),
        ClosedPolygon(np.array([[15.0, -1.0], [25.0, -1.0], [25.0, 1.0], [15.0, 1.0]])),
    ]

    mapping = farm.map_turbines_to_areas(areas)

    assert mapping == {"area_000": [0, 1], "area_001": [2]}


def test_map_turbines_to_areas_with_named_area_list():
    farm = _build_farm()

    areas = [
        (
            "left",
            ClosedPolygon(
                np.array([[-1.0, -1.0], [15.0, -1.0], [15.0, 1.0], [-1.0, 1.0]])
            ),
        ),
        (
            "right",
            ClosedPolygon(
                np.array([[15.0, -1.0], [25.0, -1.0], [25.0, 1.0], [15.0, 1.0]])
            ),
        ),
    ]

    mapping = farm.map_turbines_to_areas(areas)

    assert mapping == {"left": [0, 1], "right": [2]}


def test_map_turbines_to_areas_with_geojson_path(tmp_path):
    farm = _build_farm()

    gj_path = tmp_path / "areas.geojson"
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "left"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        _geojson_polygon_from_xy(
                            [
                                [-1.0, -1.0],
                                [15.0, -1.0],
                                [15.0, 1.0],
                                [-1.0, 1.0],
                                [-1.0, -1.0],
                            ]
                        )
                    ],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "right"},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [
                            _geojson_polygon_from_xy(
                                [
                                    [15.0, -1.0],
                                    [25.0, -1.0],
                                    [25.0, 1.0],
                                    [15.0, 1.0],
                                    [15.0, -1.0],
                                ]
                            )
                        ],
                        [
                            _geojson_polygon_from_xy(
                                [
                                    [100.0, 100.0],
                                    [110.0, 100.0],
                                    [110.0, 110.0],
                                    [100.0, 110.0],
                                    [100.0, 100.0],
                                ]
                            )
                        ],
                    ],
                },
            },
        ],
    }
    gj_path.write_text(json.dumps(geojson_data), encoding="utf-8")

    mapping = farm.map_turbines_to_areas(gj_path)

    assert mapping == {"left": [0, 1], "right": [2]}


def test_map_turbines_to_areas_with_geojson_dict():
    farm = _build_farm()

    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "left"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        _geojson_polygon_from_xy(
                            [
                                [-1.0, -1.0],
                                [15.0, -1.0],
                                [15.0, 1.0],
                                [-1.0, 1.0],
                                [-1.0, -1.0],
                            ]
                        )
                    ],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "right"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        _geojson_polygon_from_xy(
                            [
                                [15.0, -1.0],
                                [25.0, -1.0],
                                [25.0, 1.0],
                                [15.0, 1.0],
                                [15.0, -1.0],
                            ]
                        )
                    ],
                },
            },
        ],
    }

    mapping = farm.map_turbines_to_areas(geojson_data)

    assert mapping == {"left": [0, 1], "right": [2]}


def test_load_areas_from_geojson_data_returns_name_area_dict():
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "left"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        _geojson_polygon_from_xy(
                            [
                                [-1.0, -1.0],
                                [15.0, -1.0],
                                [15.0, 1.0],
                                [-1.0, 1.0],
                                [-1.0, -1.0],
                            ]
                        )
                    ],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "right"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        _geojson_polygon_from_xy(
                            [
                                [15.0, -1.0],
                                [25.0, -1.0],
                                [25.0, 1.0],
                                [15.0, 1.0],
                                [15.0, -1.0],
                            ]
                        )
                    ],
                },
            },
        ],
    }

    area_map = load_areas_from_geojson_data(geojson_data)

    assert list(area_map.keys()) == ["left", "right"]
    assert all(hasattr(a, "add_to_figure") for a in area_map.values())


def test_map_turbines_to_areas_with_geojson_custom_name_key():
    farm = _build_farm()

    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"label": "left"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        _geojson_polygon_from_xy(
                            [
                                [-1.0, -1.0],
                                [15.0, -1.0],
                                [15.0, 1.0],
                                [-1.0, 1.0],
                                [-1.0, -1.0],
                            ]
                        )
                    ],
                },
            },
            {
                "type": "Feature",
                "properties": {"label": "right"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        _geojson_polygon_from_xy(
                            [
                                [15.0, -1.0],
                                [25.0, -1.0],
                                [25.0, 1.0],
                                [15.0, 1.0],
                                [15.0, -1.0],
                            ]
                        )
                    ],
                },
            },
        ],
    }

    mapping = farm.map_turbines_to_areas(geojson_data, geojson_name_key="label")

    assert mapping == {"left": [0, 1], "right": [2]}


def test_map_turbines_to_areas_with_named_area_dict():
    farm = _build_farm()

    areas = {
        "left": ClosedPolygon(
            np.array([[-1.0, -1.0], [15.0, -1.0], [15.0, 1.0], [-1.0, 1.0]])
        ),
        "right": ClosedPolygon(
            np.array([[15.0, -1.0], [25.0, -1.0], [25.0, 1.0], [15.0, 1.0]])
        ),
    }

    mapping = farm.map_turbines_to_areas(areas)

    assert mapping == {"left": [0, 1], "right": [2]}


def test_map_turbines_to_areas_set_cluster():
    farm = _build_farm()

    areas = {
        "left": ClosedPolygon(
            np.array([[-1.0, -1.0], [15.0, -1.0], [15.0, 1.0], [-1.0, 1.0]])
        ),
        "right": ClosedPolygon(
            np.array([[15.0, -1.0], [25.0, -1.0], [25.0, 1.0], [15.0, 1.0]])
        ),
    }

    mapping = farm.map_turbines_to_areas(areas, set_cluster=True)

    assert mapping == {"left": [0, 1], "right": [2]}
    assert farm.cluster_list == ["left", "left", "right"]


def test_map_turbines_to_areas_writes_plot_file(tmp_path):
    farm = _build_farm()

    areas = {
        "left": ClosedPolygon(
            np.array([[-1.0, -1.0], [15.0, -1.0], [15.0, 1.0], [-1.0, 1.0]])
        ),
        "right": ClosedPolygon(
            np.array([[15.0, -1.0], [25.0, -1.0], [25.0, 1.0], [15.0, 1.0]])
        ),
    }

    out_file = tmp_path / "plots" / "mapping.png"
    mapping = farm.map_turbines_to_areas(areas, plot_file=out_file)

    assert mapping == {"left": [0, 1], "right": [2]}
    assert out_file.is_file()
    assert out_file.stat().st_size > 0


def test_map_turbines_to_areas_plot_file_respects_output_dir(tmp_path):
    farm = _build_farm()

    areas = {
        "left": ClosedPolygon(
            np.array([[-1.0, -1.0], [15.0, -1.0], [15.0, 1.0], [-1.0, 1.0]])
        ),
        "right": ClosedPolygon(
            np.array([[15.0, -1.0], [25.0, -1.0], [25.0, 1.0], [15.0, 1.0]])
        ),
    }

    old_output_dir = config["out_dir"]
    config["out_dir"] = tmp_path
    try:
        mapping = farm.map_turbines_to_areas(areas, plot_file="plots/mapping_rel.png")
    finally:
        config["out_dir"] = old_output_dir

    out_file = tmp_path / "plots" / "mapping_rel.png"
    assert mapping == {"left": [0, 1], "right": [2]}
    assert out_file.is_file()
    assert out_file.stat().st_size > 0


def test_write_area_mapping_plot_handles_unknown_mapping_names(tmp_path):
    farm = _build_farm()
    out = _build_multiple_farms_output(farm)

    areas = [
        ClosedPolygon(np.array([[-1.0, -1.0], [15.0, -1.0], [15.0, 1.0], [-1.0, 1.0]])),
        ClosedPolygon(np.array([[15.0, -1.0], [25.0, -1.0], [25.0, 1.0], [15.0, 1.0]])),
    ]
    mapping = {"outside_name_set": [0]}

    out_file = tmp_path / "plots" / "mapping_unknown_names.png"
    out.write_area_mapping_plot(out_file, areas=areas, mapping=mapping)

    assert out_file.is_file()
    assert out_file.stat().st_size > 0


def test_write_area_mapping_plot_accepts_geojson_input(tmp_path):
    farm = _build_farm()
    out = _build_multiple_farms_output(farm)

    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "left"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        _geojson_polygon_from_xy(
                            [
                                [-1.0, -1.0],
                                [15.0, -1.0],
                                [15.0, 1.0],
                                [-1.0, 1.0],
                                [-1.0, -1.0],
                            ]
                        )
                    ],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "right"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        _geojson_polygon_from_xy(
                            [
                                [15.0, -1.0],
                                [25.0, -1.0],
                                [25.0, 1.0],
                                [15.0, 1.0],
                                [15.0, -1.0],
                            ]
                        )
                    ],
                },
            },
        ],
    }

    mapping = {"left": [0, 1], "right": [2]}
    out_file = tmp_path / "plots" / "mapping_geojson_input.png"

    out.write_area_mapping_plot(out_file, areas=geojson_data, mapping=mapping)

    assert out_file.is_file()
    assert out_file.stat().st_size > 0


def test_area_mapping_plot_layout_adapts_to_density_and_legend_size():
    areas = {
        "left": ClosedPolygon(
            np.array([[-1.0, -1.0], [15.0, -1.0], [15.0, 1.0], [-1.0, 1.0]])
        ),
        "right": ClosedPolygon(
            np.array([[15.0, -1.0], [25.0, -1.0], [25.0, 1.0], [15.0, 1.0]])
        ),
    }
    small_out = _build_multiple_farms_output(_build_farm())
    dense_out = _build_multiple_farms_output(_build_dense_farm())

    small_figsize, small_scatter = small_out.get_area_mapping_plot_layout(areas, 2)
    dense_figsize, dense_scatter = dense_out.get_area_mapping_plot_layout(areas, 2)
    legend_figsize, _ = small_out.get_area_mapping_plot_layout(areas, 12)

    assert small_figsize[0] > small_figsize[1]
    assert dense_scatter < small_scatter
    assert legend_figsize[1] > small_figsize[1]


def test_write_area_mapping_plot_supports_farm_level_choice(tmp_path):
    farm = _build_farm()
    for i, t in enumerate(farm.turbines):
        t.wind_farm_name = "west" if i < 2 else "east"

    areas = {
        "west": ClosedPolygon(
            np.array([[-1.0, -1.0], [15.0, -1.0], [15.0, 1.0], [-1.0, 1.0]])
        ),
        "east": ClosedPolygon(
            np.array([[15.0, -1.0], [25.0, -1.0], [25.0, 1.0], [15.0, 1.0]])
        ),
    }

    out = _build_multiple_farms_output(farm)
    out_file = tmp_path / "plots" / "mapping_farm_level.png"

    out.write_area_mapping_plot(out_file, areas=areas, level=FC.FARM)

    assert out_file.is_file()
    assert out_file.stat().st_size > 0


def test_map_turbines_to_areas_geojson_unsupported_geometry(tmp_path):
    farm = _build_farm()

    gj_path = tmp_path / "bad.geojson"
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                "properties": {},
            }
        ],
    }
    gj_path.write_text(json.dumps(geojson_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported GeoJSON geometry type"):
        farm.map_turbines_to_areas(gj_path)
