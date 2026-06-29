import pytest
import numpy as np

from foxes.utils.geom2d import AreaGeometry, AreaUnion, ClosedPolygon, from_shp


def test_area_geometry_from_shp_forwards_arguments(monkeypatch):
    from foxes.utils import geopandas_utils

    sentinel = object()
    calls = []

    def _fake_shp2geom2d(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(geopandas_utils, "shp2geom2d", _fake_shp2geom2d)

    out = AreaGeometry.from_shp(
        "areas.shp",
        names=["A", "B"],
        name_col="Name",
        geom_col="geometry",
        to_utm=False,
        combine_mode="intersection",
        ret_utm_zone=False,
        rows=slice(0, 2),
    )

    assert out is sentinel
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("areas.shp",)
    assert kwargs["names"] == ["A", "B"]
    assert kwargs["name_col"] == "Name"
    assert kwargs["geom_col"] == "geometry"
    assert kwargs["to_utm"] is False
    assert kwargs["combine_mode"] == "intersection"
    assert kwargs["ret_utm_zone"] is False
    assert kwargs["rows"] == slice(0, 2)


def test_area_geometry_from_shp_supports_ret_utm_zone(monkeypatch):
    from foxes.utils import geopandas_utils

    sentinel = (object(), "32U")

    def _fake_shp2geom2d(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(geopandas_utils, "shp2geom2d", _fake_shp2geom2d)

    out = AreaGeometry.from_shp("areas.shp", ret_utm_zone=True)

    assert out is sentinel


def test_area_geometry_from_shp_propagates_loader_errors(monkeypatch):
    from foxes.utils import geopandas_utils

    def _fake_shp2geom2d(*args, **kwargs):
        raise FileNotFoundError("areas.shp")

    monkeypatch.setattr(geopandas_utils, "shp2geom2d", _fake_shp2geom2d)

    with pytest.raises(FileNotFoundError, match="areas\\.shp"):
        AreaGeometry.from_shp("areas.shp")


def test_geom2d_from_shp_forwards_to_area_geometry(monkeypatch):
    sentinel = object()
    calls = []

    def _fake_from_shp(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(AreaGeometry, "from_shp", _fake_from_shp)

    out = from_shp("areas.shp", to_utm=False)

    assert out is sentinel
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("areas.shp",)
    assert kwargs["to_utm"] is False


def test_area_geometry_from_shp_glob_builds_union(tmp_path, monkeypatch):
    from foxes.utils import geopandas_utils

    shp_a = tmp_path / "a.shp"
    shp_b = tmp_path / "b.shp"
    shp_a.write_text("", encoding="utf-8")
    shp_b.write_text("", encoding="utf-8")

    def _fake_read_shp_polygons(*args, **kwargs):
        if args[0].endswith("a.shp"):
            ex = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        else:
            ex = np.array([[2.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
        return ex, {}

    monkeypatch.setattr(geopandas_utils, "read_shp_polygons", _fake_read_shp_polygons)

    out = geopandas_utils.shp2geom2d(str(tmp_path / "*.shp"), to_utm=False)

    assert isinstance(out, AreaUnion)
    assert len(out.geometries) == 2
    assert isinstance(out.geometries[0], ClosedPolygon)
    assert isinstance(out.geometries[1], ClosedPolygon)


def test_shp2geom2d_glob_builds_intersection(tmp_path, monkeypatch):
    from foxes.utils import geopandas_utils

    shp_a = tmp_path / "a.shp"
    shp_b = tmp_path / "b.shp"
    shp_a.write_text("", encoding="utf-8")
    shp_b.write_text("", encoding="utf-8")

    def _fake_read_shp_polygons(*args, **kwargs):
        if args[0].endswith("a.shp"):
            ex = np.array([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]])
        else:
            ex = np.array([[1.0, 1.0], [1.0, 3.0], [3.0, 3.0], [3.0, 1.0]])
        return ex, {}

    monkeypatch.setattr(geopandas_utils, "read_shp_polygons", _fake_read_shp_polygons)

    out = geopandas_utils.shp2geom2d(
        str(tmp_path / "*.shp"),
        to_utm=False,
        combine_mode="intersection",
    )

    pts = np.array(
        [
            [1.5, 1.5],
            [0.5, 0.5],
            [2.5, 2.5],
        ],
        dtype=np.float64,
    )
    inside = out.points_inside(pts)
    np.testing.assert_array_equal(inside, np.array([True, False, False]))


def test_area_geometry_from_shp_glob_forwards_to_loader(monkeypatch):
    from foxes.utils import geopandas_utils

    sentinel = object()
    calls = []

    def _fake_shp2geom2d(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(geopandas_utils, "shp2geom2d", _fake_shp2geom2d)

    out = AreaGeometry.from_shp("glob_*.shp")

    assert out is sentinel
    assert len(calls) == 1
    assert calls[0][0] == ("glob_*.shp",)


def test_area_geometry_from_shp_glob_rejects_ret_utm_zone(tmp_path):
    from foxes.utils import geopandas_utils

    with pytest.raises(ValueError, match="ret_utm_zone"):
        geopandas_utils.shp2geom2d(str(tmp_path / "*.shp"), ret_utm_zone=True)


def test_area_geometry_from_shp_glob_no_match(tmp_path):
    from foxes.utils import geopandas_utils

    with pytest.raises(FileNotFoundError, match="No files matched glob pattern"):
        geopandas_utils.shp2geom2d(str(tmp_path / "*.shp"))


def test_shp2geom2d_rejects_invalid_combine_mode(monkeypatch):
    from foxes.utils import geopandas_utils

    with pytest.raises(ValueError, match="Invalid combine_mode"):
        geopandas_utils.shp2geom2d("areas.shp", combine_mode="sum")
