import pytest

from foxes.utils.geom2d import AreaGeometry, from_shp


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
