import numpy as np
import pytest

from foxes.config import config
from foxes.utils.gaussian_lookup import (
    AXIS_R_OVER_SIGMA,
    AXIS_SIGMA_OVER_D,
    DATA_WEIGHT,
    build_lookup_dataset,
    create_lookup_axes,
    evaluate_lookup_dataset,
    evaluate_lookup_geometry,
    gaussian_disc_weight,
    generate_lookup_dataset,
    load_lookup_dataset,
    save_lookup_dataset,
    validate_lookup_dataset,
)


def test_create_lookup_axes_defaults_are_monotonic_and_positive():
    r_axis, s_axis = create_lookup_axes()

    assert r_axis.ndim == 1
    assert s_axis.ndim == 1
    assert np.all(np.diff(r_axis) > 0.0)
    assert np.all(np.diff(s_axis) > 0.0)
    assert r_axis[0] == pytest.approx(0.0)
    assert np.all(s_axis > 0.0)
    assert r_axis[-1] == pytest.approx(28.0)
    assert s_axis[-1] == pytest.approx(20.0)


def test_gaussian_disc_weight_large_sigma_approaches_one():
    r_axis = np.array([0.0, 0.2, 0.6, 1.0])
    s_axis = np.array([1.0e3])

    w = gaussian_disc_weight(r_axis, s_axis, n_rho=128)

    np.testing.assert_allclose(w[:, 0], np.exp(-0.5 * r_axis**2), atol=1.0e-6)


def test_gaussian_disc_weight_decreases_with_offset():
    r_axis = np.array([0.0, 0.2, 0.5, 1.0])
    s_axis = np.array([0.15])

    w = gaussian_disc_weight(r_axis, s_axis, n_rho=512)[:, 0]

    assert np.all(np.diff(w) < 0.0)


def test_build_and_evaluate_lookup_dataset_shape_and_coords():
    r_axis = np.linspace(0.0, 1.2, 25)
    s_axis = np.geomspace(0.05, 0.6, 21)
    ds = build_lookup_dataset(r_axis, s_axis, n_rho=256)

    assert AXIS_R_OVER_SIGMA in ds.coords
    assert AXIS_SIGMA_OVER_D in ds.coords
    assert DATA_WEIGHT in ds.data_vars

    rq = np.array([[0.0, 0.3], [0.8, 1.1]])
    sq = np.array([[0.08, 0.10], [0.25, 0.50]])
    out = evaluate_lookup_dataset(ds, rq, sq)

    assert out.shape == rq.shape
    assert np.all(np.isfinite(out))


def test_create_lookup_axes_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="r_over_sigma_max"):
        create_lookup_axes(r_over_sigma_max=0.0)

    with pytest.raises(ValueError, match="sigma_over_d_min"):
        create_lookup_axes(sigma_over_d_min=0.0)

    with pytest.raises(ValueError, match="sigma_over_d_max"):
        create_lookup_axes(sigma_over_d_min=0.2, sigma_over_d_max=0.1)


def test_generate_lookup_dataset_has_expected_metadata_and_is_deterministic():
    kw = dict(
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_over_d_max=1.0,
        sigma_resolution=0.05,
        sigma_spacing="log",
        n_rho=192,
        version_tag="test-v1",
    )
    ds0 = generate_lookup_dataset(**kw)
    ds1 = generate_lookup_dataset(**kw)

    assert ds0.attrs["version_tag"] == "test-v1"
    assert ds0.attrs["sigma_spacing"] == "log"
    assert ds0.attrs["n_rho"] == 192
    assert ds0.attrs["radial_resolution"] == pytest.approx(0.1)
    assert ds0.attrs["sigma_resolution"] == pytest.approx(0.05)
    assert np.array_equal(ds0[DATA_WEIGHT].to_numpy(), ds1[DATA_WEIGHT].to_numpy())


def test_save_lookup_dataset_uses_configured_nc_engine(tmp_path):
    old_engine = config["nc_engine"]
    config["nc_engine"] = "h5netcdf"
    try:
        ds = generate_lookup_dataset(
            radial_resolution=0.1,
            sigma_over_d_min=0.02,
            sigma_over_d_max=1.0,
            sigma_resolution=0.05,
            sigma_spacing="linear",
            n_rho=160,
            version_tag="config-engine-v1",
        )
        fpath = tmp_path / "gaussian_lookup_cfg_engine.nc"
        save_lookup_dataset(ds, fpath)
        assert fpath.exists()
    finally:
        config["nc_engine"] = old_engine


def test_generate_lookup_dataset_rejects_invalid_sigma_bounds():
    with pytest.raises(ValueError, match="sigma_over_d_min"):
        generate_lookup_dataset(sigma_over_d_min=0.0)

    with pytest.raises(ValueError, match="sigma_over_d_max"):
        generate_lookup_dataset(
            sigma_over_d_min=0.2,
            sigma_over_d_max=0.1,
        )


def test_lookup_dataset_netcdf_roundtrip(tmp_path):
    ds = generate_lookup_dataset(
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_over_d_max=1.0,
        sigma_resolution=0.05,
        sigma_spacing="linear",
        n_rho=160,
        version_tag="roundtrip-v1",
    )
    fpath = tmp_path / "gaussian_lookup_roundtrip.nc"

    save_lookup_dataset(ds, fpath)
    loaded = load_lookup_dataset(fpath)
    validate_lookup_dataset(loaded)

    assert np.array_equal(
        loaded.coords[AXIS_R_OVER_SIGMA].to_numpy(),
        ds.coords[AXIS_R_OVER_SIGMA].to_numpy(),
    )
    assert np.array_equal(
        loaded.coords[AXIS_SIGMA_OVER_D].to_numpy(),
        ds.coords[AXIS_SIGMA_OVER_D].to_numpy(),
    )
    np.testing.assert_allclose(
        loaded[DATA_WEIGHT].to_numpy(), ds[DATA_WEIGHT].to_numpy(), atol=1.0e-12
    )
    assert loaded.attrs["version_tag"] == "roundtrip-v1"


def test_evaluate_lookup_dataset_bounds_policies():
    ds = generate_lookup_dataset(
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_over_d_max=1.0,
        sigma_resolution=0.05,
        n_rho=128,
    )

    r_q = np.array([0.2, 100.0])
    s_q = np.array([0.2, 0.2])

    out_nan = evaluate_lookup_dataset(ds, r_q, s_q, bounds_policy="nan")
    assert np.isfinite(out_nan[0])
    assert np.isnan(out_nan[1])

    out_clip = evaluate_lookup_dataset(ds, r_q, s_q, bounds_policy="clip")
    assert np.all(np.isfinite(out_clip))

    with pytest.raises(
        ValueError,
        match="outside bounds.*offending point=.*nearest_weight=",
    ):
        evaluate_lookup_dataset(ds, r_q, s_q, bounds_policy="raise")


def test_evaluate_lookup_dataset_clip_uses_high_sigma_asymptote():
    ds = generate_lookup_dataset(
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_over_d_max=1.0,
        sigma_resolution=0.05,
        n_rho=128,
    )
    r_max = float(ds.coords[AXIS_R_OVER_SIGMA].to_numpy()[-1])
    sigma_min = float(ds.coords[AXIS_SIGMA_OVER_D].to_numpy()[0])
    sigma_max = float(ds.coords[AXIS_SIGMA_OVER_D].to_numpy()[-1])

    out = evaluate_lookup_dataset(
        ds,
        r_over_sigma=np.array([2.0 * r_max, 0.2, 2.0]),
        sigma_over_d=np.array([0.5 * sigma_min, 2.0 * sigma_max, 2.0 * sigma_max]),
        bounds_policy="clip",
    )

    assert np.isfinite(out[0])
    assert out[1] == pytest.approx(np.exp(-0.5 * 0.2**2))
    assert out[2] == pytest.approx(np.exp(-0.5 * 2.0**2))


def test_evaluate_lookup_geometry_masks_non_waked_and_guards_invalid():
    ds = generate_lookup_dataset(
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_over_d_max=1.0,
        sigma_resolution=0.05,
        n_rho=128,
    )

    r = np.array([10.0, 10.0, 10.0])
    d = np.array([100.0, 0.0, 100.0])
    sigma = np.array([10.0, 10.0, -1.0])
    is_waked = np.array([True, False, False])

    out = evaluate_lookup_geometry(ds, r, d, sigma, is_waked=is_waked)
    assert np.isfinite(out[0])
    assert out[1] == pytest.approx(0.0)
    assert out[2] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="Invalid geometry"):
        evaluate_lookup_geometry(ds, r, d, sigma, is_waked=np.array([True, True, False]))


def test_evaluate_lookup_geometry_clip_masks_radial_oob_below_min_weight():
    ds = generate_lookup_dataset(
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_over_d_max=1.0,
        sigma_resolution=0.05,
        n_rho=128,
    )

    out = evaluate_lookup_geometry(
        ds,
        r=np.array([1000.0]),
        d=np.array([1.0]),
        sigma=np.array([0.1]),
        is_waked=np.array([True]),
        bounds_policy="clip",
        min_weight=1.0e-8,
        clip_check_min_weight=1.0e-8,
    )
    assert out[0] == pytest.approx(0.0)


def test_generate_lookup_dataset_auto_expands_r_for_min_weight():
    ds = generate_lookup_dataset(
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_over_d_max=1.0,
        sigma_resolution=0.05,
        n_rho=96,
        min_weight=1.0e-4,
        radial_expand_factor=1.5,
    )

    assert ds.attrs["axis_r_over_sigma_max"] > 0.5
    assert ds.attrs["axis_sigma_over_d_min"] == pytest.approx(0.02)
    assert ds.attrs["axis_sigma_over_d_max"] == pytest.approx(1.0)
    assert ds.attrs["auto_edge_weight_max"] <= 1.0e-4
    assert ds.attrs["auto_edge_weight_r_max"] <= 1.0e-4


def test_generate_lookup_dataset_respects_explicit_r_over_sigma_max():
    ds = generate_lookup_dataset(
        min_weight=1.0e-8,
        r_over_sigma_max=4.0,
        sigma_over_d_min=0.02,
        sigma_over_d_max=1.0,
        radial_resolution=0.1,
        sigma_resolution=0.05,
        n_rho=96,
    )

    assert ds.attrs["axis_r_over_sigma_max"] == pytest.approx(4.0)
    assert ds.coords[AXIS_R_OVER_SIGMA].to_numpy()[-1] == pytest.approx(4.0)


def test_evaluate_lookup_geometry_clip_uses_high_sigma_asymptote():
    ds = generate_lookup_dataset(
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_over_d_max=1.0,
        sigma_resolution=0.05,
        n_rho=128,
    )

    out = evaluate_lookup_geometry(
        ds,
        r=np.array([0.05]),
        d=np.array([1.0]),
        sigma=np.array([100.0]),
        is_waked=np.array([True]),
        bounds_policy="clip",
        min_weight=1.0e-4,
        clip_check_min_weight=1.0e-4,
    )
    assert out[0] == pytest.approx(np.exp(-0.5 * (0.05 / 100.0) ** 2))
