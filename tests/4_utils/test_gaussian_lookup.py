import numpy as np
import pytest

from foxes.config import config
from foxes.utils.gaussian_pwakes_utils import (
    AXIS_R_OVER_SIGMA,
    AXIS_SIGMA_OVER_D,
    DATA_WEIGHT,
    build_lookup_dataset,
    create_lookup_axes,
    evaluate_lookup_dataset,
    evaluate_lookup_geometry,
    gaussian_disc_weight,
    gaussian_disc_weight_analytical,
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
    assert s_axis[-1] > s_axis[0]


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


def test_gaussian_disc_weight_analytical_matches_quadrature():
    r_axis = np.linspace(0.0, 8.0, 81)
    s_axis = np.geomspace(0.02, 20.0, 41)
    reference = gaussian_disc_weight(r_axis, s_axis, n_rho=2048)

    actual = gaussian_disc_weight_analytical(
        r_axis[:, None] * s_axis[None, :],
        np.ones(reference.shape),
        s_axis[None, :],
    )

    np.testing.assert_allclose(actual, reference, rtol=1.0e-5, atol=1.0e-8)


def test_gaussian_disc_weight_analytical_masks_and_validates_geometry():
    out = gaussian_disc_weight_analytical(
        r=np.array([1.0, 1.0]),
        d=np.array([100.0, 0.0]),
        sigma=np.array([10.0, -1.0]),
        is_waked=np.array([True, False]),
    )
    assert out[0] > 0.0
    assert out[1] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="Invalid geometry"):
        gaussian_disc_weight_analytical(
            r=np.array([1.0]),
            d=np.array([0.0]),
            sigma=np.array([10.0]),
        )


def test_build_and_evaluate_lookup_dataset_shape_and_coords():
    r_axis = np.linspace(0.0, 1.2, 25)
    s_axis = np.geomspace(0.05, 40.0, 21)
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

    with pytest.raises(ValueError, match="asymptote_rel_tol"):
        create_lookup_axes(asymptote_rel_tol=0.0)


def test_generate_lookup_dataset_has_expected_metadata_and_is_deterministic():
    kw = dict(
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_resolution=2.0,
        sigma_spacing="log",
        n_rho=192,
        version_tag="test-v1",
    )
    ds0 = generate_lookup_dataset(**kw)
    ds1 = generate_lookup_dataset(**kw)

    assert ds0.attrs["version_tag"] == "test-v1"
    assert ds0.attrs["sigma_spacing"] == "log"
    assert ds0.attrs["n_rho"] == 192
    assert ds0.attrs["min_weight"] == pytest.approx(1.0e-8)
    assert ds0.attrs["radial_resolution"] == pytest.approx(0.1)
    assert ds0.attrs["sigma_resolution"] == pytest.approx(2.0)
    assert ds0.attrs["asymptote_rel_tol"] == pytest.approx(1.0e-3)
    assert np.array_equal(ds0[DATA_WEIGHT].to_numpy(), ds1[DATA_WEIGHT].to_numpy())


def test_save_lookup_dataset_uses_configured_nc_engine(tmp_path):
    old_engine = config["nc_engine"]
    config["nc_engine"] = "h5netcdf"
    try:
        ds = generate_lookup_dataset(
            radial_resolution=0.1,
            sigma_over_d_min=0.02,
            sigma_resolution=2.0,
            sigma_spacing="linear",
            n_rho=160,
            version_tag="config-engine-v1",
        )
        fpath = tmp_path / "gaussian_lookup_cfg_engine.nc"
        save_lookup_dataset(ds, fpath)
        assert fpath.exists()
    finally:
        config["nc_engine"] = old_engine


def test_generate_lookup_dataset_rejects_invalid_asymptote_tolerance():
    with pytest.raises(ValueError, match="sigma_over_d_min"):
        generate_lookup_dataset(sigma_over_d_min=0.0)

    with pytest.raises(ValueError, match="asymptote_rel_tol"):
        generate_lookup_dataset(asymptote_rel_tol=0.0)


def test_lookup_dataset_netcdf_roundtrip(tmp_path):
    ds = generate_lookup_dataset(
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_resolution=2.0,
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
        sigma_resolution=2.0,
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
        sigma_resolution=2.0,
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
        sigma_resolution=2.0,
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
        evaluate_lookup_geometry(
            ds, r, d, sigma, is_waked=np.array([True, True, False])
        )


def test_evaluate_lookup_geometry_clip_masks_radial_oob_below_min_weight():
    ds = generate_lookup_dataset(
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_resolution=2.0,
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
        sigma_resolution=2.0,
        n_rho=96,
        min_weight=1.0e-4,
        radial_expand_factor=1.5,
    )

    assert ds.attrs["axis_r_over_sigma_max"] > 0.5
    assert ds.attrs["axis_sigma_over_d_min"] == pytest.approx(0.02)
    assert ds.attrs["axis_sigma_over_d_upper"] > 0.02
    assert ds.attrs["auto_edge_weight_max"] <= 1.0e-4
    assert ds.attrs["auto_edge_weight_r_max"] <= 1.0e-4


def test_generate_lookup_dataset_respects_explicit_r_over_sigma_max():
    ds = generate_lookup_dataset(
        min_weight=1.0e-8,
        r_over_sigma_max=4.0,
        sigma_over_d_min=0.02,
        radial_resolution=0.1,
        sigma_resolution=2.0,
        n_rho=96,
    )

    assert ds.attrs["axis_r_over_sigma_max"] == pytest.approx(4.0)
    assert ds.coords[AXIS_R_OVER_SIGMA].to_numpy()[-1] == pytest.approx(4.0)


def test_evaluate_lookup_geometry_clip_uses_high_sigma_asymptote():
    ds = generate_lookup_dataset(
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_resolution=2.0,
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


@pytest.mark.parametrize("bounds_policy", ["clip", "nan", "raise"])
def test_evaluate_lookup_dataset_always_uses_high_sigma_asymptote(bounds_policy):
    ds = generate_lookup_dataset(
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_resolution=2.0,
        n_rho=128,
    )

    out = evaluate_lookup_dataset(
        ds,
        r_over_sigma=np.array([0.2]),
        sigma_over_d=np.array([80.0]),
        bounds_policy=bounds_policy,
    )

    assert out[0] == pytest.approx(np.exp(-0.5 * 0.2**2))


@pytest.mark.parametrize("bounds_policy", ["clip", "nan", "raise"])
def test_evaluate_lookup_dataset_policies_do_not_apply_to_low_sigma(bounds_policy):
    ds = generate_lookup_dataset(
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_resolution=2.0,
        n_rho=128,
    )
    sigma_min = float(ds.coords[AXIS_SIGMA_OVER_D].to_numpy()[0])

    out = evaluate_lookup_dataset(
        ds,
        r_over_sigma=np.array([0.2]),
        sigma_over_d=np.array([0.5 * sigma_min]),
        bounds_policy=bounds_policy,
    )
    expected = evaluate_lookup_dataset(
        ds,
        r_over_sigma=np.array([0.2]),
        sigma_over_d=np.array([sigma_min]),
    )

    np.testing.assert_allclose(out, expected)


def test_validate_lookup_dataset_rejects_inaccurate_high_sigma_asymptote():
    r_axis = np.linspace(0.0, 8.0, 81)
    s_axis = np.geomspace(0.02, 10.0, 21)
    ds = build_lookup_dataset(r_axis, s_axis, n_rho=128)

    with pytest.raises(ValueError, match="large-sigma asymptote"):
        validate_lookup_dataset(ds)


@pytest.mark.parametrize(
    ("invalid_weight", "message"),
    [
        (np.nan, "must be finite"),
        (-0.1, "must be within"),
        (1.1, "must be within"),
    ],
)
def test_validate_lookup_dataset_rejects_invalid_weight_values(invalid_weight, message):
    ds = generate_lookup_dataset(
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_resolution=2.0,
        n_rho=128,
    )
    ds[DATA_WEIGHT].data[0, 0] = invalid_weight

    with pytest.raises(ValueError, match=message):
        validate_lookup_dataset(ds)


def test_generate_lookup_dataset_expands_sigma_range_for_tolerance():
    loose = generate_lookup_dataset(
        radial_resolution=0.1,
        sigma_resolution=2.0,
        n_rho=128,
        asymptote_rel_tol=1.0e-2,
    )
    strict = generate_lookup_dataset(
        radial_resolution=0.1,
        sigma_resolution=2.0,
        n_rho=128,
        asymptote_rel_tol=1.0e-3,
    )

    assert loose.attrs["asymptote_rel_tol"] == pytest.approx(1.0e-2)
    assert strict.attrs["asymptote_rel_tol"] == pytest.approx(1.0e-3)
    assert (
        strict.attrs["axis_sigma_over_d_upper"] > loose.attrs["axis_sigma_over_d_upper"]
    )
