from __future__ import annotations

import foxes
import numpy as np
import pytest

from _model_smoke_helpers import _engine
from _model_smoke_helpers import _farm
from _model_smoke_helpers import _mbook_with_ttype
from foxes.models.partial_wakes.gaussian_lookup import PartialGaussianLookup
from foxes.utils.gaussian_lookup import DATA_WEIGHT
from foxes.utils.gaussian_lookup import build_lookup_dataset
from foxes.utils.gaussian_lookup import create_lookup_axes
from foxes.utils.gaussian_lookup import evaluate_lookup_geometry
from foxes.utils.gaussian_lookup import generate_lookup_dataset
from foxes.utils.gaussian_lookup import save_lookup_dataset
import foxes.variables as FV


def _algo_with_partial(partial_model: PartialGaussianLookup, wake_model: str):
    mbook, type_alias = _mbook_with_ttype()
    alias = "test_partial_gaussian_lookup"
    partial_model.name = alias
    mbook.partial_wakes[alias] = partial_model

    farm = _farm([type_alias])
    states = foxes.input.states.SingleStateStates(ws=8.0, wd=270.0, ti=0.08, rho=1.225)
    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model="centre",
        wake_models=[wake_model],
        partial_wakes={wake_model: alias},
        mbook=mbook,
        verbosity=0,
    )
    return algo


def test_partial_gaussian_lookup_loads_dataset_from_path(tmp_path):
    ds = generate_lookup_dataset(
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_over_d_max=1.0,
        sigma_resolution=0.05,
        n_rho=96,
        version_tag="lookup-test-v1",
    )
    fpath = tmp_path / "gaussian_lookup.nc"
    save_lookup_dataset(ds, fpath)

    model = PartialGaussianLookup(lookup_data=fpath)
    algo = _algo_with_partial(model, "Bastankhah2014_linear_k004")

    with _engine():
        _ = algo.calc_farm()

    assert model.lookup_dataset is not None
    assert model.lookup_dataset.attrs["version_tag"] == "lookup-test-v1"


def test_partial_gaussian_lookup_zero_weights_suppress_wakes():
    ds = generate_lookup_dataset(
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_over_d_max=1.0,
        sigma_resolution=0.05,
        n_rho=96,
        version_tag="lookup-zero-v1",
    )
    ds[DATA_WEIGHT][:] = 0.0

    model = PartialGaussianLookup(lookup_data=ds)
    algo = _algo_with_partial(model, "Bastankhah2014_linear_k004")

    with _engine():
        farm_results = algo.calc_farm()

    assert np.allclose(
        farm_results[FV.REWS].to_numpy(),
        farm_results[FV.AMB_REWS].to_numpy(),
    )


def test_partial_gaussian_lookup_rejects_non_gaussian_wake_model():
    model = PartialGaussianLookup()
    model.name = "test_partial_gaussian_lookup"
    mbook = foxes.models.ModelBook()
    wmodel = mbook.wake_models["Jensen_linear_k007"]

    with pytest.raises(TypeError, match="GaussianWakeModel"):
        model.check_wmodel(wmodel, error=True)


def test_partial_gaussian_lookup_out_of_range_clips_and_can_raise_by_default():
    r_axis, s_axis = create_lookup_axes(
        r_over_sigma_max=0.2,
        n_r=17,
        sigma_over_d_min=0.005,
        sigma_over_d_max=0.01,
        n_sigma=19,
    )
    ds = build_lookup_dataset(r_axis, s_axis, n_rho=96)

    model = PartialGaussianLookup(lookup_data=ds)
    assert model.bounds_policy == "clip"

    with pytest.raises(ValueError, match="Clipped out-of-bounds lookup query"):
        _ = evaluate_lookup_geometry(
            ds,
            r=np.array([100.0]),
            d=np.array([1.0]),
            sigma=np.array([0.01]),
            is_waked=np.array([True]),
            bounds_policy="clip",
            min_weight=model.min_weight,
            clip_check_min_weight=model.min_weight,
        )


def test_partial_gaussian_lookup_sigma_only_oob_clip_uses_asymptote():
    r_axis, s_axis = create_lookup_axes(
        r_over_sigma_max=0.2,
        n_r=17,
        sigma_over_d_min=0.005,
        sigma_over_d_max=0.01,
        n_sigma=19,
    )
    ds = build_lookup_dataset(r_axis, s_axis, n_rho=96)

    out = evaluate_lookup_geometry(
        ds,
        r=np.array([0.04]),
        d=np.array([1.0]),
        sigma=np.array([0.02]),
        is_waked=np.array([True]),
        bounds_policy="clip",
        min_weight=1.0e-8,
        clip_check_min_weight=1.0e-8,
    )
    assert out[0] == pytest.approx(np.exp(-0.5 * (0.04 / 0.02) ** 2))


def test_partial_gaussian_lookup_min_weight_zeros_small_weights():
    ds = generate_lookup_dataset(
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_over_d_max=1.0,
        sigma_resolution=0.05,
        n_rho=96,
        version_tag="lookup-min-weight-v1",
    )

    out = evaluate_lookup_geometry(
        ds,
        r=np.array([20.0]),
        d=np.array([1.0]),
        sigma=np.array([1.0]),
        is_waked=np.array([True]),
        bounds_policy="clip",
        min_weight=1.0e-8,
    )
    assert out[0] == pytest.approx(0.0)
