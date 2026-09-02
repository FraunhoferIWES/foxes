from __future__ import annotations

from pathlib import Path

import foxes
import numpy as np
import pytest

from _model_smoke_helpers import _engine
from _model_smoke_helpers import _farm
from _model_smoke_helpers import _mbook_with_ttype
from foxes.core.algorithm import Algorithm
from foxes.core.model import LoadedData
from foxes.models.partial_wakes.gaussian import PartialGaussian
from foxes.models.partial_wakes.gaussian import PartialGaussianLookup
from foxes.utils.gaussian_pwakes_utils import DATA_WEIGHT
from foxes.utils.gaussian_pwakes_utils import build_lookup_dataset
from foxes.utils.gaussian_pwakes_utils import create_lookup_axes
from foxes.utils.gaussian_pwakes_utils import evaluate_lookup_geometry
from foxes.utils.gaussian_pwakes_utils import generate_lookup_dataset
from foxes.utils.gaussian_pwakes_utils import save_lookup_dataset
import foxes.variables as FV


def _algo_with_partial(
    partial_model: PartialGaussianLookup | PartialGaussian, wake_model: str
) -> Algorithm:
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


def test_partial_gaussian_lookup_loads_dataset_from_path(tmp_path: Path) -> None:
    ds = generate_lookup_dataset(
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_resolution=2.0,
        n_rho=96,
        version_tag="lookup-test-v1",
    )
    fpath = tmp_path / "gaussian_lookup.nc"
    save_lookup_dataset(ds, fpath)

    model = PartialGaussianLookup(lookup_data=fpath)
    algo = _algo_with_partial(model, "Bastankhah2014_linear_k004")

    loaded_data: LoadedData = {"coords": {}, "data_vars": {}, "extra_data": {}}
    model.load_data(algo, loaded_data)

    lookup_dataset = loaded_data["extra_data"][model.lookup_dataset_key]
    assert lookup_dataset.attrs["version_tag"] == "lookup-test-v1"
    assert not hasattr(model, "lookup_dataset")

    with _engine():
        _ = algo.calc_farm()


def test_partial_gaussian_does_not_load_lookup_dataset() -> None:
    model = PartialGaussian()
    algo = _algo_with_partial(model, "Bastankhah2014_linear_k004")
    loaded_data: LoadedData = {"coords": {}, "data_vars": {}, "extra_data": {}}

    model.load_data(algo, loaded_data)

    assert loaded_data["extra_data"] == {}


def test_partial_gaussian_lookup_zero_weights_suppress_wakes() -> None:
    ds = generate_lookup_dataset(
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_resolution=2.0,
        n_rho=96,
        version_tag="lookup-zero-v1",
    )
    ds[DATA_WEIGHT][:] = 0.0
    r_axis = ds.coords["r_over_sigma"].to_numpy()
    ds[DATA_WEIGHT][:, -1] = np.exp(-0.5 * r_axis**2)

    model = PartialGaussianLookup(lookup_data=ds)
    algo = _algo_with_partial(model, "Bastankhah2014_linear_k004")

    with _engine():
        farm_results = algo.calc_farm()

    assert np.allclose(
        farm_results[FV.REWS].to_numpy(),
        farm_results[FV.AMB_REWS].to_numpy(),
    )


def test_partial_gaussian_lookup_rejects_non_gaussian_wake_model() -> None:
    model = PartialGaussianLookup()
    model.name = "test_partial_gaussian_lookup"
    mbook = foxes.models.ModelBook()
    wmodel = mbook.wake_models["Jensen_linear_k007"]

    with pytest.raises(TypeError, match="GaussianWakeModel"):
        model.check_wmodel(wmodel, error=True)


def test_partial_gaussian_lookup_out_of_range_clips_and_can_raise_by_default() -> None:
    r_axis, s_axis = create_lookup_axes(
        r_over_sigma_max=0.2,
        n_r=17,
        sigma_over_d_min=0.005,
    )
    ds = build_lookup_dataset(r_axis, s_axis, n_rho=96)
    ds.attrs["min_weight"] = 1.0e-8

    model = PartialGaussianLookup(lookup_data=ds)
    assert model.bounds_policy == "clip"
    min_weight = float(ds.attrs["min_weight"])

    with pytest.raises(ValueError, match="Clipped out-of-bounds lookup query"):
        _ = evaluate_lookup_geometry(
            ds,
            r=np.array([100.0]),
            d=np.array([1.0]),
            sigma=np.array([0.01]),
            is_waked=np.array([True]),
            bounds_policy="clip",
            min_weight=min_weight,
            clip_check_min_weight=min_weight,
        )


def test_partial_gaussian_lookup_sigma_only_oob_clip_uses_asymptote() -> None:
    r_axis, s_axis = create_lookup_axes(
        r_over_sigma_max=0.2,
        n_r=17,
        sigma_over_d_min=0.005,
    )
    ds = build_lookup_dataset(r_axis, s_axis, n_rho=96)

    out = evaluate_lookup_geometry(
        ds,
        r=np.array([0.04]),
        d=np.array([1.0]),
        sigma=np.array([80.0]),
        is_waked=np.array([True]),
        bounds_policy="clip",
        min_weight=1.0e-8,
        clip_check_min_weight=1.0e-8,
    )
    assert out[0] == pytest.approx(np.exp(-0.5 * (0.04 / 80.0) ** 2))


def test_partial_gaussian_lookup_min_weight_zeros_small_weights() -> None:
    ds = generate_lookup_dataset(
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_resolution=2.0,
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
