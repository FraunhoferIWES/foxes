"""Sector identity across local direction sorting and engine chunks."""

import numpy as np
import pytest
import xarray as xr

import foxes
import foxes.constants as FC
import foxes.variables as FV
from foxes.input.states import FieldData, MesoMicroField, SectorSimRefPointField


def _field(speeds, directions):
    shape = (len(speeds), 1, 2, 3)
    dataset = xr.Dataset(
        {
            variable: (
                ("state", "height", "y", "x"),
                np.broadcast_to(np.asarray(values)[:, None, None, :], shape).copy(),
            )
            for variable, values in ((FV.WS, speeds), (FV.WD, directions))
        },
        coords={
            "state": np.arange(len(speeds)),
            "height": [90.0],
            "y": [-100.0, 100.0],
            "x": [-100.0, 0.0, 100.0],
        },
    )
    return FieldData(
        dataset,
        output_vars=[FV.WS, FV.WD],
        fixed_vars={FV.TI: 0.08, FV.RHO: 1.225},
        states_coord="state",
        time_format=None,
        x_coord="x",
        y_coord="y",
        h_coord="height",
    )


def _calculate(kind, speeds, directions, meso_directions, blending, chunk_size):
    micro = _field(speeds, directions)
    meso = _field(np.full((4, 3), 10.0), np.repeat(meso_directions[:, None], 3, axis=1))
    parameters = {
        "output_vars": [FV.WS, FV.WD, FV.TI, FV.RHO],
        "apply_blending": blending,
    }
    if kind == "memi":
        states = MesoMicroField(
            micro,
            meso,
            ref_points=[[-100.0, 0.0, 90.0], [100.0, 0.0, 90.0]],
            **parameters,
        )
    else:
        states = SectorSimRefPointField(
            micro,
            meso,
            ref_point=[0.0, 0.0, 90.0],
            **parameters,
        )
    farm = foxes.WindFarm()
    for x in (-100.0, 0.0, 100.0):
        farm.add_turbine(
            foxes.Turbine([x, 0.0], H=90.0, D=90.0, turbine_models=["null_type"]),
            verbosity=0,
        )
    algo = foxes.algorithms.Downwind(
        farm,
        states,
        wake_models=[],
        rotor_model="centre",
        verbosity=0,
    )
    with foxes.Engine.new(
        "numpy", n_procs=1, chunk_size_states=chunk_size, verbosity=0
    ):
        return algo.calc_farm(outputs=[FV.AMB_REWS, FV.WD, FV.WEIGHT])


@pytest.mark.parametrize("kind", ["memi", "cfd"])
@pytest.mark.parametrize("blending", [False, True])
def test_uniform_field_reproduces_reference_after_sector_wrap(kind, blending):
    speeds = np.repeat(np.array([8.0, 9.0, 10.0, 11.0])[:, None], 3, axis=1)
    directions = np.repeat(np.array([350.0, 80.0, 170.0, 260.0])[:, None], 3, axis=1)
    result = _calculate(kind, speeds, directions, np.full(4, 80.0), blending, 2)

    np.testing.assert_allclose(result[FV.AMB_REWS], 10.0)
    np.testing.assert_allclose(result[FV.WD], 80.0)
    np.testing.assert_allclose(result[FV.WEIGHT].sum(FC.STATE), 1.0)


@pytest.mark.parametrize("kind", ["memi", "cfd"])
@pytest.mark.parametrize("blending", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 4])
def test_sector_permutation_preserves_fields_with_local_wrap(
    kind, blending, chunk_size
):
    directions = np.mod(
        np.arange(0.0, 360.0, 90.0)[:, None] + np.array([-5.0, 0.0, 5.0]),
        360.0,
    )
    speeds = np.arange(8.0, 12.0)[:, None] + np.array([-0.5, 0.0, 0.5])
    meso_directions = np.array([0.0, 45.0, 80.0, 355.0])
    order = np.array([2, 0, 3, 1])
    expected = _calculate(kind, speeds, directions, meso_directions, blending, 4)
    actual = _calculate(
        kind,
        speeds[order],
        directions[order],
        meso_directions,
        blending,
        chunk_size,
    )

    for variable in (FV.AMB_REWS, FV.WD, FV.WEIGHT):
        np.testing.assert_allclose(actual[variable], expected[variable], atol=1e-12)


@pytest.mark.parametrize("kind", ["memi", "cfd"])
def test_duplicate_reference_directions_are_rejected(kind):
    with pytest.raises(ValueError, match="identical wind direction"):
        _calculate(
            kind,
            np.full((4, 3), 10.0),
            np.full((4, 3), 80.0),
            np.full(4, 80.0),
            True,
            4,
        )
