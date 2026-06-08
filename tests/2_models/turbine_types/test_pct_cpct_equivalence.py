from types import SimpleNamespace

import numpy as np
import pandas as pd

import foxes.constants as FC
import foxes.variables as FV
from foxes.core import FData
from foxes.models.turbine_types import CpCtFile, PCtFile


def _init_model(model):
    algo = SimpleNamespace(store_model_data=lambda *args, **kwargs: None)
    model.initialize(algo=algo, verbosity=0)
    return model


def _build_fdata(ws, rho):
    shape = (1, 1)
    return FData(
        data={
            FV.REWS2: np.full(shape, ws, dtype=float),
            FV.REWS3: np.full(shape, ws, dtype=float),
            FV.RHO: np.full(shape, rho, dtype=float),
            FV.P: np.zeros(shape, dtype=float),
            FV.CT: np.zeros(shape, dtype=float),
        },
        dims={
            FV.REWS2: (FC.STATE, FC.TURBINE),
            FV.REWS3: (FC.STATE, FC.TURBINE),
            FV.RHO: (FC.STATE, FC.TURBINE),
            FV.P: (FC.STATE, FC.TURBINE),
            FV.CT: (FC.STATE, FC.TURBINE),
        },
    )


def test_cpctfile_matches_pctfile_power_for_equivalent_input():
    rho = 1.225
    rotor_diameter = 100.0
    area = np.pi * (rotor_diameter / 2) ** 2

    ws = np.array([5.0, 10.0], dtype=float)
    cp = np.array([0.35, 0.45], dtype=float)
    ct = np.array([0.70, 0.80], dtype=float)

    cpct_df = pd.DataFrame({"ws": ws, "cp": cp, "ct": ct})
    pct_df = pd.DataFrame(
        {
            "ws": ws,
            "P": 0.5 * rho * area * cp * ws**3 / FC.P_UNITS[FC.kW],
            "ct": ct,
        }
    )

    cpct_model = _init_model(
        CpCtFile(
            data_source=cpct_df,
            col_ws="ws",
            col_cp="cp",
            col_ct="ct",
            P_nominal=1000.0,
            D=rotor_diameter,
            H=90.0,
        )
    )
    pct_model = _init_model(
        PCtFile(
            data_source=pct_df,
            col_ws="ws",
            col_P="P",
            col_ct="ct",
            D=rotor_diameter,
            H=90.0,
            P_nominal=1000.0,
        )
    )

    fdata = _build_fdata(ws=7.5, rho=rho)
    st_sel = np.array([[True]])

    cpct_out = cpct_model.calculate(algo=None, mdata={}, fdata=fdata, st_sel=st_sel)
    pct_out = pct_model.calculate(algo=None, mdata={}, fdata=fdata, st_sel=st_sel)

    assert np.isclose(cpct_out[FV.P][0, 0], pct_out[FV.P][0, 0])


def test_cpctfile_matches_pctfile_power_for_different_ambient_rho():
    rho_ref = 1.225
    rho_ambient = 1.0
    rotor_diameter = 100.0
    area = np.pi * (rotor_diameter / 2) ** 2

    ws = np.array([5.0, 10.0], dtype=float)
    cp = np.array([0.40, 0.40], dtype=float)
    ct = np.array([0.70, 0.70], dtype=float)

    cpct_df = pd.DataFrame({"ws": ws, "cp": cp, "ct": ct})
    pct_df = pd.DataFrame(
        {
            "ws": ws,
            "P": 0.5 * rho_ref * area * cp * ws**3 / FC.P_UNITS[FC.kW],
            "ct": ct,
        }
    )

    cpct_model = _init_model(
        CpCtFile(
            data_source=cpct_df,
            col_ws="ws",
            col_cp="cp",
            col_ct="ct",
            P_nominal=1000.0,
            D=rotor_diameter,
            H=90.0,
        )
    )
    pct_model = _init_model(
        PCtFile(
            data_source=pct_df,
            col_ws="ws",
            col_P="P",
            col_ct="ct",
            rho=rho_ref,
            rho_corr_P="wind_speed",
            D=rotor_diameter,
            H=90.0,
            P_nominal=1000.0,
        )
    )

    fdata = _build_fdata(ws=7.5, rho=rho_ambient)
    st_sel = np.array([[True]])

    cpct_out = cpct_model.calculate(algo=None, mdata={}, fdata=fdata, st_sel=st_sel)
    pct_out = pct_model.calculate(algo=None, mdata={}, fdata=fdata, st_sel=st_sel)

    assert np.isclose(cpct_out[FV.P][0, 0], pct_out[FV.P][0, 0])


def main():
    test_cpctfile_matches_pctfile_power_for_equivalent_input()
    test_cpctfile_matches_pctfile_power_for_different_ambient_rho()


if __name__ == "__main__":
    main()
