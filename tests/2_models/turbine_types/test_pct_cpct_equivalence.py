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


def _dtu_10mw_curve_all():
    """Hard-coded DTU-10MW curve (ws, P[kW], ct), all points."""
    ws = np.array(
        [
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
            20.0,
            21.0,
            22.0,
            23.0,
            24.0,
            25.0,
        ]
    )
    p_kw = np.array(
        [
            280.2,
            799.1,
            1532.7,
            2506.1,
            3730.7,
            5311.8,
            7286.5,
            9698.3,
            10639.1,
            10648.5,
            10639.3,
            10683.7,
            10642.0,
            10640.0,
            10639.9,
            10652.8,
            10646.2,
            10644.0,
            10641.2,
            10639.5,
            10643.6,
            10635.7,
        ]
    )
    ct = np.array(
        [
            0.932,
            0.919,
            0.904,
            0.858,
            0.814,
            0.814,
            0.814,
            0.814,
            0.577,
            0.419,
            0.323,
            0.259,
            0.211,
            0.175,
            0.148,
            0.126,
            0.109,
            0.095,
            0.084,
            0.074,
            0.066,
            0.059,
        ]
    )
    return ws, p_kw, ct


def test_cpctfile_matches_pctfile_power_for_equivalent_input():
    rho = 1.225
    rotor_diameter = 178.3
    area = np.pi * (rotor_diameter / 2) ** 2

    ws, p_kw, ct = _dtu_10mw_curve_all()
    cp = p_kw * FC.P_UNITS[FC.kW] / (0.5 * rho * area * ws**3)

    cpct_df = pd.DataFrame({"ws": ws, "cp": cp, "ct": ct})
    pct_df = pd.DataFrame(
        {
            "ws": ws,
            "P": p_kw,
            "ct": ct,
        }
    )

    cpct_model = _init_model(
        CpCtFile(
            data_source=cpct_df,
            col_ws="ws",
            col_cp="cp",
            col_ct="ct",
            P_nominal=10650.0,
            D=rotor_diameter,
            H=119.0,
        )
    )
    pct_model = _init_model(
        PCtFile(
            data_source=pct_df,
            col_ws="ws",
            col_P="P",
            col_ct="ct",
            D=rotor_diameter,
            H=119.0,
            P_nominal=10650.0,
        )
    )

    fdata = _build_fdata(ws=7.5, rho=rho)
    st_sel = np.array([[True]])

    cpct_out = cpct_model.calculate(algo=None, mdata={}, fdata=fdata, st_sel=st_sel)
    pct_out = pct_model.calculate(algo=None, mdata={}, fdata=fdata, st_sel=st_sel)

    assert np.isclose(cpct_out[FV.P][0, 0], pct_out[FV.P][0, 0], atol=1e-5)


def test_cpctfile_matches_pctfile_power_for_different_ambient_rho():
    rho_ref = 1.225
    rho_ambient = 1.0
    rotor_diameter = 178.3
    area = np.pi * (rotor_diameter / 2) ** 2

    ws, p_kw, ct = _dtu_10mw_curve_all()
    cp = p_kw * FC.P_UNITS[FC.kW] / (0.5 * rho_ref * area * ws**3)

    cpct_df = pd.DataFrame({"ws": ws, "cp": cp, "ct": ct})
    pct_df = pd.DataFrame(
        {
            "ws": ws,
            "P": p_kw,
            "ct": ct,
        }
    )

    cpct_model = _init_model(
        CpCtFile(
            data_source=cpct_df,
            col_ws="ws",
            col_cp="cp",
            col_ct="ct",
            P_nominal=10650.0,
            D=rotor_diameter,
            H=119.0,
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
            H=119.0,
            P_nominal=10650.0,
        )
    )

    fdata = _build_fdata(ws=7.5, rho=rho_ambient)
    st_sel = np.array([[True]])

    cpct_out = cpct_model.calculate(algo=None, mdata={}, fdata=fdata, st_sel=st_sel)
    pct_out = pct_model.calculate(algo=None, mdata={}, fdata=fdata, st_sel=st_sel)

    assert np.isclose(cpct_out[FV.P][0, 0], pct_out[FV.P][0, 0], atol=1e-5)


def main():
    test_cpctfile_matches_pctfile_power_for_equivalent_input()
    test_cpctfile_matches_pctfile_power_for_different_ambient_rho()


if __name__ == "__main__":
    main()
