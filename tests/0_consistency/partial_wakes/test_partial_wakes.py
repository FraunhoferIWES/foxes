from pathlib import Path
import inspect

import numpy as np
import pytest

import foxes
import foxes.variables as FV
from foxes.utils.gaussian_lookup import generate_lookup_dataset


@pytest.fixture(scope="module")
def gaussian_lookup_xy_range_dataset():
    return generate_lookup_dataset(
        min_weight=1.0e-8,
        r_over_sigma_max=35.0,
        sigma_over_d_min=0.02,
        sigma_over_d_max=2.0,
        radial_resolution=0.05,
        sigma_resolution=0.025,
        n_rho=256,
        version_tag="xy-range-v1",
    )


def test():
    thisdir = Path(inspect.getabsfile(inspect.currentframe())).parent
    print("TESTDIR:", thisdir)

    tfile = thisdir / "NREL-5MW-D126-H90.csv"
    sfile = thisdir / "states.csv.gz"
    lfile = thisdir / "test_farm.csv"
    cases = [
        ("grid400", "rotor_points", None),
        ("grid4", "rotor_points", 0.15),
        ("grid9", "rotor_points", 0.07),
        ("centre", "axiwake5", 0.03),
        ("centre", "axiwake10", 0.0081),
        ("centre", "grid9", 0.07),
        ("centre", "grid16", 0.05),
        ("centre", "grid36", 0.025),
    ]

    base_results = None
    with foxes.Engine.new("threads", chunk_size_states=100):
        for rotor, pwake, lim in cases:
            print(f"\nENTERING CASE {(rotor, pwake, lim)}\n")

            mbook = foxes.models.ModelBook()
            ttype = foxes.models.turbine_types.PCtFile(
                data_source=tfile, var_ws_ct=FV.REWS, var_ws_P=FV.REWS
            )
            mbook.turbine_types[ttype.name] = ttype

            states = foxes.input.states.StatesTable(
                data_source=sfile,
                output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
                var2col={FV.WS: "ws", FV.WD: "wd", FV.TI: "ti"},
                fixed_vars={FV.RHO: 1.225},
            )

            farm = foxes.WindFarm()
            foxes.input.farm_layout.add_from_file(
                farm, lfile, turbine_models=[ttype.name], verbosity=1
            )

            algo = foxes.algorithms.Downwind(
                farm,
                states,
                mbook=mbook,
                rotor_model=rotor,
                wake_models=["Bastankhah025_linear_k002"],
                wake_frame="rotor_wd",
                partial_wakes=pwake,
                verbosity=1,
            )

            data = algo.calc_farm()

            df = data.to_dataframe()[
                [FV.AMB_WD, FV.WD, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]
            ]

            print()
            print("TRESULTS\n")
            print(df)

            df = df.reset_index()

            if base_results is None:
                base_results = df

            else:
                print(f"CASE {(rotor, pwake, lim)}")
                delta = df - base_results
                print(delta)
                print(delta.min(), delta.max())
                chk = delta[FV.REWS].abs()
                print(f"CASE {(rotor, pwake, lim)}:", chk.max())

                assert (chk < lim).all()


if __name__ == "__main__":
    test()


@pytest.mark.parametrize(
    "xy_step",
    [
        (450.0, 0.0),
        (450.0, 22.5),
        (900.0, 0.0),
        (900.0, 45.0),
        (1260.0, 0.0),
        (1260.0, 63.0),
        (1800.0, 0.0),
        (1800.0, 90.0),
        (2520.0, 0.0),
        (2520.0, 126.0),
    ],
    ids=[
        "x3p6d_y0",
        "x3p6d_y0p05x",
        "x7p1d_y0",
        "x7p1d_y0p05x",
        "x10d_y0",
        "x10d_y0p05x",
        "x14p3d_y0",
        "x14p3d_y0p05x",
        "x20d_y0",
        "x20d_y0p05x",
    ],
)
def test_gaussian_lookup_close_to_axiwake9_bastankhah2014(xy_step):
    thisdir = Path(inspect.getabsfile(inspect.currentframe())).parent
    tfile = thisdir / "NREL-5MW-D126-H90.csv"

    states = foxes.input.states.SingleStateStates(
        ws=8.0,
        wd=270.0,
        ti=0.08,
        rho=1.225,
    )

    def _calc_downstream_rews(pwake: str, mbook: foxes.models.ModelBook) -> float:
        ttype = foxes.models.turbine_types.PCtFile(
            data_source=tfile,
            var_ws_ct=FV.REWS,
            var_ws_P=FV.REWS,
        )
        mbook.turbine_types[ttype.name] = ttype

        farm = foxes.WindFarm()
        foxes.input.farm_layout.add_row(
            farm=farm,
            xy_base=[0.0, 0.0],
            xy_step=[xy_step[0], xy_step[1]],
            n_turbines=2,
            turbine_models=[ttype.name],
            H=90.0,
            verbosity=0,
        )

        algo = foxes.algorithms.Downwind(
            farm,
            states,
            mbook=mbook,
            rotor_model="centre",
            wake_models=["Bastankhah2014_linear_k004"],
            wake_frame="rotor_wd",
            partial_wakes=pwake,
            verbosity=0,
        )

        data = algo.calc_farm()
        return float(data[FV.REWS].to_numpy()[0, 1])

    with foxes.Engine.new("threads", chunk_size_states=1):
        mbook_axi = foxes.models.ModelBook()
        rews_axi = _calc_downstream_rews("axiwake9", mbook_axi)

        mbook_lookup = foxes.models.ModelBook()
        r_axis, s_axis = foxes.utils.create_lookup_axes(
            r_over_sigma_max=28.0,
            n_r=121,
            sigma_over_d_min=0.02,
            sigma_over_d_max=2.0,
            n_sigma=121,
        )
        lookup_ds = foxes.utils.build_lookup_dataset(
            r_over_sigma=r_axis,
            sigma_over_d=s_axis,
            n_rho=256,
            version_tag="consistency-v1",
        )
        mbook_lookup.partial_wakes["gaussian_lookup_test"] = (
            foxes.models.partial_wakes.PartialGaussianLookup(lookup_data=lookup_ds)
        )
        rews_lookup = _calc_downstream_rews("gaussian_lookup_test", mbook_lookup)

    # Best-effort match target against axiwake9 for Bastankhah2014.
    assert np.abs(rews_lookup - rews_axi) < 0.06
@pytest.mark.parametrize("x_over_d", [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0])
@pytest.mark.parametrize("y_over_d", [0.0, 0.25, 0.5, 1.0, 1.5])
def test_gaussian_lookup_matches_axiwake9_for_x_y_range(
    x_over_d,
    y_over_d,
    gaussian_lookup_xy_range_dataset,
):
    thisdir = Path(inspect.getabsfile(inspect.currentframe())).parent
    tfile = thisdir / "NREL-5MW-D126-H90.csv"
    rotor_diameter = 126.0
    xy_step = (x_over_d * rotor_diameter, y_over_d * rotor_diameter)
    states = foxes.input.states.SingleStateStates(
        ws=8.0,
        wd=270.0,
        ti=0.08,
        rho=1.225,
    )

    def _calc_downstream_rews(pwake: str, mbook: foxes.models.ModelBook) -> float:
        ttype = foxes.models.turbine_types.PCtFile(
            data_source=tfile,
            var_ws_ct=FV.REWS,
            var_ws_P=FV.REWS,
        )
        mbook.turbine_types[ttype.name] = ttype

        farm = foxes.WindFarm()
        foxes.input.farm_layout.add_row(
            farm=farm,
            xy_base=[0.0, 0.0],
            xy_step=[xy_step[0], xy_step[1]],
            n_turbines=2,
            turbine_models=[ttype.name],
            H=90.0,
            verbosity=0,
        )

        algo = foxes.algorithms.Downwind(
            farm,
            states,
            mbook=mbook,
            rotor_model="centre",
            wake_models=["Bastankhah2014_linear_k004"],
            wake_frame="rotor_wd",
            partial_wakes=pwake,
            verbosity=0,
        )

        data = algo.calc_farm()
        return float(data[FV.REWS].to_numpy()[0, 1])

    with foxes.Engine.new("threads", chunk_size_states=1):
        mbook_axi = foxes.models.ModelBook()
        rews_axi = _calc_downstream_rews("axiwake9", mbook_axi)

        mbook_lookup = foxes.models.ModelBook()
        mbook_lookup.partial_wakes["gaussian_lookup_test"] = (
            foxes.models.partial_wakes.PartialGaussianLookup(
                lookup_data=gaussian_lookup_xy_range_dataset
            )
        )
        rews_lookup = _calc_downstream_rews("gaussian_lookup_test", mbook_lookup)

    err = abs(rews_lookup - rews_axi)
    assert err < 0.12, (
        f"lookup table range mismatch for x/D={x_over_d}, y/D={y_over_d}, "
        f"xy_step={xy_step}: error={err:.4f}"
    )
