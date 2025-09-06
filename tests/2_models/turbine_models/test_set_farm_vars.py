import numpy as np

import foxes
import foxes.variables as FV
import foxes.constants as FC


def test_set_farm_vars():
    n_s = 360
    n_tr = 3

    wd = np.arange(0.0, 360.0, 360 / n_s)
    states = foxes.input.states.ScanStates(
        scans={
            FV.WD: wd,
            FV.WS: [9.0],
            FV.TI: [0.04],
            FV.RHO: [1.225],
        },
    )

    n_t = n_tr**2
    x = np.zeros((n_s, n_t), dtype=foxes.config.dtype_double)
    x[:] = wd[:, None] + np.arange(n_t)[None, :] / 10

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_grid(
        farm,
        xy_base=[0.0, 0.0],
        step_vectors=[[800.0, 0.0], [0.0, 800.0]],
        steps=[n_tr, n_tr],
        turbine_models=["NREL5MW", "set_x"],
        verbosity=0,
    )

    with foxes.Engine.new("default", verbosity=0):
        for pr in [False, True]:
            print(f"\npre_rotor = {pr}\n")

            mbook = foxes.ModelBook()
            mbook.turbine_models["set_x"] = foxes.models.turbine_models.SetFarmVars(
                pre_rotor=pr
            )
            mbook.turbine_models["set_x"].add_var("x", x)

            algo = foxes.algorithms.Downwind(
                farm=farm,
                states=states,
                wake_models=["Bastankhah2014_linear_lim_k004"],
                mbook=mbook,
                verbosity=0,
            )

            farm_results = algo.calc_farm()

            fr = farm_results.to_dataframe()
            print(fr[[FV.WD, "x"]])

            for i, g in fr.reset_index().groupby(FC.TURBINE):
                assert np.allclose(g["x"].values, g[FV.WD].values + i / 10)


if __name__ == "__main__":
    test_set_farm_vars()
