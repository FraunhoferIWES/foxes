import numpy as np
import time
import dask
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV
from dask.distributed import Client

if __name__ == "__main__":

    n_s = 10
    n_t = 6
    n_p = 2200
    p0 = np.array([0.0, 0.0])
    stp = np.array([500.0, 0.0])
    cks = None  # {FV.STATE: 2000}#, FV.POINT:5}
    D = 120.0
    H = 100.0
    h = 100.0

    dask.config.set(scheduler="synchronous")
    # dask.config.set(scheduler='threads')
    # dask.config.set(scheduler='processes')
    # dask.config.set(scheduler='distributed')
    # client = Client(n_workers=4, threads_per_worker=1)
    # print(f"\n{client}")
    # print(f"Dashboard: {client.dashboard_link}\n")

    mbook = foxes.models.ModelBook()
    mbook.turbine_types["TOYT"] = foxes.models.turbine_types.PCtFile(
        name="TOYT", filepath="toyTurbine.csv", D=D, H=H
    )

    states = foxes.input.states.ScanWS(
        ws_list=np.linspace(3.0, 30.0, n_s), wd=270.0, ti=0.08, rho=1.225
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=p0,
        xy_step=stp,
        n_turbines=n_t,
        turbine_models=["kTI_04", "TOYT"],
    )

    algo = foxes.algorithms.Downwind(
        mbook,
        farm,
        states=states,
        rotor_model="centre",
        turbine_order="order_wd",
        wake_models=["Bastankhah_linear"],
        wake_frame="rotor_wd",
        partial_wakes_model="rotor_points",
        chunks=cks,
    )

    time0 = time.time()

    with ProgressBar():
        fresults = algo.calc_farm(vars_to_amb=[FV.REWS, FV.TI, FV.P])

    time1 = time.time()
    print("\nCalc time =", time1 - time0, "\n")

    print(fresults)
    df = fresults.to_dataframe()
    print(df[[FV.WD, FV.AMB_TI, FV.TI, FV.AMB_REWS, FV.REWS, FV.AMB_P, FV.P]])

    points = np.zeros((n_s, n_p, 3))
    points[:, :, 0] = np.linspace(p0[0], p0[0] + n_s * stp[0] + 10 * D, n_p)[None, :]
    points[:, :, 1] = p0[1]
    points[:, :, 2] = h
    print("\nPOINTS:\n", points[0])

    time0 = time.time()

    with ProgressBar():
        presults = algo.calc_points(fresults, points, vars_to_amb=[FV.WS, FV.TI])

    time1 = time.time()
    print("\nCalc time =", time1 - time0, "\n")

    print(presults)

    for s in range(points.shape[0]):
        plt.plot(points[s, :, 0], presults[FV.WS][s, :])
    plt.show()
