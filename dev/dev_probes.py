
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
    n_p = 22
    p0  = np.array([0., 0.])
    stp = np.array([500., 0.])
    cks = None#FV.STATE: 5, FV.POINT:5}
    D   = 120.
    H   = 100.
    h   = 100.

    dask.config.set(scheduler='synchronous')
    #dask.config.set(scheduler='threads')
    #dask.config.set(scheduler='processes')
    #client = Client(n_workers=4, threads_per_worker=1) 
    #print(f"\n{client}")
    #print(f"Dashboard: {client.dashboard_link}\n")

    mbook = foxes.models.ModelBook()
    mbook.turbine_types["TOYT"] = foxes.models.turbine_types.PCtFile(
                                    name="TOYT", filepath="toyTurbine.csv", D=D, H=H)

    states = foxes.input.states.ScanWS(
        ws_list=np.linspace(3., 30., n_s),
        wd=90.,
        ti=0.08,
        rho=1.225
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=p0, 
        xy_step=stp, 
        n_turbines=n_t,
        turbine_models=["kTI_04", "TOYT"]
    )
    
    algo = foxes.algorithms.Downwind(
                mbook,
                farm,
                states=states,
                rotor_model="centre",
                turbine_order="order_wd",
                wake_models=['Bastankhah_linear'],
                wake_frame="mean_wd",
                partial_wakes_model="rotor_points",
                chunks=cks
            )

    time0 = time.time()
    
    with ProgressBar():
        fdata = algo.calc_farm()

    time1 = time.time()
    print("\nCalc time =",time1 - time0, "\n")

    print(fdata)
    
    points          = np.zeros((n_s, n_p, 3))
    points[:, :, 0] = np.linspace(p0[0], p0[0] + 10*D, n_p)[None, :]
    points[:, :, 1] = p0[1]
    points[:, :, 2] = h
    print("\nPOINTS:\n", points[0])

    time0 = time.time()

    with ProgressBar():
        pdata = algo.calc_points(fdata, points)

    time1 = time.time()
    print("\nCalc time =",time1 - time0, "\n")

    print(pdata)

    for s in range(points.shape[0]):
        plt.plot(points[s, :, 0], pdata[FV.WS][s, :])
    plt.show()
    

