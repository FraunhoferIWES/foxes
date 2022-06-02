
import numpy as np
import argparse
import dask
import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("n_t", help="The number of turbines", type=int)
    parser.add_argument("--ws", help="The wind speed", type=float, default=9.0)
    parser.add_argument("--wd", help="The wind direction", type=float, default=270.0)
    parser.add_argument("--ti", help="The TI value", type=float, default=0.08)
    parser.add_argument("--rho", help="The air density", type=float, default=1.225)
    parser.add_argument("-d", "--dist_x", help="The turbine distance in x", type=float, default=500.0)
    parser.add_argument("-w", "--wakes", help="The wake models", default=['Bastankhah_linear'], nargs='+')
    parser.add_argument("-r", "--rotor", help="The rotor model", default="centre")
    parser.add_argument("-p", "--pwakes", help="The partial wakes model", default="rotor_points")
    parser.add_argument("-m", "--tmodels", help="The turbine models", default=["kTI_02", "TOYT"], nargs='+')
    parser.add_argument("-dy", "--deltay", help="Turbine layout y step", type=float, default=0.)
    parser.add_argument("-v", "--var", help="The plot variable", default=FV.WS)
    args  = parser.parse_args()

    p0  = np.array([0., 0.])
    stp = np.array([args.dist_x, args.deltay])
    cks = None

    with dask.config.set(scheduler=None):

        mbook = foxes.ModelBook()
        mbook.turbine_types["TOYT"] = foxes.models.turbine_types.PCtFile(
                                        name="TOYT", filepath="toyTurbine.csv", 
                                        D=100., H=100.)

        states = foxes.input.states.SingleStateStates(
            ws=args.ws,
            wd=args.wd,
            ti=args.ti,
            rho=args.rho
        )

        farm = foxes.WindFarm()
        foxes.input.farm_layout.add_row(
            farm=farm,
            xy_base=p0, 
            xy_step=stp, 
            n_turbines=args.n_t,
            turbine_models=args.tmodels
        )
        
        algo = foxes.algorithms.Downwind(
                    mbook,
                    farm,
                    states=states,
                    rotor_model=args.rotor,
                    turbine_order="order_wd",
                    wake_models=args.wakes,
                    wake_frame="mean_wd",
                    partial_wakes_model=args.pwakes,
                    chunks=cks
                )
        
        farm_results = algo.calc_farm()

        print("\nResults data:\n", farm_results)
        
        o   = foxes.output.FlowPlots2D(algo, farm_results)
        g   = o.gen_states_fig_horizontal("DELTA_WS", resolution=10)
        fig = next(g)
        plt.show()
        plt.close(fig)

        o   = foxes.output.FlowPlots2D(algo, farm_results)
        g   = o.gen_states_fig_vertical(args.var, resolution=10, x_direction=90.)
        fig = next(g)
        plt.show()

        fr = farm_results.to_dataframe()
        print()
        print(fr[[FV.X, FV.WD, FV.AMB_REWS, FV.REWS, 
                FV.AMB_TI, FV.TI, FV.AMB_P, FV.P, FV.CT]])
