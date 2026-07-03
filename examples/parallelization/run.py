import matplotlib.pyplot as plt

import foxes
import foxes.variables as FV

if __name__ == "__main__":
    n_times = 1000
    n_turbines = 50
    seed = 42

    sdata = foxes.input.states.create.random_timseries_data(
        n_times,
        seed=seed,
    )
    states = foxes.input.states.Timeseries(
        data_source=sdata,
        output_vars=[FV.WS, FV.WD, FV.TI, FV.RHO],
        fixed_vars={FV.RHO: 1.225, FV.TI: 0.02},
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_random(
        farm,
        n_turbines,
        min_dist=500,
        turbine_models=["DTU10MW"],
        seed=seed,
        verbosity=0,
    )

    sdata

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        wake_models=["Bastankhah2014"],
        verbosity=1,
    )

    farm_results = algo.calc_farm()
    farm_results.to_dataframe()

    algo = foxes.algorithms.Downwind(
        farm,
        states,
        wake_models=["Jensen_quadratic_k0075"],
        verbosity=0,
    )

    engine = foxes.Engine.new(
        "process",
        n_procs=4,
        chunk_size_states=200,
        chunk_size_points=5000,
    )

    with engine:
        farm_results = algo.calc_farm()

        o = foxes.output.FlowPlots2D(algo, farm_results)
        plot_data = o.get_states_data_xy(FV.WS, resolution=30, states_isel=[0])

    g = o.gen_states_fig_xy(plot_data, figsize=(6, 6))
    next(g)
    plt.show()
