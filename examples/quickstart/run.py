import foxes

if __name__ == "__main__":

    states = foxes.input.states.Timeseries(
        "timeseries_3000.csv.gz", ["WS", "WD", "TI", "RHO"]
    )

    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_from_file(
        farm, "test_farm_67.csv", turbine_models=["NREL5MW"]
    )

    algo = foxes.algorithms.Downwind(farm, states, ["Jensen_linear_k007"])
    farm_results = algo.calc_farm()

    print(farm_results)
