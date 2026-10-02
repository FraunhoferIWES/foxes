import numpy as np
import pytest

import foxes
from foxes.core import Algorithm


def test_algorithm_update_n_turbines_tracks_farm_reset():
    farm = foxes.WindFarm()
    farm.add_turbine(foxes.Turbine([0.0, 0.0]), verbosity=0)

    algo = Algorithm(mbook=foxes.ModelBook(), farm=farm, verbosity=0)
    assert algo.n_turbines == 1

    new_turbines = [
        foxes.Turbine([0.0, 0.0]),
        foxes.Turbine([500.0, 0.0]),
        foxes.Turbine([1000.0, 0.0]),
    ]
    farm.reset_turbines(algo=algo, turbines=new_turbines)

    assert algo.n_turbines == 3


def test_reset_turbines_clears_algorithm_cached_data_and_chunk_store():
    farm = foxes.WindFarm()
    farm.add_turbine(foxes.Turbine([0.0, 0.0]), verbosity=0)

    algo = Algorithm(mbook=foxes.ModelBook(), farm=farm, verbosity=0)
    algo.loaded_data["coords"]["dummy_coord"] = [1, 2, 3]
    algo.loaded_data["data_vars"]["dummy_var"] = (("dummy",), np.array([1, 2, 3]))
    algo.chunk_store[(0, 0)] = {"dummy": 1}

    farm.reset_turbines(
        algo=algo,
        turbines=[foxes.Turbine([0.0, 0.0]), foxes.Turbine([500.0, 0.0])],
    )

    assert algo.n_turbines == 2
    assert algo.loaded_data == {"coords": {}, "data_vars": {}, "extra_data": {}}
    assert len(algo.chunk_store) == 0


@pytest.mark.parametrize(
    ("calculation", "launcher"),
    [
        ("calc_farm", "_launch_parallel_farm_calc"),
        ("calc_points", "_launch_parallel_points_calc"),
    ],
)
def test_algorithm_resets_running_after_calculation_failure(
    monkeypatch, calculation, launcher
):
    algo = Algorithm(mbook=foxes.ModelBook(), farm=foxes.WindFarm(), verbosity=0)

    def fail_calculation(*args, **kwargs):
        raise RuntimeError("calculation failed")

    monkeypatch.setattr(algo, launcher, fail_calculation)

    with pytest.raises(RuntimeError, match="calculation failed"):
        getattr(algo, calculation)()

    assert not algo.running
    algo.finalize()
