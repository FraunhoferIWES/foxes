import foxes
import foxes.variables as FV
import numpy as np
from xarray import Dataset

from foxes.algorithms.downwind.models.population import (
    PopulationModel,
    PopulationStates,
)
from foxes.core import States, MData, FData, TData
import foxes.constants as FC


class _FlyStatesMock(States):
    def __init__(self, n_states):
        super().__init__(load_mode="fly")
        self._n_states = n_states
        self.calls = []

    def size(self):
        return self._n_states

    def output_point_vars(self, algo):
        return []

    def load_chunk_data(self, algo, mdata, fdata, tdata):
        i0 = mdata.states_i0(counter=True)
        n_states = mdata.n_states
        self.calls.append((i0, n_states))

        mdata["mock_var"] = np.arange(i0, i0 + n_states, dtype=np.int32)
        mdata.dims["mock_var"] = (FC.STATE,)

    def calculate(self, algo, mdata, fdata, tdata):
        return {}


class _LoadedStatesMock(_FlyStatesMock):
    def load_data(self, algo, loaded_data, force=False, verbosity=0):
        super().load_data(algo, loaded_data, force=force, verbosity=verbosity)
        loaded_data["data_vars"]["state_data"] = (
            (FC.STATE,),
            np.arange(self.size()),
        )


def test_population_states_load_chunk_data_fly():
    states = _FlyStatesMock(5)
    pstates = PopulationStates(states, n_pop=2)
    loaded_data = {"coords": {}, "data_vars": {}, "extra_data": {}}
    pstates.load_data(None, loaded_data)

    mdata = MData(
        data={
            FC.STATE: np.arange(4, dtype=np.int32),
            pstates.SMAP: np.array([3, 4, 0, 1], dtype=np.int32),
        },
        dims={
            FC.STATE: (FC.STATE,),
            pstates.SMAP: (FC.STATE,),
        },
        states_i0=3,
        name="mdata_test",
    )
    fdata = FData.from_sizes(4, 1)
    tdata = TData.from_points(np.zeros((4, 1, 3), dtype=np.float64), mdata=mdata)

    pstates.load_chunk_data(None, mdata, fdata, tdata)

    assert states.calls == [(0, 4)]
    assert pstates.STATE0 in mdata
    assert mdata.dims[pstates.STATE0] == (pstates.STATE0,)
    assert "mock_var" not in mdata


def test_population_states_load_chunk_data_preload_is_noop():
    states = _FlyStatesMock(5)
    pstates = PopulationStates(states, n_pop=2)
    pstates.load_mode = "preload"
    loaded_data = {"coords": {}, "data_vars": {}, "extra_data": {}}
    pstates.load_data(None, loaded_data)

    mdata = MData(
        data={
            FC.STATE: np.arange(3, dtype=np.int32),
            pstates.SMAP: np.array([0, 1, 2], dtype=np.int32),
        },
        dims={
            FC.STATE: (FC.STATE,),
            pstates.SMAP: (FC.STATE,),
        },
        states_i0=0,
        name="mdata_test",
    )

    pstates.load_chunk_data(None, mdata, None, None)

    assert states.calls == []
    assert "mock_var" not in mdata


def test_population_states_ignores_data_without_dimensions():
    states = _FlyStatesMock(2)
    pstates = PopulationStates(states, n_pop=2)
    loaded_data = {"coords": {}, "data_vars": {}, "extra_data": {}}
    pstates.load_data(None, loaded_data)

    mdata = MData(
        data={
            FC.STATE: np.arange(2, dtype=np.int32),
            pstates.SMAP: np.array([0, 1], dtype=np.int32),
        },
        dims={
            FC.STATE: (FC.STATE,),
            pstates.SMAP: (FC.STATE,),
        },
        states_i0=0,
        name="mdata_test",
    )
    fdata = FData(
        data={
            FC.STATE: np.arange(2, dtype=np.int32),
            FC.TURBINE: np.arange(1, dtype=np.int32),
        },
        dims={
            FC.STATE: (FC.STATE,),
            FC.TURBINE: (FC.TURBINE,),
        },
    )
    fdata[("weight", 1)] = np.ones((2, 1))
    tdata = TData.from_points(np.zeros((2, 1, 3)), mdata=mdata)
    tdata.add("weight", np.ones((2, 1, 1)), (FC.STATE, FC.TARGET, FC.TPOINT))

    pstates.calculate(None, mdata, fdata, tdata)


def test_population_states_reapplies_mapping_after_reinitialization():
    states = _LoadedStatesMock(2)
    population = PopulationStates(states, n_pop=3)
    loaded_data = population.initialize(None)

    population.finalize(None)
    population.initialize(None, loaded_data)

    assert loaded_data["data_vars"]["state_data"][0] == (population.STATE0,)
    np.testing.assert_array_equal(
        loaded_data["data_vars"][population.SMAP][1],
        np.array([0, 1, 0, 1, 0, 1]),
    )


def test_population_model_data_matches_population_major_order():
    states = _FlyStatesMock(2)
    population_states = PopulationStates(states, n_pop=3)
    population_data = np.array(
        [
            [10.0, 11.0],
            [20.0, 21.0],
            [30.0, 31.0],
        ]
    )
    population_model = PopulationModel(
        Dataset({"k": (("index", FC.TURBINE), population_data)}),
        verbosity=0,
    )

    class _AlgoMock:
        states = population_states
        n_turbines = 2

    loaded_data = {"coords": {}, "data_vars": {}, "extra_data": {}}
    population_model.load_data(_AlgoMock(), loaded_data)

    actual = loaded_data["data_vars"][population_model.DATA][1][..., 0]
    expected = np.repeat(population_data, states.size(), axis=0)
    np.testing.assert_array_equal(actual, expected)

    farm_results = Dataset(
        {"k": ((FC.STATE, FC.TURBINE), actual)},
    )
    pop_results = population_model.farm2pop_results(_AlgoMock(), farm_results)
    assert pop_results["k"].dims == (FC.POP, FC.STATE, FC.TURBINE)
    np.testing.assert_array_equal(
        pop_results["k"],
        np.repeat(population_data[:, None, :], states.size(), axis=1),
    )


def test_population_model_calc_points_after_calc_farm():
    farm = foxes.WindFarm()
    foxes.input.farm_layout.add_row(
        farm=farm,
        xy_base=np.array([0.0, 0.0]),
        xy_step=np.array([250.0, 0.0]),
        n_turbines=2,
        turbine_models=["NREL5MW"],
    )
    states = foxes.input.states.ScanStates(
        scans={
            FV.WS: [8.0, 10.0],
            FV.WD: [270.0],
            FV.TI: [0.08],
            FV.RHO: [1.225],
        }
    )
    population_data = Dataset(
        {
            FV.K: (
                ("candidate", FC.TURBINE),
                np.array([[0.03, 0.03], [0.07, 0.07]]),
            )
        }
    )
    algo = foxes.algorithms.Downwind(
        farm,
        states,
        rotor_model="centre",
        wake_models=["Bastankhah2014_linear"],
        population_params={
            "data_source": population_data,
            "index_coord": "candidate",
        },
        verbosity=0,
    )

    with foxes.Engine.new("default", verbosity=0):
        farm_results = algo.calc_farm()
        points = np.zeros((4, 1, 3))
        points[:, 0] = [100.0, 50.0, 80.0]
        point_results = algo.calc_points(farm_results, points)

    np.testing.assert_array_equal(
        farm_results[FV.K].values[:, 0],
        np.array([0.03, 0.03, 0.07, 0.07]),
    )
    assert point_results.sizes[FC.STATE] == 4
