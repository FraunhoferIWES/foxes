import numpy as np
import pytest

import foxes.constants as FC
import foxes.variables as FV
from foxes.algorithms.downwind.models.population import PopulationStates
from foxes.core import FData, MData, States, SubsetStates, TData


class _MockStates(States):
    def __init__(self, n_states, load_mode="preload"):
        super().__init__(load_mode=load_mode)
        self._n_states = n_states
        self.calls = []

    def size(self):
        return self._n_states

    def index(self):
        return [10 * (i + 1) for i in range(self._n_states)]

    def output_point_vars(self, algo):
        return ["mock_var"]

    def load_data(self, algo, loaded_data, force=False, verbosity=0):
        super().load_data(algo, loaded_data, force=force, verbosity=verbosity)
        loaded_data["coords"][FC.STATE] = np.arange(self._n_states)
        loaded_data["data_vars"]["mock_var"] = (
            (FC.STATE,),
            np.arange(self._n_states, dtype=np.float64),
        )
        loaded_data["data_vars"][FV.WEIGHT] = (
            (FC.STATE,),
            np.arange(1, self._n_states + 1, dtype=np.float64) / 10.0,
        )

    def load_chunk_data(self, algo, mdata, fdata, tdata):
        if self.load_mode != "preload" and "fly_var" not in mdata:
            source_i0 = mdata.states_i0(counter=True)
            self.calls.append((source_i0, mdata.n_states))
            mdata["fly_var"] = np.arange(
                source_i0, source_i0 + mdata.n_states, dtype=np.int32
            )
            mdata.dims["fly_var"] = (FC.STATE,)

    def calculate(self, algo, mdata, fdata, tdata):
        return {"mock_var": mdata["mock_var"][:, None, None]}


class _LazySizeStates(_MockStates):
    def __init__(self):
        super().__init__(0)

    def load_data(self, algo, loaded_data, force=False, verbosity=0):
        self._n_states = 4
        super().load_data(algo, loaded_data, force=force, verbosity=verbosity)


def test_subset_states_validates_indices():
    states = _MockStates(4)

    with pytest.raises(ValueError, match="empty"):
        SubsetStates(states, [])
    with pytest.raises(ValueError, match="one-dimensional"):
        SubsetStates(states, [[0, 1]])
    with pytest.raises(TypeError, match="integer"):
        SubsetStates(states, [0.0, 1.0])
    with pytest.raises(IndexError, match="out of range"):
        SubsetStates(states, [4])
    with pytest.raises(ValueError, match="duplicate"):
        SubsetStates(states, [1, 1])


def test_subset_states_preserves_order_index_and_weights():
    states = _MockStates(4)
    subset = SubsetStates(states, [3, -3])
    loaded_data = subset.initialize(None)

    assert subset.size() == 2
    assert subset.index() == [40, 20]
    np.testing.assert_array_equal(
        loaded_data["data_vars"][subset.SMAP][1], np.array([3, 1])
    )
    weight_dims, weights = loaded_data["data_vars"][FV.WEIGHT]
    assert weight_dims == (subset.STATE0,)
    np.testing.assert_allclose(weights, [0.1, 0.2, 0.3, 0.4])


def test_subset_states_defers_validation_for_lazy_source():
    subset = SubsetStates(_LazySizeStates(), [3, -3])

    loaded_data = subset.initialize(None)

    assert subset.index() == [40, 20]
    np.testing.assert_array_equal(
        loaded_data["data_vars"][subset.SMAP][1], np.array([3, 1])
    )


def test_subset_states_gathers_fly_data_in_selection_order():
    states = _MockStates(6, load_mode="fly")
    subset = SubsetStates(states, [4, 1, 3])
    loaded_data = subset.initialize(None)
    assert loaded_data is not None

    mdata = MData(
        data={
            FC.STATE: np.arange(3, dtype=np.int32),
            subset.SMAP: np.array([4, 1, 3], dtype=np.int32),
        },
        dims={FC.STATE: (FC.STATE,), subset.SMAP: (FC.STATE,)},
        states_i0=0,
        name="subset_test",
    )
    fdata = FData.from_sizes(3, 1)
    tdata = TData.from_points(np.zeros((3, 1, 3)), mdata=mdata)

    subset.load_chunk_data(None, mdata, fdata, tdata)

    assert states.calls == [(1, 4)]
    np.testing.assert_array_equal(mdata["fly_var"], [4, 1, 3])
    assert mdata.dims["fly_var"] == (FC.STATE,)


def test_subset_states_composes_with_population_states():
    states = _MockStates(5)
    subset = SubsetStates(states, [3, 1])
    population = PopulationStates(subset, n_pop=2)
    loaded_data = population.initialize(None)

    assert population.size() == 4
    assert subset.size() == 2
    np.testing.assert_array_equal(
        loaded_data["data_vars"][subset.SMAP][1], np.array([3, 1])
    )
    np.testing.assert_array_equal(
        loaded_data["data_vars"][population.SMAP][1], np.array([0, 0, 1, 1])
    )
