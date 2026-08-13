import numpy as np

from foxes.algorithms.downwind.models.population import PopulationStates
from foxes.core import States, MData
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

    pstates.load_chunk_data(None, mdata, None, None)

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
