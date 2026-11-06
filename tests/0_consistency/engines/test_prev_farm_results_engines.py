import numpy as np
import pytest

from foxes.core import FData
import foxes.constants as FC
from foxes.engines.process import ProcessEngineRunner
from foxes.engines.threads import ThreadsEngineRunner
from foxes.engines.dask import DaskProcessRunner
from foxes.engines.single import SingleChunkEngineRunner


class _DummyMData:
    def __init__(self):
        self.extra_data = {FC.PREV_FARM_RESULTS: "token"}
        self.n_states = 3
        self._states_i0_calls = []

    def states_i0(self, counter=False):
        self._states_i0_calls.append(counter)
        return 4

    def recombine_with_shared(self, shared):
        return None


class _DummyAlgo:
    def __init__(self):
        self.verbosity = 0
        self.n_turbines = 2
        self.chunk_store = {}
        self.prev_calls = 0

    def prev_farm_results(self, mdata):
        self.prev_calls += 1
        return "prev-dataset"

    def reset_chunk_store(self, chunk_store=None):
        if chunk_store is not None:
            self.chunk_store = chunk_store
        out = self.chunk_store
        self.chunk_store = {}
        return out


class _EchoModel:
    def __init__(self):
        self.last_fdata = None
        self.last_rest = ()

    def calculate(self, algo, mdata, fdata, *rest, **cpars):
        self.last_fdata = fdata
        self.last_rest = rest
        return {"model_marker": np.array([1], dtype=np.int32)}


class _ReturnOutput:
    def calculate(self, algo, mdata, fdata, tdata, model_results):
        # Output consumers can explicitly retain both prepared data and model returns.
        return {**fdata, **model_results}


def _patch_fdata_from_dataset(monkeypatch):
    calls = []

    def _fake_from_dataset(cls, dataset, **kwargs):
        calls.append((dataset, kwargs))
        return {"prev_marker": np.array([7], dtype=np.int32)}

    monkeypatch.setattr(FData, "from_dataset", classmethod(_fake_from_dataset))
    return calls


@pytest.mark.parametrize(
    "runner_cls",
    [ProcessEngineRunner, ThreadsEngineRunner, DaskProcessRunner],
)
def test_process_family_runners_rebuild_and_merge_prev_farm_results(
    monkeypatch, runner_cls
):
    calls = _patch_fdata_from_dataset(monkeypatch)

    runner = runner_cls()

    algo = _DummyAlgo()
    model = _EchoModel()
    mdata = _DummyMData()
    input_fdata = {"input_marker": np.array([0], dtype=np.int32)}

    results, cstore = runner.run(
        algo,
        model,
        mdata,
        input_fdata,
        None,
        output=_ReturnOutput(),
        shared=None,
        chunk_store={},
        chunk_key=(0, 0),
        out_dims=("x",),
        utm_zone=None,
    )

    assert cstore == {}
    assert algo.prev_calls == (0 if runner_cls is DaskProcessRunner else 1)
    assert set(model.last_fdata.keys()) == {"prev_marker"}
    assert np.array_equal(
        model.last_fdata["prev_marker"], np.array([7], dtype=np.int32)
    )
    assert "prev_marker" in results
    assert "model_marker" in results

    assert len(calls) == 1
    dataset, kwargs = calls[0]
    assert dataset == "prev-dataset"
    assert kwargs["s_states"] == slice(4, 7, None)
    assert kwargs["states_i0"] == 4
    assert kwargs["n_states"] == 3
    assert kwargs["n_turbines"] == 2


def test_single_runner_rebuilds_and_merges_prev_farm_results(monkeypatch):
    calls = _patch_fdata_from_dataset(monkeypatch)

    runner = SingleChunkEngineRunner()

    algo = _DummyAlgo()
    model = _EchoModel()
    mdata = _DummyMData()
    input_fdata = {"input_marker": np.array([0], dtype=np.int32)}

    results, cstore = runner.run(
        algo,
        model,
        mdata,
        input_fdata,
        output=_ReturnOutput(),
        chunk_store={},
        shared=None,
        chunk_key=(0, 0),
        out_dims=("x",),
    )

    assert cstore == {}
    assert algo.prev_calls == 1
    assert set(model.last_fdata.keys()) == {"prev_marker"}
    assert np.array_equal(
        model.last_fdata["prev_marker"], np.array([7], dtype=np.int32)
    )
    assert "prev_marker" in results
    assert "model_marker" in results

    assert len(calls) == 1
    dataset, kwargs = calls[0]
    assert dataset == "prev-dataset"
    assert kwargs["s_states"] == slice(4, 7, None)


def test_single_runner_uses_previous_farm_data_without_output(monkeypatch):
    _patch_fdata_from_dataset(monkeypatch)

    runner = SingleChunkEngineRunner()

    algo = _DummyAlgo()
    model = _EchoModel()
    mdata = _DummyMData()

    # Prepared farm data remains required, while selecting an output is optional.
    results, cstore = runner.run(
        algo,
        model,
        mdata,
        {},
        shared=None,
        chunk_store={},
        chunk_key=(0, 0),
        out_dims=("x",),
    )
    assert results is None
    assert cstore == {}
    assert "prev_marker" in model.last_fdata
