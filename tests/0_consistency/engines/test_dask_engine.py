import numpy as np

from foxes.core import MData, FData
from foxes.engines.dask import DaskProcessRunner, LocalClusterEngine


class _DummyAlgo:
    def __init__(self):
        self.verbosity = 0
        self.n_turbines = 1
        self._chunk_store = {}

    def reset_chunk_store(self, chunk_store=None):
        if chunk_store is not None:
            self._chunk_store = chunk_store
        out = self._chunk_store
        self._chunk_store = {}
        return out


class _DummyModel:
    def calculate(self, algo, mdata, fdata, tdata=None, **cpars):
        assert isinstance(mdata["A"], np.ndarray)
        assert isinstance(fdata.extra_data["payload"]["nested"][0], np.ndarray)
        assert isinstance(cpars["vector"], np.ndarray)
        return {"out": np.array([mdata["A"].sum()], dtype=np.float64)}


class _FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class _FakeClient:
    def __init__(self):
        self.scatter_calls = []
        self.submit_calls = []

    def scatter(self, arr, broadcast=False, hash=False):
        self.scatter_calls.append(
            {
                "shape": tuple(arr.shape),
                "dtype": str(arr.dtype),
                "nbytes": int(arr.nbytes),
                "broadcast": broadcast,
                "hash": hash,
            }
        )
        return _FakeFuture(np.array(arr, copy=True))

    def submit(self, f, *args, **kwargs):
        self.submit_calls.append((f, args, kwargs))
        return "submitted"

    def __del__(self):
        return None


def test_local_cluster_submit_futureizes_large_arrays(monkeypatch):
    monkeypatch.setattr("foxes.engines.dask.load_dask", lambda: None)
    monkeypatch.setattr("foxes.engines.dask.load_distributed", lambda: None)

    engine = LocalClusterEngine(
        n_procs=2,
        verbosity=0,
        min_submit_array_bytes=64,
    )
    engine._client = _FakeClient()

    mdata = MData(
        data={
            "A": np.arange(32, dtype=np.float64),
            "B": np.arange(4, dtype=np.float64),
        },
        dims={"A": ("u",), "B": ("v",)},
        extra_data={
            "large": np.arange(32, dtype=np.float64),
            "small": np.arange(4, dtype=np.float64),
        },
        name="chunk",
    )
    payload = {
        "nested": [np.arange(32, dtype=np.float64), np.arange(2, dtype=np.float64)]
    }

    out = engine.submit(lambda *a, **k: None, mdata, payload)

    assert out == "submitted"
    assert len(engine._client.scatter_calls) == 2

    _, submitted_args, _ = engine._client.submit_calls[-1]
    submitted_mdata = submitted_args[0]
    submitted_payload = submitted_args[1]

    assert isinstance(submitted_mdata["A"], np.ndarray)
    assert isinstance(submitted_mdata["B"], np.ndarray)
    assert isinstance(submitted_mdata.extra_data["large"], _FakeFuture)
    assert isinstance(submitted_mdata.extra_data["small"], np.ndarray)
    assert isinstance(submitted_payload["nested"][0], _FakeFuture)
    assert isinstance(submitted_payload["nested"][1], np.ndarray)


def test_dask_runner_resolves_nested_future_payloads():
    runner = DaskProcessRunner()

    mdata = MData(
        data={"A": np.arange(8, dtype=np.float64)},
        dims={"A": ("u",)},
        name="chunk",
    )
    mdata["A"] = _FakeFuture(np.arange(8, dtype=np.float64))
    fdata = FData.from_sizes(8, 1)
    fdata.extra_data = {
        "payload": {"nested": [_FakeFuture(np.arange(5, dtype=np.float64))]}
    }

    results, cstore = runner.run(
        _DummyAlgo(),
        _DummyModel(),
        mdata,
        fdata,
        None,
        shared=None,
        chunk_store={},
        chunk_key=(0, 0),
        out_dims=("u",),
        write_nc=None,
        vector=_FakeFuture(np.arange(3, dtype=np.float64)),
    )

    assert np.isclose(results["out"][0], np.sum(np.arange(8, dtype=np.float64)))
    assert cstore == {}
