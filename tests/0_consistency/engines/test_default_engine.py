from types import SimpleNamespace

import foxes.constants as FC
from foxes.engines.default import DefaultEngine


class _Algo:
    def __init__(self, n_states, n_turbines):
        self.n_states = n_states
        self.n_turbines = n_turbines


class _FakeDelegatedEngine:
    def __init__(self, engine_type):
        self.engine_type = engine_type
        self.entered = False
        self.called = []

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *args):
        self.entered = False

    def submit(self, f, *args, **kwargs):
        self.called.append(("submit", self.engine_type))
        return ("submit", self.engine_type)

    def future_is_done(self, future):
        self.called.append(("future_is_done", self.engine_type))
        return True

    def await_result(self, future):
        self.called.append(("await_result", self.engine_type))
        return ("await_result", self.engine_type)

    def map(self, func, inputs, *args, **kwargs):
        self.called.append(("map", self.engine_type))
        return [func(i, *args, **kwargs) for i in inputs]

    def new_runner(self):
        self.called.append(("new_runner", self.engine_type))
        return self.engine_type

    def run_calculation(self, *args, **kwargs):
        self.called.append(("run_calculation", self.engine_type))
        return self.engine_type


def test_default_engine_selects_single_vs_process_by_condition():
    eng = DefaultEngine(n_procs=2, verbosity=0)

    assert eng._select_engine_name() == "process"

    small_algo = _Algo(n_states=10, n_turbines=10)
    assert eng._select_engine_name(algo=small_algo, point_data=None) == "single"

    large_algo = _Algo(n_states=1000, n_turbines=10)
    assert eng._select_engine_name(algo=large_algo, point_data=None) == "process"

    point_data = SimpleNamespace(sizes={FC.TARGET: 2000})
    assert eng._select_engine_name(algo=small_algo, point_data=point_data) == "process"


def test_default_engine_run_calculation_uses_selected_engine(monkeypatch):
    selected = []

    def _fake_new(engine_type, *args, **kwargs):
        selected.append(engine_type)
        return _FakeDelegatedEngine(engine_type)

    monkeypatch.setattr("foxes.engines.default.Engine.new", _fake_new)

    eng = DefaultEngine(n_procs=2, verbosity=0)
    monkeypatch.setattr(eng, "_select_engine_name", lambda **kwargs: "single")

    algo = _Algo(n_states=10, n_turbines=10)
    result = eng.run_calculation(algo, object(), object(), point_data=None)

    assert result == "single"
    assert selected == ["single"]


def test_default_engine_non_calc_methods_fallback_to_process(monkeypatch):
    selected = []

    def _fake_new(engine_type, *args, **kwargs):
        selected.append(engine_type)
        return _FakeDelegatedEngine(engine_type)

    monkeypatch.setattr("foxes.engines.default.Engine.new", _fake_new)

    eng = DefaultEngine(n_procs=2, verbosity=0)

    assert eng.submit(lambda: 1) == ("submit", "process")
    assert eng.future_is_done(object()) is True
    assert eng.await_result(object()) == ("await_result", "process")
    assert eng.map(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]
    assert eng.new_runner() == "process"

    assert selected == ["process", "process", "process", "process", "process"]
