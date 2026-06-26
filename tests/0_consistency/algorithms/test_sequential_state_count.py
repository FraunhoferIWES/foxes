import inspect

from foxes.algorithms.sequential.sequential import Sequential


def test_sequential_does_not_override_global_n_states():
    """Sequential steps must not mutate the algorithm-global state count."""
    source = inspect.getsource(Sequential.__next__)
    assert "self.n_states = 1" not in source
    assert "self.n_states = len(self._inds)" not in source
