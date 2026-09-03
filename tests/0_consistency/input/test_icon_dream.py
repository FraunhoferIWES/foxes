from pathlib import Path
from typing import Any

from foxes.input.states.create import icon_dream


class _FailingGridEngine:
    def submit(self, func: Any, *args: Any, **kwargs: Any) -> str:
        return "grid-future"

    def await_result(self, future: str) -> int:
        assert future == "grid-future"
        return -1


def test_icon_dream_returns_failure_count_for_grid_failure(
    monkeypatch: Any, tmp_path: Path
) -> None:
    class FakeStaticData:
        def get_file_path(self, *args: Any, **kwargs: Any) -> Path:
            return tmp_path / "target_grid.txt"

    cdo_tmp_dir = tmp_path / "cdo_tmp"

    monkeypatch.setattr(icon_dream, "StaticData", FakeStaticData)
    monkeypatch.setattr(icon_dream, "get_engine", _FailingGridEngine)

    result = icon_dream.iconDream2foxes(
        out_dir=tmp_path / "out",
        region="northsea",
        min_year=2020,
        min_month=1,
        max_year=2020,
        max_month=1,
        skip_download=True,
        cdo_tmp_dir=cdo_tmp_dir,
        verbosity=0,
    )

    assert result == 1
    assert not cdo_tmp_dir.exists()
