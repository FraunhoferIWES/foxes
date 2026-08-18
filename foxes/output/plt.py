from __future__ import annotations

from matplotlib import pyplot
from typing import Any

from .output import Output


class plt(Output):
    """
    Class that runs plt commands


    """

    def __getattr__(self, name: str) -> Any:
        return getattr(pyplot, name)

    def savefig(self, fname: str, *args: Any, **kwargs: Any) -> None:
        fpath = super().get_fpath(fname)
        pyplot.savefig(fpath, *args, **kwargs)
