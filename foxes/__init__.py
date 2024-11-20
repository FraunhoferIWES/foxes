"""
Farm Optimization and eXtended yield Evaluation Software
    
"""

from .config import config, get_path  # noqa: F401
from .core import Engine, WindFarm, Turbine, get_engine, reset_engine  # noqa: F401
from .models import ModelBook  # noqa: F401
from .data import (
    parse_Pct_file_name,
    parse_Pct_two_files,
    FARM,
    STATES,
    PCTCURVE,
    StaticData,
)  # noqa: F401

from . import algorithms  # noqa: F401
from . import engines  # noqa: F401
from . import models  # noqa: F401
from . import input  # noqa: F401
from . import output  # noqa: F401
from . import utils  # noqa: F401

import importlib
from pathlib import Path

try:
    tomllib = importlib.import_module("tomllib")
    source_location = Path(__file__).parent
    if (source_location.parent / "pyproject.toml").exists():
        with open(source_location.parent / "pyproject.toml", "rb") as f:
            __version__ = tomllib.load(f)["project"]["version"]
    else:
        __version__ = importlib.metadata.version(__package__ or __name__)
except ModuleNotFoundError:
    __version__ = importlib.metadata.version(__package__ or __name__)
