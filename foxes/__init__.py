"""
Farm Optimization and eXtended yield Evaluation Software
    
"""

from .core import WindFarm, Turbine  # noqa: F401
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
from . import models  # noqa: F401
from . import input  # noqa: F401
from . import output  # noqa: F401
from . import utils  # noqa: F401

try:
    from importlib.resources import files

    __version__ = files(__package__).joinpath("VERSION").read_text()
except ImportError:
    from importlib.resources import read_text

    __version__ = read_text(__package__, "VERSION")
