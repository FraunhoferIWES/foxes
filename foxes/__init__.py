"""
Farm Optimization and eXtended yield Evaluation Software
"""
from .core import WindFarm, Turbine # noqa: F401
from .models import ModelBook # noqa: F401
from .data import parse_Pct_file_name, FARM, STATES, PCTCURVE, StaticData # noqa: F401

from . import algorithms # noqa: F401
from . import models # noqa: F401
from . import input # noqa: F401
from . import output # noqa: F401
from . import tools # noqa: F401

try:
    from importlib_resources import read_text
except ModuleNotFoundError:
    from importlib.resources import read_text

__version__ = read_text(__package__, "VERSION")
