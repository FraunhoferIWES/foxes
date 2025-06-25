"""
Farm Optimization and eXtended yield Evaluation Software

"""

from .config import config as config
from .config import get_path as get_path
from .core import Engine as Engine
from .core import WindFarm as WindFarm
from .core import Turbine as Turbine
from .core import get_engine as get_engine
from .core import reset_engine as reset_engine  
from .models import ModelBook  as ModelBook

from .data import parse_Pct_file_name as parse_Pct_file_name
from .data import parse_Pct_two_files as parse_Pct_two_files
from .data import FARM as FARM
from .data import STATES as STATES
from .data import PCTCURVE as PCTCURVE
from .data import StaticData as StaticData

from . import algorithms as algorithms
from . import engines as engines
from . import models as models
from . import input as input
from . import output as output
from . import utils as utils

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
