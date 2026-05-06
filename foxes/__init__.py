"""
Farm Optimization and eXtended yield Evaluation Software

"""

from .config import config as config
from .config import get_path as get_path
from .core import Engine as Engine
from .core import WindFarm as WindFarm
from .core import Turbine as Turbine
from .core import get_engine as get_engine
from .models import ModelBook as ModelBook

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


# Robust __version__ assignment for all supported Python versions
from pathlib import Path

# Try to import importlib.metadata (Python 3.8+), else importlib_metadata (backport)
try:
    from importlib.metadata import version as _pkg_version
except ImportError:
    try:
        from importlib_metadata import version as _pkg_version  # type: ignore
    except ImportError:
        _pkg_version = None

# Try to import tomllib (Python 3.11+), else tomli (backport)
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None


def _get_version():
    source_location = Path(__file__).parent
    pyproject = source_location.parent / "pyproject.toml"
    if tomllib is not None and pyproject.exists():
        try:
            with open(pyproject, "rb") as f:
                return tomllib.load(f)["project"]["version"]
        except Exception:
            pass

    if _pkg_version is not None:
        try:
            return _pkg_version(__package__ or __name__)
        except Exception:
            pass
    return "unknown"


__version__ = _get_version()
