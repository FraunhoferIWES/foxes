"""
Farm Optimization and eXtended yield Evaluation Software
"""
from .core import WindFarm, Turbine
from .models import ModelBook
from .data import data_book, parse_Pct_file_name

from . import algorithms
from . import models
from . import input
from . import output
from . import tools

from importlib.resources import read_text
__version__ = read_text(__package__, "VERSION")
