"""
Algorithms define the main calculation routines.
"""

from .downwind.downwind import Downwind as Downwind
from .iterative.iterative import Iterative as Iterative
from .sequential import Sequential as Sequential

from . import downwind as downwind
from . import iterative as iterative
from . import sequential as sequential
