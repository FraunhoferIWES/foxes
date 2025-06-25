"""
Wake superposition models.
"""

from .ws_linear import WSLinear as WSLinear
from .ws_linear import WSLinearLocal as WSLinearLocal

from .ws_pow import WSPow as WSPow
from .ws_pow import WSPowLocal as WSPowLocal

from .ws_max import WSMax as WSMax
from .ws_max import  WSMaxLocal as WSMaxLocal

from .ws_quadratic import WSQuadratic as WSQuadratic
from .ws_quadratic import WSQuadraticLocal as WSQuadraticLocal

from .wind_vector import WindVectorLinear as WindVectorLinear
from .ws_product import WSProduct as WSProduct
from .ti_linear import TILinear as TILinear
from .ti_quadratic import TIQuadratic as TIQuadratic
from .ti_pow import TIPow as TIPow
from .ti_max import TIMax as TIMax
