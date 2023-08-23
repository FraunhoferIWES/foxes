import numpy as np

XYH = "xyh"
""" The vector (x, y, height)
:group: foxes.variables
"""

FARM = "farm"
""" Wind farm identifier
:group: foxes.constants
"""

STATE = "state"
""" States identifier
:group: foxes.constants
"""

TURBINE = "turbine"
""" Wind turbine identifier
:group: foxes.constants
"""

TNAME = "tname"
""" Wind turbine name identifier
:group: foxes.constants
"""

POINT = "point"
""" Point identifier
:group: foxes.constants
"""

POINTS = "points"
""" Points identifier
:group: foxes.constants
"""

RPOINT = "rotor_point"
""" Rotor point identifier
:group: foxes.constants
"""

RPOINTS = "rotor_points"
""" Rotor points identifier
:group: foxes.constants
"""

RWEIGHTS = "rotor_weights"
""" Rotor point weights identifier
:group: foxes.constants
"""

AMB_RPOINT_RESULTS = "amb_rpoint_res"
""" Identified for ambient rotor point results
:group: foxes.constants
"""


VARS = "vars"
""" Variables identifier
:group: foxes.constants
"""

VALID = "valid"
""" Validity identifier
:group: foxes.constants
"""

TMODELS = "tmodels"
""" Turbine models identifier
:group: foxes.constants
"""

TMODEL_SELS = "tmodel_sels"
"""Selected turbine models identifier
:group: foxes.constants
"""

STATES_SEL = "states_sel"
"""Identifier for states selection
:group: foxes.constants
"""

STATE_SOURCE_TURBINE = "state_source_turbine"
"""Identifier for the source turbines per state
:group: foxes.constants
"""

STATE_TURBINE = "state-turbine"
"""Identifier for state-turbine dimensions
:group: foxes.constants
"""

STATE_POINT = "state-point"
"""Identifier for state-point dimensions
:group: foxes.constants
"""

DTYPE = np.float64
""" Default data type for floats
:group: foxes.constants
"""

ITYPE = np.int64
""" Default data type for int
:group: foxes.constants
"""


KAPPA = 0.41
""" The Van-Karman constant
:group: foxes.constants
"""


W = "W"
""" The unit watt
:group: foxes.constants
"""

kW = "kW"
""" The unit kilo watt
:group: foxes.constants
"""

MW = "MW"
""" The unit mega watt
:group: foxes.constants
"""

GW = "GW"
""" The unit giga watt
:group: foxes.constants
"""

TW = "TW"
""" The unit terra watt
:group: foxes.constants
"""

P_UNITS = {W: 1.0, kW: 1.0e3, MW: 1.0e6, GW: 1.0e9, TW: 1.0e12}
""" Power unit factors relative to watts,
key: unit str, value: factor
:group: foxes.constants
"""


POP = "pop"
""" Population identifier
:group: foxes.constants
"""
