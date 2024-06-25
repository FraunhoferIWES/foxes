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

TIME = "time"
""" Time identifier
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

TARGET = "target"
""" Target identifier
:group: foxes.constants
"""

TARGETS = "targets"
""" Targets identifier
:group: foxes.constants
"""

TPOINT = "target_point"
""" Target point identifier
:group: foxes.constants
"""

TPOINTS = "target_points"
""" Points per target identifier
:group: foxes.constants
"""

TWEIGHTS = "tpoint_weights"
""" Target point weights identifier
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

AMB_TARGET_RESULTS = "amb_target_res"
""" Identifier for ambient target results
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

STATE_TURBINE = "state-turbine"
"""Identifier for state-turbine dimensions
:group: foxes.constants
"""

STATE_TARGET = "state-target"
"""Identifier for state-target dimensions
:group: foxes.constants
"""

STATE_TARGET_TPOINT = "state-target-tpoint"
"""Identifier for state-target-tpoints dimensions
:group: foxes.constants
"""

STATE_SOURCE_ORDERI = "state-source-orderi"
"""Identifier for order index of wake causing turbines
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
