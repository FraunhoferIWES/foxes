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


ROTOR_POINTS = "rotor_points"
""" Identifier for rotor points
:group: foxes.constants
"""

ROTOR_WEIGHTS = "rotor_weights"
""" Identifier for rotor point weights
:group: foxes.constants
"""
AMB_ROTOR_RES = "amb_rotor_res"
""" Identifier for ambient rotor point results
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

DTYPE = "DTYPE"
"""Identifier for default double data type
:group: foxes.constants
"""

ITYPE = "ITYPE"
"""Identifier for default integer data type
:group: foxes.constants
"""

BLOCK_CONVERGENCE = "block_convergence"
"""Identifier for convergence blocking signal
:group: foxes.constants
"""


KAPPA = 0.41
""" The Von Karman constant
:group: foxes.constants
"""


W = "W"
""" The unit watt
:group: foxes.constants
"""

kW = "kW"
""" The unit kilowatt
:group: foxes.constants
"""

MW = "MW"
""" The unit megawatt
:group: foxes.constants
"""

GW = "GW"
""" The unit gigawatt
:group: foxes.constants
"""

TW = "TW"
""" The unit terawatt
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


WORK_DIR = "work_dir"
"""Identifier for the working directory
:group: foxes.constants
"""

OUT_DIR = "out_dir"
"""Identifier for the default output directory
:group: foxes.constants
"""
