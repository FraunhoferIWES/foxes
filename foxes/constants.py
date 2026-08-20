XY = "xy"
""" The vector (x, y)
"""

XYH = "xyh"
""" The vector (x, y, height)
"""

FARM = "wind_farm"
""" Wind farm identifier
"""

CLUSTER = "cluster"
""" Cluster identifier
"""

STATE = "state"
""" States identifier
"""

TIME = "time"
""" Time identifier
"""

TURBINE = "turbine"
""" Wind turbine identifier
"""

TNAME = "tname"
""" Wind turbine name identifier
"""

TARGET = "target"
""" Target identifier
"""

TARGETS = "targets"
""" Targets identifier
"""

TPOINT = "target_point"
""" Target point identifier
"""

TPOINTS = "target_points"
""" Points per target identifier
"""

TWEIGHTS = "tpoint_weights"
""" Target point weights identifier
"""

POINT = "point"
""" Point identifier
"""

POINTS = "points"
""" Points identifier
"""


ROTOR_POINTS = "rotor_points"
""" Identifier for rotor points
"""

ROTOR_POINT = "rotor_point"
""" Identifier for a rotor point
"""

ROTOR_WEIGHTS = "rotor_weights"
""" Identifier for rotor point weights
"""

AMB_ROTOR_RES = "amb_rotor_res"
""" Identifier for ambient rotor point results
"""

WEIGHT_RES = "weight_res"
""" Identifier for weights results at rotor points
"""


VARS = "vars"
""" Variables identifier
"""

VALID = "valid"
""" Validity identifier
"""

TMODELS = "tmodels"
""" Turbine models identifier
"""

TMODEL_SELS = "tmodel_sels"
"""Selected turbine models identifier
"""

STATES_SEL = "states_sel"
"""Identifier for states selection
"""

STATE_TURBINE = "state-turbine"
"""Identifier for state-turbine dimensions
"""

STATE_TARGET = "state-target"
"""Identifier for state-target dimensions
"""

STATE_TARGET_TPOINT = "state-target-tpoint"
"""Identifier for state-target-tpoints dimensions
"""

STATE_SOURCE_ORDERI = "state-source-orderi"
"""Identifier for order index of wake causing turbines
"""

DTYPE = "DTYPE"
"""Identifier for default double data type
"""

ITYPE = "ITYPE"
"""Identifier for default integer data type
"""

BLOCK_CONVERGENCE = "block_convergence"
"""Identifier for convergence blocking signal
"""

PREV_FARM_RESULTS = "prev_farm_results"
"""Identifier for previous iteration farm results
"""


WDEFL_ROT_ANGLE = "wake_deflection_rotation_angle"
"""Identifier for the wake deflection rotation angle data
"""

WDEFL_DWS_FACTOR = "wake_deflection_deltaws_factor"
"""Identifier for the wake deflection delta wind speed factor data
"""


KAPPA = 0.41
""" The Von Karman constant
"""

Rd = 287.052874
""" The specific gas constant for dry air
"""


W = "W"
""" The unit watt
"""

kW = "kW"
""" The unit kilowatt
"""

MW = "MW"
""" The unit megawatt
"""

GW = "GW"
""" The unit gigawatt
"""

TW = "TW"
""" The unit terawatt
"""

P_UNITS = {W: 1.0, kW: 1.0e3, MW: 1.0e6, GW: 1.0e9, TW: 1.0e12}
""" Power unit factors relative to watts,
key: unit str, value: factor
"""


POP = "pop"
""" Population identifier
"""


WORK_DIR = "work_dir"
"""Identifier for the working directory
"""

INPUT_DIR = "in_dir"
"""Identifier for the input base directory
"""

OUTPUT_DIR = "out_dir"
"""Identifier for the default output directory
"""


NC_ENGINE = "nc_engine"
"""Identifier for the NetCDF engine
"""


UTM_ZONE = "utm_zone"
"""Identifier for the UTM zone (number, letter) tuple
"""
