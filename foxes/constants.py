"""
List of foxes constants.

Attributes
----------
FARM : str
    Wind farm identifier
STATE : str
    States identifier
TURBINE : str
    Wind turbine identifier
TNAME : str
    Wind turbine name identifier
POINT : str
    Point identifier
POINTS : str
    Points identifier
RPOINT : str
    Rotor point identifier
RPOINTS : str
    Rotor points identifier
RWEIGHTS : str
    Rotor point weights identifier
AMB_RPOINT_RESULTS : str
    Identified for ambient rotor point results
VARS : str
    Variables identifier
VALID : str
    Validity identifier
TMODELS : str
    Turbine models identifier
TMODELS_SELS : str
    Selected turbine models identifier
DTYPE : alias
    Default data type for floats
ITYPE : alias
    Default data type for int
KAPPA : float
    The Van-Karman constant
W : str
    The unit watt
kW : str
    The unit kilo watt
MW : str
    The unit mega watt
GW : str
    The unit giga watt
TW : str
    The unit terra watt
P_UNITS : dict
    Power unit factors relative to watts,
    key: unit str, value: factor
POP : str
    Population identifier
    
"""

import numpy as np

FARM = "farm"
STATE = "state"
TURBINE = "turbine"
TNAME = "tname"
POINT = "point"
POINTS = "points"
RPOINT = "rotor_point"
RPOINTS = "rotor_points"
RWEIGHTS = "rotor_weights"
AMB_RPOINT_RESULTS = "amb_rpoint_res"

VARS = "vars"
VALID = "valid"
TMODELS = "tmodels"
TMODEL_SELS = "tmodel_sels"

DTYPE = np.float64
ITYPE = np.int64

KAPPA = 0.41

W = "W"
kW = "kW"
MW = "MW"
GW = "GW"
TW = "TW"
P_UNITS = {W: 1.0, kW: 1.0e3, MW: 1.0e6, GW: 1.0e9, TW: 1.0e12}

POP = "pop"
