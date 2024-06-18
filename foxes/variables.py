X = "X"
""" The x coordinate in m
:group: foxes.variables
"""

Y = "Y"
""" The y coordinate in 
:group: foxes.variables
"""

H = "H"
""" The height over ground in m
:group: foxes.variables
"""

D = "D"
""" The rotor diameter in m
:group: foxes.variables
"""

TXYH = "txyh"
""" The turbine rotor centre coordinate 
vector (x, y, height)
:group: foxes.variables
"""

WEIGHT = "weight"
""" The statistical weight of a state
:group: foxes.variables
"""

ORDER = "order"
""" The turbine order
:group: foxes.variables
"""

ORDER_INV = "order_inv"
""" The inverse of the turbine order
:group: foxes.variables
"""

ORDER_SSEL = "order_ssel"
""" The states selection for applying the order
:group: foxes.variables
"""

WS = "WS"
""" The wind speed in m/s
:group: foxes.variables
"""

WD = "WD"
""" The wind direction in degrees
:group: foxes.variables
"""

TI = "TI"
""" The turbulence intensity
:group: foxes.variables
"""

RHO = "RHO"
""" The air density in kg/m3
:group: foxes.variables
"""

YAW = "YAW"
""" The absolute yaw angle of a turbine in degrees
:group: foxes.variables
"""

YAWM = "YAWM"
""" The relative yaw angle of a turbine in degrees
:group: foxes.variables
"""

P = "P"
""" The power, unit depends on user choice
:group: foxes.variables
"""

MAX_P = "MAXP"
""" The maximal power, for derating/boost
:group: foxes.variables
"""

CT = "CT"
""" The thrust coefficient
:group: foxes.variables
"""

T = "T"
""" The temperature in Kelvin
:group: foxes.variables
"""

YLD = "YLD"
""" Yield in GWh/a
:group: foxes.variables
"""

EFF = "EFF"
""" Efficiency, equals P/AMB_P
:group: foxes.variables
"""

CAP = "CAP"
""" Capacity, equals P/P_nominal
:group: foxes.variables
"""


REWS = "REWS"
""" Rotor effective wind speed in m/s
:group: foxes.variables
"""

REWS2 = "REWS2"
""" Rotor effective wind speed in m/s,
calculated from second moment
:group: foxes.variables
"""

REWS3 = "REWS3"
""" Rotor effective wind speed in m/s,
calculated from third moment
:group: foxes.variables
"""


AMB_WS = "AMB_WS"
""" The ambient wind speed in m/s
:group: foxes.variables
"""

AMB_WD = "AMB_WD"
""" The ambient wind direction in degrees
:group: foxes.variables
"""

AMB_TI = "AMB_TI"
""" The ambient turbulence intensity
:group: foxes.variables
"""

AMB_RHO = "AMB_RHO"
""" The ambient air density in kg/m3
:group: foxes.variables
"""

AMB_YAW = "AMB_YAW"
""" The ambient absolute yaw angle of 
a turbine in degrees
:group: foxes.variables
"""

AMB_YAWM = "AMB_YAWM"
""" The ambient relative yaw angle of 
a turbine in degrees
:group: foxes.variables
"""

AMB_P = "AMB_P"
""" The ambient power, unit depends on user choice
:group: foxes.variables
"""

AMB_CT = "AMB_CT"
""" The ambient thrust coefficient
:group: foxes.variables
"""

AMB_T = "AMB_T"
""" The ambient temperature in Kelvin
:group: foxes.variables
"""

AMB_YLD = "AMB_YLD"
""" Ambient yield in GWh/a
:group: foxes.variables
"""

AMB_CAP = "AMB_CAP"
""" Ambient capacity, equals AMB_P/P_nominal
:group: foxes.variables
"""


AMB_REWS = "AMB_REWS"
""" Ambient rotor effective wind speed in m/s
:group: foxes.variables
"""

AMB_REWS2 = "AMB_REWS2"
""" Ambient rotor effective wind speed in m/s,
calculated from second moment
:group: foxes.variables
"""

AMB_REWS3 = "AMB_REWS3"
""" Ambient rotor effective wind speed in m/s,
calculated from third moment
:group: foxes.variables
"""


var2amb = {
    v: f"AMB_{v}"
    for v in [
        WS,
        WD,
        TI,
        RHO,
        YAW,
        YAWM,
        P,
        CT,
        T,
        REWS,
        REWS2,
        REWS3,
        YLD,
        CAP,
    ]
}
""" Mapping from variable to the corresponding
ambient variable
:group: foxes.variables
"""

amb2var = {a: v for v, a in var2amb.items()}
""" Mapping from ambient variable to the corresponding
waked variable
:group: foxes.variables
"""


K = "k"
""" Wake growth parameter
:group: foxes.variables
"""

KB = "kb"
""" KTI value for zero TI, K = KB + KTI*TI
:group: foxes.variables
"""

KTI = "kTI"
""" Factor between K and TI, K = KB + KTI*TI
:group: foxes.variables
"""


Z0 = "z0"
""" The roughness length in m
:group: foxes.variables
"""

MOL = "MOL"
""" The Monin-Ubukhof length in m
:group: foxes.variables
"""

SHEAR = "shear"
""" The shear exponent
:group: foxes.variables
"""


PA_ALPHA = "PA_alpha"
""" The alpha parameter of the PorteAgel wake model
:group: foxes.variables
"""

PA_BETA = "PA_beta"
""" The beta parameter of the PorteAgel wake model
:group: foxes.variables
"""
