"""
List of foxes variables.

Attributes
----------
X : str
    The x coordinate in m
Y : str
    The y coordinate in m
H : str
    The height over ground in m
D : str
    The rotor diameter in m
XYH : str
    The vector (x, y, height)
TXYH : str
    The turbine rotor centre coordinate 
    vector (x, y, height)
WEIGHT : str
    The statistical weight of a state
ORDER : str
    The turbine order
WS : str
    The wind speed in m/s
WD : str
    The wind direction in degrees
TI : str
    The turbulence intensity
RHO : str
    The air density in kg/m3
YAW : str
    The absolute yaw angle of a turbine in degrees
YAWM : str
    The relative yaw angle of a turbine in degrees
P : str
    The power, unit depends on user choice
MAX_P : str
    The maximal power, for derating/boost
CT : str
    The thrust coefficient
T : str
    The temperature in Kelvin
YLD : str
    Yield in GWh/a
EFF : str
    Efficiency, equals P/AMB_P
CAP : str
    Capacity, equals P/P_nominal
REWS : str
    Rotor effective wind speed in m/s
REWS2 : str
    Rotor effective wind speed in m/s,
    calculated from second moment
REWS3 : str
    Rotor effective wind speed in m/s,
    calculated from third moment
AMB_WS : str
    The ambient wind speed in m/s
AMB_WD : str
    The ambient wind direction in degrees
AMB_TI : str
    The ambient turbulence intensity
AMB_RHO : str
    The ambient air density in kg/m3
AMB_YAW : str
    The ambient absolute yaw angle of a turbine in degrees
AMB_YAWM : str
    The ambient relative yaw angle of a turbine in degrees
AMB_P : str
    The ambient power, unit depends on user choice
AMB_CT : str
    The ambient thrust coefficient
AMB_T : str
    The ambient temperature in Kelvin
AMB_YLD : str
    Ambient yield in GWh/a
AMB_CAP : str
    Ambient capacity, equals AMB_P/P_nominal
AMB_REWS : str
    Ambient rotor effective wind speed in m/s
AMB_REWS2 : str
    Ambient rotor effective wind speed in m/s,
    calculated from second moment
AMB_REWS3 : str
    Ambient rotor effective wind speed in m/s,
    calculated from third moment
var2amb : dict
    Mapping from variable to the corresponding
    ambient variable
var2amb : dict
    Mapping from ambient variable to the corresponding
    waked variable
K : str
    Wake growth parameter
KB : str
    KTI value for zero TI, K = KB + KTI*TI
KTI : str
    Factor between K and TI, K = KB + KTI*TI
Z0 : str
    The roughness length in m
MOL : str
    The Monin-Ubukhof length in m
SHEAR : str
    The shear exponent

"""

X = "X"
Y = "Y"
H = "H"
D = "D"
XYH = "xyh"
TXYH = "txyh"
WEIGHT = "weight"
ORDER = "order"

WS = "WS"
WD = "WD"
TI = "TI"
RHO = "RHO"
YAW = "YAW"
YAWM = "YAWM"
P = "P"
MAX_P = "MAXP"
CT = "CT"
T = "T"
YLD = "YLD"
EFF = "EFF"
CAP = "CAP"

REWS = "REWS"
REWS2 = "REWS2"
REWS3 = "REWS3"

AMB_WS = "AMB_WS"
AMB_WD = "AMB_WD"
AMB_TI = "AMB_TI"
AMB_RHO = "AMB_RHO"
AMB_YAW = "AMB_YAW"
AMB_YAWM = "AMB_YAWM"
AMB_P = "AMB_P"
AMB_CT = "AMB_CT"
AMB_T = "AMB_T"
AMB_YLD = "AMB_YLD"
AMB_CAP = "AMB_CAP"

AMB_REWS = "AMB_REWS"
AMB_REWS2 = "AMB_REWS2"
AMB_REWS3 = "AMB_REWS3"

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
amb2var = {a: v for v, a in var2amb.items()}

K = "k"
KB = "kb"
KTI = "kTI"

Z0 = "z0"
MOL = "MOL"
SHEAR = "shear"
