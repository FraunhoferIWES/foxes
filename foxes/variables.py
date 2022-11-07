FARM = "farm"
STATE = "state"
TURBINE = "turbine"
POINT = "point"
POINTS = "points"
RPOINT = "rotor_point"
RPOINTS = "rotor_points"
RWEIGHTS = "rotor_weights"
ORDER = "order"

VARS = "vars"
TMODELS = "tmodels"
TMODEL_SELS = "tmodel_sels"

X = "X"
Y = "Y"
H = "H"
D = "D"
XYH = "xyh"
TXYH = "txyh"
WEIGHT = "weight"

WS = "WS"
WD = "WD"
TI = "TI"
RHO = "RHO"
YAW = "YAW"
YAWM = "YAWM"
P = "P"
CT = "CT"
T = "T"

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

AMB_REWS = "AMB_REWS"
AMB_REWS2 = "AMB_REWS2"
AMB_REWS3 = "AMB_REWS3"

var2amb = {
    v: f"AMB_{v}" for v in [WS, WD, TI, RHO, YAW, YAWM, P, CT, T, REWS, REWS2, REWS3]
}
amb2var = {a: v for v, a in var2amb.items()}

AMB_RPOINT_RESULTS = "amb_rpoint_res"

K = "k"
KY = "ky"
KZ = "kz"
KTI = "kTI"
KTIY = "kTIy"
KTIZ = "kTIz"

Z0 = "z0"
MOL = "MOL"

POP = "pop"

MAX_P = "MAXP"
