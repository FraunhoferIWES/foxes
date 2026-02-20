OPERATING = "operating"
""" Flag for turbine operation
:group: foxes.variables
"""

X = "X"
""" The x coordinate in m
:group: foxes.variables
"""

Y = "Y"
""" The y coordinate in m
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

UV = "UV"
""" The 2D wind vector in m/s
:group: foxes.variables
"""

U = "U"
""" The first horizontal wind vector component in m/s
:group: foxes.variables
"""

V = "V"
""" The second horizontal wind vector component in m/s
:group: foxes.variables
"""

TI = "TI"
""" The turbulence intensity
:group: foxes.variables
"""

TKE = "TKE"
""" The turbulent kinetic energy
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

p = "p"
""" The pressure in Pa
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


WEIBULL_A = "Weibull_A"
""" The Weibull scale parameter,
:group: foxes.variables
"""

WEIBULL_k = "Weibull_k"
""" The Weibull shape parameter,
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

AMB_UV = "AMB_UV"
""" The ambient 2D wind vector in m/s
:group: foxes.variables
"""

AMB_U = "AMB_U"
""" The first horizontal ambient wind vector component in m/s
:group: foxes.variables
"""

AMB_V = "AMB_V"
""" The second horizontal ambient wind vector component in m/s
:group: foxes.variables
"""

AMB_TI = "AMB_TI"
""" The ambient turbulence intensity
:group: foxes.variables
"""

AMB_TKE = "AMB_TKE"
""" The ambient turbulent kinetic energy
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

AMB_p = "AMB_p"
""" The ambient pressure in Pa
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

AMB_WEIBULL_A = "AMB_Weibull_A"
""" Ambient Weibull scale parameter,
:group: foxes.variables
"""

AMB_WEIBULL_k = "AMB_Weibull_k"
""" Ambient Weibull shape parameter,
:group: foxes.variables
"""


var2amb = {
    v: f"AMB_{v}"
    for v in [
        WS,
        WD,
        TI,
        TKE,
        RHO,
        YAW,
        YAWM,
        P,
        CT,
        T,
        p,
        REWS,
        REWS2,
        REWS3,
        WEIBULL_A,
        WEIBULL_k,
        YLD,
        CAP,
        UV,
        U,
        V,
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
""" The Monin–Obukhov length in m
:group: foxes.variables
"""

USTAR = "USTAR"
""" The friction velocity in m/s
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

DEFAULT_DIGITS = 4

ROUND_DIGITS = {
    WD: 3,
    TI: 6,
    RHO: 5,
    P: 3,
    CT: 6,
    T: 3,
    YLD: 3,
    CAP: 5,
    EFF: 5,
    WEIBULL_A: 3,
    WEIBULL_k: 3,
    YAW: 3,
    YAWM: 3,
    "lat": 6,
    "lon": 6,
    "latitude": 6,
    "longitude": 6,
    "LAT": 6,
    "LON": 6,
}
ROUND_DIGITS.update(
    {var2amb[v]: ROUND_DIGITS[v] for v in var2amb.keys() if v in ROUND_DIGITS}
)


def get_default_digits(variable):
    """
    Gets the default number of output digits

    Parameters
    ----------
    variable: str
        The variable name

    Returns
    -------
    digits: int
        The default number of output digits

    """
    return ROUND_DIGITS.get(variable, DEFAULT_DIGITS)
