OPERATING = "operating"
""" Flag for turbine operation
"""

X = "X"
""" The x coordinate in m
"""

Y = "Y"
""" The y coordinate in m
"""

H = "H"
""" The height over ground in m
"""

D = "D"
""" The rotor diameter in m
"""

TXYH = "txyh"
""" The turbine rotor centre coordinate
vector (x, y, height)
"""

WEIGHT = "weight"
""" The statistical weight of a state
"""

ORDER = "order"
""" The turbine order
"""

ORDER_INV = "order_inv"
""" The inverse of the turbine order
"""

ORDER_SSEL = "order_ssel"
""" The states selection for applying the order
"""

WS = "WS"
""" The wind speed in m/s
"""

WD = "WD"
""" The wind direction in degrees
"""

UV = "UV"
""" The 2D wind vector in m/s
"""

U = "U"
""" The first horizontal wind vector component in m/s
"""

V = "V"
""" The second horizontal wind vector component in m/s
"""

TI = "TI"
""" The turbulence intensity
"""

TKE = "TKE"
""" The turbulent kinetic energy
"""

RHO = "RHO"
""" The air density in kg/m3
"""

YAW = "YAW"
""" The absolute yaw angle of a turbine in degrees
"""

YAWM = "YAWM"
""" The relative yaw angle of a turbine in degrees
"""

P = "P"
""" The power, unit depends on user choice
"""

CAP = "CAP"
""" The capacity (equals P_nominal for wind turbines)
"""

MAX_P = "MAXP"
""" The maximal power, for derating/boost
"""

CT = "CT"
""" The thrust coefficient
"""

T = "T"
""" The temperature in Kelvin
"""

p = "p"
""" The pressure in Pa
"""

YLD = "YLD"
""" Yield in GWh/a
"""

EFF = "EFF"
""" Efficiency, equals P/AMB_P
"""

CAPF = "CAPF"
""" Capacity factor, equals P/CAP
"""

FLF = "FLF"
""" The full load fraction
"""


REWS = "REWS"
""" Rotor effective wind speed in m/s
"""

REWS2 = "REWS2"
""" Rotor effective wind speed in m/s,
calculated from second moment
"""

REWS3 = "REWS3"
""" Rotor effective wind speed in m/s,
calculated from third moment
"""


WEIBULL_A = "Weibull_A"
""" The Weibull scale parameter,
"""

WEIBULL_k = "Weibull_k"
""" The Weibull shape parameter,
"""


AMB_WS = "AMB_WS"
""" The ambient wind speed in m/s
"""

AMB_WD = "AMB_WD"
""" The ambient wind direction in degrees
"""

AMB_UV = "AMB_UV"
""" The ambient 2D wind vector in m/s
"""

AMB_U = "AMB_U"
""" The first horizontal ambient wind vector component in m/s
"""

AMB_V = "AMB_V"
""" The second horizontal ambient wind vector component in m/s
"""

AMB_TI = "AMB_TI"
""" The ambient turbulence intensity
"""

AMB_TKE = "AMB_TKE"
""" The ambient turbulent kinetic energy
"""

AMB_RHO = "AMB_RHO"
""" The ambient air density in kg/m3
"""

AMB_YAW = "AMB_YAW"
""" The ambient absolute yaw angle of
a turbine in degrees
"""

AMB_YAWM = "AMB_YAWM"
""" The ambient relative yaw angle of
a turbine in degrees
"""

AMB_P = "AMB_P"
""" The ambient power, unit depends on user choice
"""

AMB_CT = "AMB_CT"
""" The ambient thrust coefficient
"""

AMB_T = "AMB_T"
""" The ambient temperature in Kelvin
"""

AMB_p = "AMB_p"
""" The ambient pressure in Pa
"""

AMB_YLD = "AMB_YLD"
""" Ambient yield in GWh/a
"""

AMB_CAPF = "AMB_CAPF"
""" Ambient capacity, equals AMB_P/CAP
"""

AMB_FLF = "AMB_FLF"
""" The ambient full load fraction
"""


AMB_REWS = "AMB_REWS"
""" Ambient rotor effective wind speed in m/s
"""

AMB_REWS2 = "AMB_REWS2"
""" Ambient rotor effective wind speed in m/s,
calculated from second moment
"""

AMB_REWS3 = "AMB_REWS3"
""" Ambient rotor effective wind speed in m/s,
calculated from third moment
"""

AMB_WEIBULL_A = "AMB_Weibull_A"
""" Ambient Weibull scale parameter,
"""

AMB_WEIBULL_k = "AMB_Weibull_k"
""" Ambient Weibull shape parameter,
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
        CAPF,
        FLF,
        UV,
        U,
        V,
    ]
}
""" Mapping from variable to the corresponding
ambient variable
"""

amb2var = {a: v for v, a in var2amb.items()}
""" Mapping from ambient variable to the corresponding
waked variable
"""


extensive_farm = set(
    [
        AMB_P,
        P,
        AMB_YLD,
        YLD,
        AMB_CAPF,
        CAPF,
    ]
)
""" Set of extensive variables, i.e. variables that should be summed when aggregating over turbines
"""

intensive_farm = (
    set(var2amb.keys()).union(set(var2amb.values())).difference(extensive_farm)
)
""" Set of intensive variables, i.e. variables that should be averaged when aggregating over turbines
"""


extensive_state = set(
    [
        AMB_P,
        P,
        AMB_YLD,
        YLD,
        AMB_CAPF,
        CAPF,
        WEIGHT,
    ]
)
""" Set of extensive variables, i.e. variables that should be summed when aggregating over states
"""
intensive_state = (
    set(var2amb.keys()).union(set(var2amb.values())).difference(extensive_state)
)
""" Set of intensive variables, i.e. variables that should be averaged when aggregating over states
"""


MEAN_WS = "MEAN_WS"
""" The mean wind speed in m/s
"""

MAIN_WD = "MAIN_WD"
""" The main wind direction in degrees
"""


K = "k"
""" Wake growth parameter
"""

KB = "kb"
""" KTI value for zero TI, K = KB + KTI*TI
"""

KTI = "kTI"
""" Factor between K and TI, K = KB + KTI*TI
"""


Z0 = "z0"
""" The roughness length in m
"""

MOL = "MOL"
""" The Monin–Obukhov length in m
"""

USTAR = "USTAR"
""" The friction velocity in m/s
"""

SHEAR = "shear"
""" The shear exponent
"""


PA_ALPHA = "PA_alpha"
""" The alpha parameter of the PorteAgel wake model
"""

PA_BETA = "PA_beta"
""" The beta parameter of the PorteAgel wake model
"""


LAT = "LAT"
""" The latitude in degrees
"""

LON = "LON"
""" The longitude in degrees
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
    CAPF: 5,
    EFF: 5,
    WEIBULL_A: 3,
    WEIBULL_k: 3,
    YAW: 3,
    YAWM: 3,
    LAT: 6,
    LON: 6,
    "lat": 6,
    "lon": 6,
    "latitude": 6,
    "longitude": 6,
}
ROUND_DIGITS.update(
    {var2amb[v]: ROUND_DIGITS[v] for v in var2amb.keys() if v in ROUND_DIGITS}
)


def get_default_digits(variable: str) -> int:
    """
    Gets the default number of output digits

    Parameters
    ----------
        variable
        The variable name

    Returns
    -------
        digits
        The default number of output digits

    """
    return ROUND_DIGITS.get(variable, DEFAULT_DIGITS)
