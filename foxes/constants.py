import numpy as np

DTYPE = np.float64
ITYPE = np.int64

KAPPA = 0.41

W = "W"
kW = "kW"
MW = "MW"
GW = "GW"
TW = "TW"
P_UNITS = {
    W: 1.0,
    kW: 1.e3,
    MW: 1.e6,
    GW: 1.e9,
    TW: 1.e12
}
