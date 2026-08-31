import numpy as np

import foxes

if __name__ == "__main__":
    mbook = foxes.ModelBook()
    b14 = mbook.wake_models["Bastankhah2014"]

    wake_field = b14.export_wake_field(
        var_scan={"WS": np.arange(3.0, 15.5, 0.5), "CT": np.arange(0.0, 1.05, 0.05)},
        default_vars={
            "TI": 0.05,
            "RHO": 1.225,
        },
        D=100.0,
        H=100.0,
    )
