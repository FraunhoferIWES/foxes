from foxes.utils import DataBook

from . import farms
from . import states
from . import power_ct_curves
from . import windio

FARM = "farm"
""" Static wind farm data identifier
:group: data
"""

STATES = "states"
""" Static states data identifier
:group: data
"""

PCTCURVE = "power_ct_curve"
""" Static power-ct curve data identifier
:group: data
"""

WINDIO = "windio"
""" Static windio data identifier
:group: data
"""


class StaticData(DataBook):
    """
    A DataBook filled with static data from
    this directory.

    :group: data

    """

    def __init__(self):
        super().__init__()

        self.add_data_package(FARM, farms, ".csv")
        self.add_data_package(STATES, states, [".csv", ".csv.gz", ".nc", ".tab"])
        self.add_data_package(PCTCURVE, power_ct_curves, ".csv")
        self.add_data_package(WINDIO, windio, ".yaml")
