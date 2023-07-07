from foxes.utils import DataBook

from . import farms
from . import states
from . import power_ct_curves

FARM = "farm"
STATES = "states"
PCTCURVE = "power_ct_curve"


class StaticData(DataBook):
    """
    A DataBook filled with static data from
    this directory.

    :group: foxes

    """

    def __init__(self):
        super().__init__()

        self.add_data_package(FARM, farms, ".csv")
        self.add_data_package(STATES, states, [".csv", ".csv.gz", ".nc"])
        self.add_data_package(PCTCURVE, power_ct_curves, ".csv")
