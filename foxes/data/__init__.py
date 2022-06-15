from .parse import parse_Pct_file_name

from . import farms
from . import states
from . import power_ct_curves

FARM     = "farm"
STATES   = "states"
PCTCURVE = "power_ct_curve"

from foxes.tools import DataBook

data_book = DataBook()
data_book.add_data_package(FARM, farms, ".csv")
data_book.add_data_package(STATES, states, [".csv.gz", ".nc"])
data_book.add_data_package(PCTCURVE, power_ct_curves, ".csv")
