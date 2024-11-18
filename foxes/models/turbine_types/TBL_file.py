import numpy as np
import pandas as pd

from foxes.config import get_path

from .PCt_file import PCtFile


class TBLFile(PCtFile):
    """
    Reads turbine data from a TBL file.

    Examples
    --------
    A TBL file is a csv file with space as separator
    and two header lines. The followind lines denote
    wind speed, ct, P.

    - first row will be ignored
    - second row: H D ct-stand-still rated-power-in-MW
    - further rows: ws ct P-in-kW

    For example:

    45
    175.0 290.0 0.03 22.0
    3.0 0.820 102.9
    3.5 0.800 444.0
    4.0 0.810 874.7
    4.5 0.820 1418.7
    5.0 0.820 2100.9
    5.5 0.830 3021.2
    6.0 0.830 3904.7
    6.5 0.830 5061.7
    7.0 0.810 6379.0

    :group: models.turbine_types

    """

    def __init__(
        self,
        tbl_file,
        rho=1.225,
        **parameters,
    ):
        """
        Constructor.

        Parameters
        ----------
        tbl_file: str
            Path to the tbl file
        rho: float
            The air density for the curves
        paramerers: dict, optional
            Additional parameters for PCtFile class

        """
        fpath = get_path(tbl_file)
        assert fpath.suffix == ".tbl", f"Expecting *.tbl file, got '{tbl_file}'"

        meta = np.genfromtxt(fpath, skip_header=1, max_rows=1)
        sdata = pd.read_csv(
            fpath, sep=" ", skiprows=2, header=None, names=["ws", "ct", "P"]
        )

        super().__init__(
            sdata,
            col_ws="ws",
            col_P="P",
            col_ct="ct",
            H=meta[0],
            D=meta[1],
            P_nominal=meta[3] * 1e3,
            P_unit="kW",
            rho=rho,
            **parameters,
        )
