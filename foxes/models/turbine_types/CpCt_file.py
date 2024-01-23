import numpy as np
import pandas as pd

from .PCt_file import PCtFile
from foxes.data import parse_Pct_file_name
from foxes.utils import PandasFileHelper
import foxes.constants as FC


class CpCtFile(PCtFile):
    """
    Calculate power and ct by interpolating
    from cp-ct-curve data file (or pandas DataFrame).

    :group: models.turbine_types

    """

    def __init__(
        self,
        data_source,
        col_ws="ws",
        col_cp="cp",
        rho=1.225,
        pd_file_read_pars={},
        **parameters,
    ):
        """
        Constructor.

        Parameters
        ----------
        data_source: str or pandas.DataFrame
            The file path, static name, or data
        col_ws: str
            The wind speed column
        col_cp: str
            The cp column
        rho: float
            The air density for the curves
        pd_file_read_pars: dict
            Parameters for pandas file reading
        paramerers: dict, optional
            Additional parameters for PCtFile class

        """
        if not isinstance(data_source, pd.DataFrame):
            pars = parse_Pct_file_name(data_source)
            pars.update(parameters)
            data = PandasFileHelper.read_file(data_source, **pd_file_read_pars)
        else:
            data = data_source
            pars = parameters

        D = pars["D"]
        A = np.pi * (D / 2) ** 2
        ws = data[col_ws].to_numpy()
        cp = data[col_cp].to_numpy()
        P_unit = pars.pop("P_unit", FC.kW)

        ws_delta = 0.0001
        ws_min = np.min(ws)
        ws_max = np.max(ws)
        N = int((ws_max - ws_min) / ws_delta)

        data_P = pd.DataFrame(index=range(N), dtype=FC.DTYPE)
        data_P["ws"] = np.linspace(ws_min, ws_max, N, endpoint=True)
        data_P["cp"] = np.interp(data_P["ws"], ws, cp, left=0, right=0)
        data_P["P"] = (
            0.5 * rho * A * data_P["cp"] * data_P["ws"] ** 3 / FC.P_UNITS[P_unit]
        )

        super().__init__(data_P, col_ws="ws", col_P="P", rho=rho, P_unit=P_unit, **pars)
