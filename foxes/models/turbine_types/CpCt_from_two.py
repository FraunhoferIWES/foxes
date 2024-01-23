import numpy as np
import pandas as pd

from .PCt_from_two import PCtFromTwo
from foxes.data import parse_Pct_two_files
from foxes.utils import PandasFileHelper
import foxes.constants as FC


class CpCtFromTwo(PCtFromTwo):
    """
    Calculate power and ct by interpolating
    cp and ct from two files (or two pandas
    DataFrames).

    :group: models.turbine_types

    """

    def __init__(
        self,
        data_source_cp,
        data_source_ct,
        col_ws_cp_file="ws",
        col_cp="cp",
        rho=1.225,
        pd_file_read_pars_cp={},
        pd_file_read_pars_ct={},
        **parameters,
    ):
        """
        Constructor.

        Parameters
        ----------
        data_source_cp: str or pandas.DataFrame
            The file path, static name, or data
        data_source_ct: str or pandas.DataFrame
            The file path, static name, or data
        col_ws_cp_file: str
            The wind speed column in the file of the cp curve
        col_cp: str
            The cp column
        rho: float
            The air density for the curves
        pd_file_read_pars_cp:  dict
            Parameters for pandas cp file reading
        pd_file_read_pars_ct:  dict
            Parameters for pandas ct file reading
        parameters: dict, optional
            Additional parameters for PCtFromTwo class

        """
        if not isinstance(data_source_cp, pd.DataFrame) or not isinstance(
            data_source_ct, pd.DataFrame
        ):
            pars = parse_Pct_two_files(data_source_cp, data_source_ct)
            data_cp = PandasFileHelper.read_file(data_source_cp, **pd_file_read_pars_cp)
            data_ct = PandasFileHelper.read_file(data_source_ct, **pd_file_read_pars_ct)
        else:
            data_cp = data_source_cp
            data_ct = data_source_ct
            pars = parameters

        D = pars["D"]
        A = np.pi * (D / 2) ** 2
        ws = data_cp[col_ws_cp_file].to_numpy()
        cp = data_cp[col_cp].to_numpy()
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

        super().__init__(
            data_P,
            data_ct,
            col_ws_P_file="ws",
            col_P="P",
            rho=rho,
            P_unit=P_unit,
            **pars,
        )
