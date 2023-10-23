import numpy as np
import pandas as pd
from pathlib import Path

from foxes.core import TurbineType
from foxes.utils import PandasFileHelper
from foxes.data import PCTCURVE, parse_Pct_file_name
import foxes.variables as FV
import foxes.constants as FC


class PCtFile(TurbineType):
    """
    Calculate power and ct by interpolating
    from power-ct-curve data file (or pandas DataFrame).

    Attributes
    ----------
    source: str or pandas.DataFrame
        The file path, static name, or data
    col_ws: str
        The wind speed column
    col_P: str
        The power column
    col_ct: str
        The ct column
    rho: float
        The air densitiy for which the data is valid
        or None for no correction
    WSCT: str
        The wind speed variable for ct lookup
    WSP: str
        The wind speed variable for power lookup
    rpars: dict, optional
        Parameters for pandas file reading

    :group: models.turbine_types

    """

    def __init__(
        self,
        data_source,
        col_ws="ws",
        col_P="P",
        col_ct="ct",
        rho=None,
        p_ct=1.0,
        p_P=1.88,
        var_ws_ct=FV.REWS2,
        var_ws_P=FV.REWS3,
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
        col_P: str
            The power column
        col_ct: str
            The ct column
        rho: float, optional
            The air densitiy for which the data is valid
            or None for no correction
        p_ct: float
            The exponent for yaw dependency of ct
        p_P: float
            The exponent for yaw dependency of P
        var_ws_ct: str
            The wind speed variable for ct lookup
        var_ws_P: str
            The wind speed variable for power lookup
        pd_file_read_pars: dict
            Parameters for pandas file reading
        parameters: dict, optional
            Additional parameters for TurbineType class

        """
        if not isinstance(data_source, pd.DataFrame):
            pars = parse_Pct_file_name(data_source)
            pars.update(parameters)
        else:
            pars = parameters

        super().__init__(**pars)

        self.source = data_source
        self.col_ws = col_ws
        self.col_P = col_P
        self.col_ct = col_ct
        self.rho = rho
        self.p_ct = p_ct
        self.p_P = p_P
        self.WSCT = var_ws_ct
        self.WSP = var_ws_P
        self.rpars = pd_file_read_pars

    def output_farm_vars(self, algo):
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        output_vars: list of str
            The output variable names

        """
        return [FV.P, FV.CT]

    def load_data(self, algo, verbosity=0):
        """
        Load and/or create all model data that is subject to chunking.

        Such data should not be stored under self, for memory reasons. The
        data returned here will automatically be chunked and then provided
        as part of the mdata object during calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        idata: dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """
        if isinstance(self.source, pd.DataFrame):
            data = self.source
        else:
            fpath = algo.dbook.get_file_path(PCTCURVE, self.source, check_raw=True)
            if verbosity > 0:
                if not Path(self.source).is_file():
                    print(
                        f"Turbine type '{self.name}': Reading static data from context '{PCTCURVE}'"
                    )
                    print(f"Path: {fpath}")
                else:
                    print(f"Turbine type '{self.name}': Reading file", self.source)
            data = PandasFileHelper.read_file(fpath, **self.rpars)

        data = data.set_index(self.col_ws).sort_index()
        self.data_ws = data.index.to_numpy()
        self.data_P = data[self.col_P].to_numpy()
        self.data_ct = data[self.col_ct].to_numpy()

        if self.P_nominal is None:
            self.P_nominal = np.max(self.data_P) / FC.P_UNITS[self.P_unit]
            if verbosity > 0:
                print(
                    f"Turbine type '{self.name}': Setting P_nominal = {self.P_nominal:.2f} {self.P_unit}"
                )

        return super().load_data(algo, verbosity)

    def calculate(self, algo, mdata, fdata, st_sel):
        """ "
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        st_sel: numpy.ndarray of bool
            The state-turbine selection,
            shape: (n_states, n_turbines)

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        rews2 = fdata[self.WSCT][st_sel]
        rews3 = fdata[self.WSP][st_sel]

        # apply air density correction:
        if self.rho is not None:
            # correct wind speed by air density, such
            # that in the partial load region the
            # correct value is reconstructed:
            rho = fdata[FV.RHO][st_sel]
            rews2 *= (self.rho / rho) ** 0.5
            rews3 *= (self.rho / rho) ** (1.0 / 3.0)
            del rho

        # in yawed case, calc yaw corrected wind speed:
        if FV.YAWM in fdata and (self.p_P is not None or self.p_ct is not None):
            # calculate corrected wind speed wsc,
            # gives ws**3 * cos**p_P in partial load region
            # and smoothly deals with full load region:
            yawm = fdata[FV.YAWM][st_sel]
            if np.any(np.isnan(yawm)):
                raise ValueError(
                    f"{self.name}: Found NaN values for variable '{FV.YAWM}'. Maybe change order in turbine_models?"
                )
            cosm = np.cos(yawm / 180 * np.pi)
            if self.p_ct is not None:
                rews2 *= (cosm**self.p_ct) ** 0.5
            if self.p_P is not None:
                rews3 *= (cosm**self.p_P) ** (1.0 / 3.0)
            del yawm, cosm

        out = {
            FV.P: fdata[FV.P],
            FV.CT: fdata[FV.CT],
        }

        out[FV.P][st_sel] = np.interp(
            rews3, self.data_ws, self.data_P, left=0.0, right=0.0
        )
        out[FV.CT][st_sel] = np.interp(
            rews2, self.data_ws, self.data_ct, left=0.0, right=0.0
        )

        return out

    def finalize(self, algo, verbosity=0):
        """
        Finalizes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level

        """
        super().finalize(algo, verbosity)
        del self.data_ws, self.data_P, self.data_ct
