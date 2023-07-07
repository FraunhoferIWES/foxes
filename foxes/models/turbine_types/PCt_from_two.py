import numpy as np
import pandas as pd

from foxes.core import TurbineType
from foxes.utils import PandasFileHelper
from foxes.data import PCTCURVE, parse_Pct_two_files
import foxes.variables as FV
import foxes.constants as FC


class PCtFromTwo(TurbineType):
    """
    Calculate power and ct by interpolating
    from power curve and ct curve data files.

    Attributes
    ----------
    source_P: str or pandas.DataFrame
        The file path for the power curve, static name, or data
    source_ct: str or pandas.DataFrame
        The file path for the ct curve, static name, or data
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
    rpars_P: dict, optional
        Parameters for pandas power file reading
    rpars_ct: dict, optional
        Parameters for pandas ct file reading

    :group: models.turbine_types

    """

    def __init__(
        self,
        data_source_P,
        data_source_ct,
        col_ws_P_file="ws",
        col_ws_ct_file="ws",
        col_P="P",
        col_ct="ct",
        rho=None,
        p_ct=1.0,
        p_P=1.88,
        var_ws_ct=FV.REWS2,
        var_ws_P=FV.REWS3,
        pd_file_read_pars_P={},
        pd_file_read_pars_ct={},
        **parameters,
    ):
        """
        Constructor.

        Parameters
        ----------
        data_source_P: str or pandas.DataFrame
            The file path for the power curve, static name, or data
        data_source_ct: str or pandas.DataFrame
            The file path for the ct curve, static name, or data
        col_ws_P_file: str
            The wind speed column in the file of the power curve
        col_ws_ct_file: str
            The wind speed column in the file of the ct curve
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
        pd_file_read_pars_P:  dict
            Parameters for pandas power file reading
        pd_file_read_pars_ct:  dict
            Parameters for pandas ct file reading
        parameters: dict, optional
            Additional parameters for TurbineType class

        """
        if not isinstance(data_source_P, pd.DataFrame) or not isinstance(
            data_source_ct, pd.DataFrame
        ):
            pars = parse_Pct_two_files(data_source_P, data_source_ct)
        else:
            pars = parameters
        super().__init__(**pars)

        self.source_P = data_source_P
        self.source_ct = data_source_ct
        self.col_ws_P_file = col_ws_P_file
        self.col_ws_ct_file = col_ws_ct_file
        self.col_P = col_P
        self.col_ct = col_ct
        self.rho = rho
        self.p_ct = p_ct
        self.p_P = p_P
        self.WSCT = var_ws_ct
        self.WSP = var_ws_P
        self.rpars_P = pd_file_read_pars_P
        self.rpars_ct = pd_file_read_pars_ct

        self._data_P = None
        self._data_ct = None
        self._data_ws_P = None
        self._data_ws_ct = None

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

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        This includes loading all required data from files. The model
        should return all array type data as part of the idata return
        dictionary (and not store it under self, for memory reasons). This
        data will then be chunked and provided as part of the mdata object
        during calculations.

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
        # read power curve:
        if isinstance(self.source_P, pd.DataFrame):
            self._data_P = self.source_P
        else:
            fpath = algo.dbook.get_file_path(PCTCURVE, self.source_P, check_raw=True)
            self._data_P = PandasFileHelper.read_file(fpath, **self.rpars_P)

        self._data_P = self._data_P.set_index(self.col_ws_P_file).sort_index()
        self._data_ws_P = self._data_P.index.to_numpy()
        self._data_P = self._data_P[self.col_P].to_numpy()

        # read ct curve:
        if isinstance(self.source_ct, pd.DataFrame):
            self._data_ct = self.source_ct
        else:
            fpath = algo.dbook.get_file_path(PCTCURVE, self.source_ct, check_raw=True)
            self._data_ct = PandasFileHelper.read_file(fpath, **self.rpars_ct)

        self._data_ct = self._data_ct.set_index(self.col_ws_ct_file).sort_index()
        self._data_ws_ct = self._data_ct.index.to_numpy()
        self._data_ct = self._data_ct[self.col_ct].to_numpy()

        if self.P_nominal is None:
            self.P_nominal = np.max(self._data_P) / FC.P_UNITS[self.P_unit]
            if verbosity > 0:
                print(
                    f"Turbine type '{self.name}': Setting P_nominal = {self.P_nominal:.2f} {self.P_unit}"
                )

        return super().initialize(algo, verbosity)

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
            rews3, self._data_ws_P, self._data_P, left=0.0, right=0.0
        )
        out[FV.CT][st_sel] = np.interp(
            rews2, self._data_ws_ct, self._data_ct, left=0.0, right=0.0
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
        del self._data_ws_P, self._data_ws_ct, self._data_P, self._data_ct
        super().finalize(algo, verbosity)
