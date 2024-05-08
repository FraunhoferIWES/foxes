import numpy as np
import pandas as pd
from scipy.interpolate import interpn

from foxes.core import TurbineType
from foxes.utils import PandasFileHelper
from foxes.data import PCTCURVE, parse_Pct_two_files
import foxes.variables as FV
import foxes.constants as FC


class WsTI2PCtFromTwo(TurbineType):
    """
    Calculate turbulent intensity dependent power
    and ct values, as given by two individual
    files.

    The structure of each file is:
    ws,0.05,0.06,0.07,...

    The first column represents wind speed in m/s
    and the subsequent columns are TI values
    (not neccessarily in order).

    Attributes
    ----------
    source_P: str or pandas.DataFrame
        The file path for the power curve, static name, or data
    source_ct: str or pandas.DataFrame
        The file path for the ct curve, static name, or data
    WSCT: str
        The wind speed variable for ct lookup
    WSP: str
        The wind speed variable for power lookup
    rpars_P: dict, optional
        Parameters for pandas power file reading
    rpars_ct: dict, optional
        Parameters for pandas ct file reading
    ipars_P: dict, optional
        Parameters for scipy.interpolate.interpn()
    ipars_ct: dict, optional
        Parameters for scipy.interpolate.interpn()
    rho: float
        The air densitiy for which the data is valid
        or None for no correction

    :group: models.turbine_types

    """

    def __init__(
        self,
        data_source_P,
        data_source_ct,
        rho=None,
        p_ct=1.0,
        p_P=1.88,
        var_ws_ct=FV.REWS2,
        var_ws_P=FV.REWS3,
        pd_file_read_pars_P={},
        pd_file_read_pars_ct={},
        interpn_pars_P=None,
        interpn_pars_ct=None,
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
        interpn_pars_P: dict, optional
            Parameters for scipy.interpolate.interpn()
        interpn_pars_ct: dict, optional
            Parameters for scipy.interpolate.interpn()
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
        self.rho = rho
        self.p_ct = p_ct
        self.p_P = p_P
        self.WSCT = var_ws_ct
        self.WSP = var_ws_P
        self.rpars_P = pd_file_read_pars_P
        self.rpars_ct = pd_file_read_pars_ct
        self.ipars_P = interpn_pars_P
        self.ipars_ct = interpn_pars_ct

        if self.ipars_P is None:
            self.ipars_P = dict(method="linear", bounds_error=True, fill_value=0.0)
        if self.ipars_ct is None:
            self.ipars_ct = dict(method="linear", bounds_error=True, fill_value=0.0)

        self._P = None
        self._ct = None

    def __repr__(self):
        a = f"D={self.D}, H={self.H}, P_nominal={self.P_nominal}, P_unit={self.P_unit}, rho={self.rho}"
        return f"{type(self).__name__}({a})"

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
        # read power curve:
        if isinstance(self.source_P, pd.DataFrame):
            data = self.source_P
        else:
            fpath = algo.dbook.get_file_path(PCTCURVE, self.source_P, check_raw=True)
            pars = {"index_col": 0}
            pars.update(self.rpars_P)
            data = PandasFileHelper.read_file(fpath, **pars)

        data.sort_index(inplace=True)
        data.columns = data.columns.astype(FC.DTYPE)
        self._ws_P = data.index.to_numpy(FC.DTYPE)
        self._ti_P = np.sort(data.columns.to_numpy())
        self._P = data[self._ti_P].to_numpy(FC.DTYPE)

        # read ct curve:
        if isinstance(self.source_ct, pd.DataFrame):
            data = self.source_ct
        else:
            fpath = algo.dbook.get_file_path(PCTCURVE, self.source_ct, check_raw=True)
            pars = {"index_col": 0}
            pars.update(self.rpars_ct)
            data = PandasFileHelper.read_file(fpath, **pars)

        data.sort_index(inplace=True)
        data.columns = data.columns.astype(FC.DTYPE)
        self._ws_ct = data.index.to_numpy(FC.DTYPE)
        self._ti_ct = np.sort(data.columns.to_numpy())
        self._ct = data[self._ti_ct].to_numpy(FC.DTYPE)

        return super().load_data(algo, verbosity)

    def _bounds_info(self, target, qts):
        """Helper function for printing bounds info"""

        print(f"\nBOUNDS INFO FOR TARGET {target}")
        WS = self.WSP if target == FV.P else self.WSCT
        ws = self._ws_P if target == FV.P else self._ws_ct
        ti = self._ti_P if target == FV.P else self._ti_ct
        print(f"  {WS}: min = {np.min(ws):.4f}, max = {np.max(ws):.4f}")
        print(f"  {FV.TI}: min = {np.min(ti):.4f}, max = {np.max(ti):.4f}")

        print(f"DATA INFO FOR TARGET {target}")
        ws = qts[:, 0]
        ti = qts[:, 1]
        print(f"  {WS}: min = {np.min(ws):.4f}, max = {np.max(ws):.4f}")
        print(f"  {FV.TI}: min = {np.min(ti):.4f}, max = {np.max(ti):.4f}")
        print()

    def calculate(self, algo, mdata, fdata, st_sel):
        """ "
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
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

        # calculate P:
        st_sel_P = (
            st_sel
            & (fdata[self.WSP] >= self._ws_P[0])
            & (fdata[self.WSP] <= self._ws_P[-1])
        )
        st_sel_P0 = st_sel & ~st_sel_P
        if np.any(st_sel_P0):
            fdata[FV.P][st_sel_P0] = 0
        if np.any(st_sel_P):
            # prepare interpolation:
            n_sel = np.sum(st_sel_P)
            qts = np.zeros((n_sel, 2), dtype=FC.DTYPE)  # ws, ti
            qts[:, 0] = fdata[self.WSP][st_sel_P]
            qts[:, 1] = fdata[FV.TI][st_sel_P]

            # apply air density correction:
            if self.rho is not None:
                # correct wind speed by air density, such
                # that in the partial load region the
                # correct value is reconstructed:
                rho = fdata[FV.RHO][st_sel]
                qts[:, 0] *= (self.rho / rho) ** (1.0 / 3.0)
                del rho

            # apply yaw corrections:
            if FV.YAWM in fdata and self.p_P is not None:
                # calculate corrected wind speed wsc,
                # gives ws**3 * cos**p_P in partial load region
                # and smoothly deals with full load region:
                yawm = fdata[FV.YAWM][st_sel_P]
                if np.any(np.isnan(yawm)):
                    raise ValueError(
                        f"{self.name}: Found NaN values for variable '{FV.YAWM}'. Maybe change order in turbine_models?"
                    )
                cosm = np.cos(yawm / 180 * np.pi)
                qts[:, 0] *= (cosm**self.p_P) ** (1.0 / 3.0)
                del yawm, cosm

            # run interpolation:
            try:
                fdata[FV.P][st_sel_P] = interpn(
                    (self._ws_P, self._ti_P), self._P, qts, **self.ipars_P
                )
            except ValueError as e:
                self._bounds_info(FV.P, qts)
                raise e
        del st_sel_P, st_sel_P0

        # calculate ct:
        st_sel_ct = (
            st_sel
            & (fdata[self.WSCT] >= self._ws_P[0])
            & (fdata[self.WSCT] <= self._ws_P[-1])
        )
        st_sel_ct0 = st_sel & ~st_sel_ct
        if np.any(st_sel_ct0):
            fdata[FV.CT][st_sel_ct0] = 0
        if np.any(st_sel_ct):
            # prepare interpolation:
            n_sel = np.sum(st_sel_ct)
            qts = np.zeros((n_sel, 2), dtype=FC.DTYPE)  # ws, ti
            qts[:, 0] = fdata[self.WSP][st_sel_ct]
            qts[:, 1] = fdata[FV.TI][st_sel_ct]

            # apply air density correction:
            if self.rho is not None:
                # correct wind speed by air density, such
                # that in the partial load region the
                # correct value is reconstructed:
                rho = fdata[FV.RHO][st_sel]
                qts[:, 0] *= (self.rho / rho) ** 0.5
                del rho

            # apply yaw corrections:
            if FV.YAWM in fdata and self.p_ct is not None:
                # calculate corrected wind speed wsc,
                # gives ws**3 * cos**p_P in partial load region
                # and smoothly deals with full load region:
                yawm = fdata[FV.YAWM][st_sel_ct]
                if np.any(np.isnan(yawm)):
                    raise ValueError(
                        f"{self.name}: Found NaN values for variable '{FV.YAWM}'. Maybe change order in turbine_models?"
                    )
                cosm = np.cos(yawm / 180 * np.pi)
                qts[:, 0] *= (cosm**self.p_ct) ** 0.5
                del yawm, cosm

            # run interpolation:
            try:
                fdata[FV.CT][st_sel_ct] = interpn(
                    (self._ws_ct, self._ti_ct), self._ct, qts, **self.ipars_ct
                )
            except ValueError as e:
                self._bounds_info(FV.CT, qts)
                raise e

        return {v: fdata[v] for v in self.output_farm_vars(algo)}

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
        del self._ws_P, self._ti_P, self._ws_ct, self._ti_ct
        self._P = None
        self._ct = None
        super().finalize(algo, verbosity)
