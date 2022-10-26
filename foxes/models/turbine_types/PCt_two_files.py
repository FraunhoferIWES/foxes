import numpy as np
import pandas as pd

from foxes.core import TurbineType
from foxes.utils import PandasFileHelper
from foxes.data import PCTCURVE
import foxes.variables as FV


class PCtTwoFiles(TurbineType):
    """
    Calculate power and ct by interpolating
    from power-curve and ct-curve data files.

    Parameters
    ----------
    data_source_P : str or pandas.DataFrame
        The file path for the power-curve, static name, or data
    data_source_ct : str or pandas.DataFrame
        The file path for the ct-curve, static name, or data
    col_ws_P_file : str
        The wind speed column in the file of the power-curve
    col_ws_ct_file : str
        The wind speed column in the file of the ct-curve
    col_P : str
        The power column
    col_ct : str
        The ct column
    rho: float, optional
        The air densitiy for which the data is valid
        or None for no correction
    flag_yawm: bool
        Flag for yaw misalignment consideration
    p_ct: float
        The exponent for yaw dependency of ct
    p_P: float
        The exponent for yaw dependency of P
    var_ws_ct : str
        The wind speed variable for ct lookup
    var_ws_P : str
        The wind speed variable for power lookup
    pd_file_read_pars_P:  dict, optional
        Parameters for pandas power file reading
    pd_file_read_pars_ct:  dict, optional
        Parameters for pandas ct file reading
    **parameters : dict, optional
        Parameters for pandas file reading

    Attributes
    ----------
    source_P : str or pandas.DataFrame
        The file path for the power-curve, static name, or data
    source_ct : str or pandas.DataFrame
        The file path for the ct-curve, static name, or data
    col_ws : str
        The wind speed column
    col_P : str
        The power column
    col_ct : str
        The ct column
    rho: float
        The air densitiy for which the data is valid
        or None for no correction
    flag_yawm: bool
        Flag for yaw misalignment consideration
    WSCT : str
        The wind speed variable for ct lookup
    WSP : str
        The wind speed variable for power lookup
    rpars_P : dict, optional
        Parameters for pandas power file reading
    rpars_ct : dict, optional
        Parameters for pandas ct file reading

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
        flag_yawm=False,
        p_ct=1.0,
        p_P=1.88,
        var_ws_ct=FV.REWS2,
        var_ws_P=FV.REWS3,
        pd_file_read_pars_P={},
        pd_file_read_pars_ct={},
        **parameters
    ):
        pars = parameters  # no parsing because two files are given
        super().__init__(**pars)

        self.source_P = data_source_P
        self.source_ct = data_source_ct
        self.col_ws_P_file = col_ws_P_file
        self.col_ws_ct_file = col_ws_ct_file
        self.col_P = col_P
        self.col_ct = col_ct
        self.rho = rho
        self.flag_yawm = flag_yawm
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
        algo : foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        output_vars : list of str
            The output variable names

        """
        return [FV.P, FV.CT]

    def initialize(self, algo, st_sel, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        st_sel : numpy.ndarray of bool
            The state-turbine selection,
            shape: (n_states, n_turbines)
        verbosity : int
            The verbosity level

        """
        # read power-curve
        if self._data_P is None:
            if isinstance(self.source_P, pd.DataFrame):
                self._data_P = self.source_P
            else:
                fpath = algo.dbook.get_file_path(
                    PCTCURVE, self.source_P, check_raw=True
                )
                self._data_P = PandasFileHelper.read_file(fpath, **self.rpars_P)

            self._data_P = self._data_P.set_index(self.col_ws_P_file).sort_index()
            self._data_ws_P = self._data_P.index.to_numpy()
            self._data_P = self._data_P[self.col_P].to_numpy()

        # read ct-curve
        if self._data_ct is None:
            if isinstance(self.source_ct, pd.DataFrame):
                self._data_ct = self.source_ct
            else:
                fpath = algo.dbook.get_file_path(
                    PCTCURVE, self.source_ct, check_raw=True
                )
                self._data_ct = PandasFileHelper.read_file(fpath, **self.rpars_ct)

            self._data_ct = self._data_ct.set_index(self.col_ws_ct_file).sort_index()
            self._data_ws_ct = self._data_ct.index.to_numpy()
            self._data_ct = self._data_ct[self.col_ct].to_numpy()

        super().initialize(algo, st_sel, verbosity=verbosity)

    def calculate(self, algo, mdata, fdata, st_sel):
        """ "
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        st_sel : numpy.ndarray of bool
            The state-turbine selection,
            shape: (n_states, n_turbines)

        Returns
        -------
        results : dict
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
        if self.flag_yawm:

            # calculate corrected wind speed wsc,
            # gives ws**3 * cos**p_P in partial load region
            # and smoothly deals with full load region:
            yawm = fdata[FV.YAWM][st_sel]
            cosm = np.cos(yawm / 180 * np.pi)
            rews2 *= (cosm**self.p_ct) ** 0.5
            rews3 *= (cosm**self.p_P) ** (1.0 / 3.0)
            del yawm, cosm

        out = {
            FV.P: fdata.get(FV.P, np.zeros_like(fdata[self.WSCT])),
            FV.CT: fdata.get(FV.CT, np.zeros_like(fdata[self.WSP])),
        }
        out[FV.P][st_sel] = np.interp(
            rews3, self._data_ws_P, self._data_P, left=0.0, right=0.0
        )
        out[FV.CT][st_sel] = np.interp(
            rews2, self._data_ws_ct, self._data_ct, left=0.0, right=0.0
        )

        return out

    def finalize(self, algo, results, st_sel, clear_mem=False, verbosity=0):
        """
        Finalizes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        results : xarray.Dataset
            The calculation results
        st_sel : numpy.ndarray of bool
            The state-turbine selection,
            shape: (n_states, n_turbines)
        clear_mem : bool
            Flag for deleting model data and
            resetting initialization flag
        verbosity : int
            The verbosity level

        """
        if clear_mem:
            del self._data_ws_P, self._data_ws_ct, self._data_P, self._data_ct

        super().finalize(algo, results, clear_mem, verbosity=verbosity)
