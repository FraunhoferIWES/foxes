import numpy as np
import pandas as pd
from pathlib import Path

from foxes.core import TurbineType
from foxes.utils import PandasFileHelper
from foxes.data import PCTCURVE, parse_Pct_file_name
import foxes.variables as FV


class PCtFile(TurbineType):
    """
    Calculate power and ct by interpolating
    from power-ct-curve data file.

    Parameters
    ----------
    data_source : str or pandas.DataFrame
        The file path, static name, or data
    col_ws : str
        The wind speed column
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
    **parameters : dict, optional
        Parameters for pandas file reading

    Attributes
    ----------
    source : str or pandas.DataFrame
        The file path, static name, or data
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
    rpars : dict, optional
        Parameters for pandas file reading

    """

    def __init__(
        self,
        data_source,
        col_ws="ws",
        col_P="P",
        col_ct="ct",
        rho=None,
        flag_yawm=False,
        p_ct=1.0,
        p_P=1.88,
        var_ws_ct=FV.REWS2,
        var_ws_P=FV.REWS3,
        pd_file_read_pars={},
        **parameters,
    ):
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
        self.flag_yawm = flag_yawm
        self.p_ct = p_ct
        self.p_P = p_P
        self.WSCT = var_ws_ct
        self.WSP = var_ws_P
        self.rpars = pd_file_read_pars
        self._data = None

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
        if self._data is None:
            if isinstance(self.source, pd.DataFrame):
                self._data = self.source
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
                self._data = PandasFileHelper.read_file(fpath, **self.rpars)

            self._data = self._data.set_index(self.col_ws).sort_index()
            self.data_ws = self._data.index.to_numpy()
            self.data_P = self._data[self.col_P].to_numpy()
            self.data_ct = self._data[self.col_ct].to_numpy()

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
            rews3, self.data_ws, self.data_P, left=0.0, right=0.0
        )
        out[FV.CT][st_sel] = np.interp(
            rews2, self.data_ws, self.data_ct, left=0.0, right=0.0
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
            del self._data, self.data_ws, self.data_P, self.data_ct
        super().finalize(algo, results, clear_mem, verbosity=verbosity)
