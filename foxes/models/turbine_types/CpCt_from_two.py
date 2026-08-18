from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Any

from foxes.data import parse_Pct_two_files
from foxes.utils import PandasFileHelper
from foxes.config import get_input_path
import foxes.constants as FC


from foxes.core import TurbineType
from foxes.data import PCTCURVE
import foxes.variables as FV

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData
    from foxes.core.model import LoadedData


class CpCtFromTwo(TurbineType):
    """
    Calculate power and ct by interpolating
    from cp curve and ct curve data files.

    Attributes
    ----------
    source_P
        The file path for the power curve, static name, or data
    source_ct
        The file path for the ct curve, static name, or data
    col_ws
        The wind speed column
    col_cp
        The cp column
    col_ct
        The ct column
    WSCT
        The wind speed variable for ct lookup
    WSP
        The wind speed variable for cp lookup
    rpars_cp
        Parameters for pandas cp file reading
    rpars_ct
        Parameters for pandas ct file reading


    """

    def __init__(
        self,
        data_source_cp: str | pd.DataFrame,
        data_source_ct: str | pd.DataFrame,
        col_ws_cp_file: str = "ws",
        col_ws_ct_file: str = "ws",
        col_cp: str = "cp",
        col_ct: str = "ct",
        rho: float | None = None,
        var_ws_ct: str = FV.REWS2,
        var_ws_cp: str = FV.REWS3,
        pd_file_read_pars_cp: dict[str, Any] = {},
        pd_file_read_pars_ct: dict[str, Any] = {},
        **parameters: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        data_source_cp
            The file path for the cp curve, static name, or data
        data_source_ct
            The file path for the ct curve, static name, or data
        col_ws_cp_file
            The wind speed column in the file of the cp curve
        col_ws_ct_file
            The wind speed column in the file of the ct curve
        col_cp
            The cp column
        col_ct
            The ct column
        rho
            The air density for which the data is valid
        var_ws_ct
            The wind speed variable for ct lookup
        var_ws_cp
            The wind speed variable for cp lookup
        pd_file_read_pars_cp
            Parameters for pandas cp file reading
        pd_file_read_pars_ct
            Parameters for pandas ct file reading
        parameters
            Additional parameters for TurbineType class

        """
        if not isinstance(data_source_cp, pd.DataFrame) or not isinstance(
            data_source_ct, pd.DataFrame
        ):
            pars = parse_Pct_two_files(data_source_cp, data_source_ct)
        else:
            pars = parameters
        super().__init__(rho_corr_P=None, rho_corr_ct=None, **pars)

        self.source_cp = data_source_cp
        self.source_ct = data_source_ct
        self.col_ws_cp_file = col_ws_cp_file
        self.col_ws_ct_file = col_ws_ct_file
        self.col_cp = col_cp
        self.col_ct = col_ct
        self.WSCT = var_ws_ct
        self.WSCP = var_ws_cp
        self.rpars_cp = pd_file_read_pars_cp
        self.rpars_ct = pd_file_read_pars_ct
        self.rho = rho

        self._data_cp = None
        self._data_ct = None
        self._data_ws_cp = None
        self._data_ws_ct = None

    def __repr__(self) -> str:
        a = f"D={self.D}, H={self.H}, P_nominal={self.P_nominal}, P_unit={self.P_unit}, rho={self.rho}"
        a += f", var_ws_ct={self.WSCT}, var_ws_cp={self.WSCP}"
        return f"{type(self).__name__}({a})"

    def needs_rews2(self) -> bool:
        """
        Returns flag for requiring REWS2 variable

        Returns
        -------
        flag
            True if REWS2 is required

        """
        return self.WSCT == FV.REWS2 or self.WSCP == FV.REWS2

    def needs_rews3(self) -> bool:
        """
        Returns flag for requiring REWS3 variable

        Returns
        -------
        flag
            True if REWS3 is required

        """
        return self.WSCT == FV.REWS3 or self.WSCP == FV.REWS3

    def output_farm_vars(self, algo: Algorithm) -> list[str]:
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo
            The calculation algorithm

        Returns
        -------
        output_vars
            The output variable names

        """
        return [FV.P, FV.CT]

    def load_data(
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        force: bool = False,
        verbosity: int = 0,
    ) -> None:
        """
        Load and/or create all model data that is subject to chunking.

        Such data should not be stored under self, for memory reasons. The
        data returned here will automatically be chunked and then provided
        as part of the mdata object during calculations.

        Parameters
        ----------
        algo
            The calculation algorithm
        loaded_data
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data
        verbosity
            The verbosity level, 0 = silent

        """
        # read power curve:
        if isinstance(self.source_cp, pd.DataFrame):
            self._data_cp = self.source_cp
        else:
            fpath = get_input_path(self.source_cp)
            if not fpath.is_file():
                fpath = algo.dbook.get_file_path(
                    PCTCURVE, self.source_cp, check_raw=False
                )
            self._data_cp = PandasFileHelper.read_file(fpath, **self.rpars_cp)
        assert self._data_cp is not None

        self._data_cp = self._data_cp.set_index(self.col_ws_cp_file).sort_index()
        self._data_ws_cp = self._data_cp.index.to_numpy()
        self._data_cp = self._data_cp[self.col_cp].to_numpy()

        # read ct curve:
        if isinstance(self.source_ct, pd.DataFrame):
            self._data_ct = self.source_ct
        else:
            fpath = get_input_path(self.source_ct)
            if not fpath.is_file():
                fpath = algo.dbook.get_file_path(
                    PCTCURVE, self.source_ct, check_raw=False
                )
            self._data_ct = PandasFileHelper.read_file(fpath, **self.rpars_ct)
        assert self._data_ct is not None

        self._data_ct = self._data_ct.set_index(self.col_ws_ct_file).sort_index()
        self._data_ws_ct = self._data_ct.index.to_numpy()
        self._data_ct = self._data_ct[self.col_ct].to_numpy()

        if self.P_nominal is None and self.rho is not None:
            D = self.D
            assert D is not None, "Rotor diameter D not initialized"
            area = np.pi * (D / 2) ** 2
            self.P_nominal = (
                0.5 * self.rho * area * np.max(self._data_cp) / FC.P_UNITS[self.P_unit]
            )
            if verbosity > 0:
                print(
                    f"Turbine type '{self.name}': Setting P_nominal = {self.P_nominal:.2f} {self.P_unit}"
                )

        super().load_data(algo, loaded_data, force=force, verbosity=verbosity)

    def modify_cutin(
        self,
        modify_ct: bool,
        modify_P: bool,
        steps: int = 20,
        iterations: int = 100,
        a: float = 0.55,
        b: float = 0.55,
    ) -> None:
        """
        Modify the data such that a discontinuity
        at cutin wind speed is avoided

        Parameters
        ----------
        variable
            The target variable
        modify_ct
            Flag for modification of the ct curve
        modify_P
            Flag for modification of the power curve
        steps
            The number of wind speed steps between 0 and
            the cutin wind speed
        iterations
            The number of iterations
        a
            Coefficient for iterative mixing
        b
            Coefficient for iterative mixing

        """
        if modify_ct:
            ws = self._data_ws_ct
            ct = self._data_ct
            assert ws is not None and ct is not None

            i = 0
            try:
                while i < len(ws) and (not modify_ct or ct[i] < 1e-5):
                    i += 1
            except IndexError:
                raise IndexError(
                    f"Turbine type '{self.name}': Failed not determine cutin wind speed. ws = {ws}, ct = {ct}"
                )

            if ws[i] > 0:
                ws = ws[i:]
                ct = ct[i:]

                new_ws = np.linspace(0.0, ws[0], steps + 1, dtype=ws.dtype)
                new_ct = np.zeros_like(new_ws)

                new_ct[-1] = ct[0]
                for it in range(iterations):
                    new_ct[1:-1] = a * new_ct[:-2] + (1 - a) * new_ct[2:]

                self._data_ws_ct = np.concatenate([new_ws[:-1], ws], axis=0)
                self._data_ct = np.concatenate([new_ct[:-1], ct], axis=0)

        if modify_P:
            ws = self._data_ws_cp
            cp = self._data_cp
            assert ws is not None and cp is not None

            i = 0
            try:
                while i < len(ws) and (not modify_P or cp[i] < 0.001):
                    i += 1
            except IndexError:
                raise IndexError(
                    f"Turbine type '{self.name}': Failed not determine cutin wind speed. ws = {ws}, cp = {cp}"
                )

            if ws[i] > 0:
                ws = ws[i:]
                cp = cp[i:]

                new_ws = np.linspace(0.0, ws[0], steps + 1, dtype=ws.dtype)
                new_cp = np.zeros_like(new_ws)

                new_cp[-1] = cp[0]
                for it in range(iterations):
                    new_cp[1:-1] = b * new_cp[:-2] + (1 - b) * new_cp[2:]

                self._data_ws_cp = np.concatenate([new_ws[:-1], ws], axis=0)
                self._data_cp = np.concatenate([new_cp[:-1], cp], axis=0)

        if not modify_ct and not modify_P:
            super().modify_cutin(modify_ct, modify_P)

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        st_sel: slice | np.ndarray = slice(None),
    ) -> dict[str, np.ndarray]:
        """
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        st_sel
            The state-turbine selection,
            shape: (n_states, n_turbines)

        Returns
        -------
        results
            The resulting data, keys: output variable str.
            Values

        """
        self.ensure_output_vars(algo, fdata)
        ws_cp = self._data_ws_cp
        ws_ct = self._data_ws_ct
        data_cp = self._data_cp
        data_ct = self._data_ct
        D = self.D
        assert (
            ws_cp is not None
            and ws_ct is not None
            and data_cp is not None
            and data_ct is not None
            and D is not None
        )
        rews2 = fdata[self.WSCT][st_sel]
        rews3 = fdata[self.WSCP][st_sel]
        rho = fdata[FV.RHO][st_sel]

        # compute yaw misalignment corrections:
        corrects_yawm = FV.YAWM in fdata and (
            self.yawm_corr_P is not None or self.yawm_corr_ct is not None
        )
        rews3, rews2, factor_cp, factor_ct = self.get_rho_yawm_corrections(
            rews_P=rews3,
            rews_ct=rews2,
            yawm=fdata[FV.YAWM][st_sel] if corrects_yawm else None,
        )

        out = {
            FV.P: fdata[FV.P],
            FV.CT: fdata[FV.CT],
        }
        cp = factor_cp * np.interp(rews3, ws_cp, data_cp, left=0.0, right=0.0)
        out[FV.P][st_sel] = (
            0.5 * rho * np.pi * (D / 2) ** 2 * cp * rews3**3 / FC.P_UNITS[self.P_unit]
        )

        out[FV.CT][st_sel] = factor_ct * np.interp(
            rews2, ws_ct, data_ct, left=0.0, right=0.0
        )

        return out

    def finalize(self, algo: Algorithm, verbosity: int = 0) -> None:
        """
        Finalizes the model.

        Parameters
        ----------
        algo
            The calculation algorithm
        verbosity
            The verbosity level

        """
        super().finalize(algo, verbosity)
        del self._data_ws_cp, self._data_ws_ct, self._data_cp, self._data_ct
