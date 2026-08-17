from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Any

from foxes.core import TurbineType
from foxes.utils import PandasFileHelper
from foxes.data import PCTCURVE, parse_Pct_file_name
from foxes.config import get_input_path
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData
    from foxes.core.model import LoadedData


class PCtFile(TurbineType):
    """
    Calculate power and ct by interpolating
    from power-ct-curve data file (or pandas DataFrame).

    Attributes
    ----------
    source
        The file path, static name, or data
    col_ws
        The wind speed column
    col_P
        The power column
    col_ct
        The ct column
    rho
        The air density for which the data is valid
        or None for no correction
    WSCT
        The wind speed variable for ct lookup
    WSP
        The wind speed variable for power lookup
    rpars
        Parameters for pandas file reading

    :group: models.turbine_types

    """

    def __init__(
        self,
        data_source: str | pd.DataFrame,
        col_ws: str = "ws",
        col_P: str = "P",
        col_ct: str = "ct",
        rho: float | None = None,
        var_ws_ct: str = FV.REWS2,
        var_ws_P: str = FV.REWS3,
        pd_file_read_pars: dict[str, Any] = {},
        **parameters: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        data_source
            The file path, static name, or data
        col_ws
            The wind speed column
        col_P
            The power column
        col_ct
            The ct column
        rho
            The air density for which the data is valid
            or None for no correction
        var_ws_ct
            The wind speed variable for ct lookup
        var_ws_P
            The wind speed variable for power lookup
        pd_file_read_pars
        parameters
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
        self.WSCT = var_ws_ct
        self.WSP = var_ws_P
        self.rpars = pd_file_read_pars

    def __repr__(self) -> str:
        a = f"D={self.D}, H={self.H}, P_nominal={self.P_nominal}, P_unit={self.P_unit}, rho={self.rho}"
        a += f", var_ws_ct={self.WSCT}, var_ws_P={self.WSP}"
        return f"{type(self).__name__}({a})"

    def needs_rews2(self) -> bool:
        """
        Returns flag for requiring REWS2 variable

        Returns
        -------
        flag
            True if REWS2 is required

        """
        return self.WSCT == FV.REWS2 or self.WSP == FV.REWS2

    def needs_rews3(self) -> bool:
        """
        Returns flag for requiring REWS3 variable

        Returns
        -------
        flag
            True if REWS3 is required

        """
        return self.WSCT == FV.REWS3 or self.WSP == FV.REWS3

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
            Flag to force reloading of data
        verbosity
            The verbosity level, 0 = silent

        """
        self.DATA = self.var("data")
        if self.DATA not in loaded_data["data_vars"] or force:
            super().load_data(algo, loaded_data, force=force, verbosity=verbosity)

            if isinstance(self.source, pd.DataFrame):
                data = self.source
            else:
                fpath = get_input_path(self.source)
                if not fpath.is_file():
                    if verbosity > 0:
                        print(
                            f"Turbine type '{self.name}': Reading static data from context '{PCTCURVE}'"
                        )
                    fpath = algo.dbook.get_file_path(
                        PCTCURVE, self.source, check_raw=False
                    )
                if verbosity > 0:
                    print(f"Turbine type '{self.name}': Reading file", fpath)
                data = PandasFileHelper.read_file(fpath, **self.rpars)

            data = data.set_index(self.col_ws).sort_index()
            data = data.reset_index()[[self.col_ws, self.col_P, self.col_ct]].to_numpy()

            self.data_ws = data[:, 0]
            self.data_P = data[:, 1]
            self.data_ct = data[:, 2]

            self.WS = self.var(FV.WS)
            self.VARS = self.var("vars")
            loaded_data["coords"][self.WS] = self.data_ws
            loaded_data["coords"][self.VARS] = np.asarray([FV.P, FV.CT], dtype=str)
            loaded_data["data_vars"][self.DATA] = (
                (self.WS, self.VARS),
                np.stack([self.data_P, self.data_ct], axis=1),
            )

            if self.P_nominal is None:
                self.P_nominal = np.max(self.data_P) / FC.P_UNITS[self.P_unit]
                if verbosity > 0:
                    print(
                        f"Turbine type '{self.name}': Setting P_nominal = {self.P_nominal:.2f} {self.P_unit}"
                    )

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
        if modify_ct or modify_P:
            ws = self.data_ws
            ct = self.data_ct
            P = self.data_P

            i = 0
            try:
                while (
                    i < len(ws)
                    and (not modify_ct or ct[i] < 1e-5)
                    and (not modify_P or P[i] < 0.1)
                ):
                    i += 1
            except IndexError:
                raise IndexError(
                    f"Turbine type '{self.name}': Failed not determine cutin wind speed. ws = {ws}, ct = {ct}, P = {P}"
                )

            if ws[i] > 0:
                ws = ws[i:]
                ct = ct[i:]
                P = P[i:]

                new_ws = np.linspace(0.0, ws[0], steps + 1, dtype=ws.dtype)
                new_ct = np.zeros_like(new_ws)
                new_P = np.zeros_like(new_ws)

                if modify_ct:
                    new_ct[-1] = ct[0]
                    for it in range(iterations):
                        new_ct[1:-1] = a * new_ct[:-2] + (1 - a) * new_ct[2:]

                if modify_P:
                    new_P[-1] = P[0]
                    for it in range(iterations):
                        new_P[1:-1] = b * new_P[:-2] + (1 - b) * new_P[2:]

                self.data_ws = np.concatenate([new_ws[:-1], ws], axis=0)
                self.data_ct = np.concatenate([new_ct[:-1], ct], axis=0)
                self.data_P = np.concatenate([new_P[:-1], P], axis=0)

        else:
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
        rews2 = fdata[self.WSCT][st_sel]
        rews3 = fdata[self.WSP][st_sel]

        # compute air density and yaw misalignment corrections:
        corrects_rho = (
            FV.RHO in fdata
            and self.rho is not None
            and (self.rho_corr_P is not None or self.rho_corr_ct is not None)
        )
        corrects_yawm = FV.YAWM in fdata and (
            self.yawm_corr_P is not None or self.yawm_corr_ct is not None
        )
        rews3, rews2, factor_P, factor_ct = self.get_rho_yawm_corrections(
            rews_P=rews3,
            rews_ct=rews2,
            rho=fdata[FV.RHO][st_sel] if corrects_rho else None,
            rho_ref=self.rho,
            yawm=fdata[FV.YAWM][st_sel] if corrects_yawm else None,
        )

        if self.WS in mdata and self.DATA in mdata:
            data_ws = mdata[self.WS]
            data_P = mdata[self.DATA][:, 0]
            data_ct = mdata[self.DATA][:, 1]
        else:
            data_ws = self.data_ws
            data_P = self.data_P
            data_ct = self.data_ct

        out = {FV.P: fdata[FV.P], FV.CT: fdata[FV.CT]}
        out[FV.P][st_sel] = factor_P * np.interp(
            rews3, data_ws, data_P, left=0.0, right=0.0
        )
        out[FV.CT][st_sel] = factor_ct * np.interp(
            rews2, data_ws, data_ct, left=0.0, right=0.0
        )

        return out
