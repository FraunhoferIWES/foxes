from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Any

from foxes.core import TurbineType, FData
from foxes.data import parse_Pct_file_name
from foxes.models.turbine_models import LookupTable
import foxes.variables as FV

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import MData
    from foxes.core.model import LoadedData, Model


class FromLookupTable(TurbineType):
    """
    Calculate power and ct by interpolating
    by using a lookup-table

    Attributes
    ----------
    source
        The file path, static name, or data
    rho
        The air density for which the data is valid
        or None for no correction
    WSCT
        The wind speed variable for ct lookup
    WSP
        The wind speed variable for power lookup
    rpars
        Parameters for pandas file reading


    """

    def __init__(
        self,
        data_source: str | pd.DataFrame,
        input_vars: list[str],
        varmap: dict[str, str] = {},
        lookup_pars: dict[str, Any] = {},
        rho: float | None = None,
        var_ws_ct: str = FV.REWS2,
        var_ws_P: str = FV.REWS3,
        pd_file_read_pars: dict[str, Any] = {},
        interpn_args: dict[str, Any] = {},
        **parameters: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        data_source
            The file path, static name, or data
        input_vars
            The foxes input variables
        varmap
            Mapping from foxes variable names
            to column names in the data_source
        lookup_pars
            Additional parameters for the LookupTable model
        rho
            The air density for which the data is valid
            or None for no correction
        var_ws_ct
            The wind speed variable for ct lookup
        var_ws_P
            The wind speed variable for power lookup
        pd_file_read_pars
            Parameters for pandas file reading
        interpn_args
            Parameters for scipy intern or interp1d
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
        self.rho = rho
        self.WSCT = var_ws_ct
        self.WSP = var_ws_P
        self.rpars = pd_file_read_pars

        if FV.REWS not in input_vars or len(
            set(input_vars).intersection([FV.WS, FV.REWS2, FV.REWS3])
        ):
            raise KeyError(
                f"Turbine type '{self.name}': Expecting '{FV.REWS}' as wind speed variable in inputv_vars, got {input_vars}"
            )

        iargs = dict(bounds_error=False, fill_value=0)
        iargs.update(interpn_args)
        self._lookup = LookupTable(
            data_source=data_source,
            input_vars=input_vars,
            output_vars=[FV.P, FV.CT],
            varmap=varmap,
            interpn_args=iargs,
            **lookup_pars,
        )

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

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls
            All sub models

        """
        return [self._lookup]

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

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """
        Initializes the model.

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

        Returns
        -------
        loaded_data
            The loaded data, containing keys "coords", "data_vars", and "extra_data".
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.

        """
        loaded_data = super().initialize(
            algo, loaded_data=loaded_data, force=force, verbosity=verbosity
        )
        if self.P_nominal is None:
            col_P = self._lookup.varmap.get(FV.P, FV.P)
            ldata = self._lookup._data
            assert ldata is not None, "Lookup table data not initialized"
            self.P_nominal = np.max(ldata[col_P].to_numpy())
        return loaded_data

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
            raise NotImplementedError

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
        # prepare data for lookup:
        input_vars = self._lookup.input_vars
        fdata_lookup = FData(
            data={v: fdata[v] for v in input_vars},
            dims={v: fdata.dims[v] for v in input_vars},
        )
        for v in self.output_farm_vars(algo):
            fdata_lookup.add(v, fdata[v], fdata.dims[v])

        rews2 = None
        rews3 = None
        factor_P = None
        factor_ct = None
        if self.WSP != FV.REWS or self.WSCT != FV.REWS:
            rews2 = fdata[self.WSCT].copy()
            rews3 = fdata[self.WSP].copy()

        # compute air density and yaw misalignment corrections:
        corrects_rho = (
            FV.RHO in fdata
            and self.rho is not None
            and (self.rho_corr_P is not None or self.rho_corr_ct is not None)
        )
        corrects_yawm = FV.YAWM in fdata and (
            self.yawm_corr_P is not None or self.yawm_corr_ct is not None
        )
        if corrects_rho or corrects_yawm:
            if rews2 is None:
                rews2 = fdata[self.WSCT].copy()
                rews3 = fdata[self.WSP].copy()
            assert rews3 is not None

            rews3s, rews2s, factor_P, factor_ct = self.get_rho_yawm_corrections(
                rews_P=rews3[st_sel],
                rews_ct=rews2[st_sel],
                rho=fdata[FV.RHO][st_sel] if corrects_rho else None,
                rho_ref=self.rho,
                yawm=fdata[FV.YAWM][st_sel] if corrects_yawm else None,
            )
            rews3[st_sel] = rews3s
            rews2[st_sel] = rews2s

        # run lookup:
        if rews2 is None:
            out = self._lookup.calculate(algo, mdata, fdata_lookup, st_sel)
        else:
            assert rews3 is not None
            fdata_lookup[FV.REWS] = rews2
            ct = self._lookup.calculate(algo, mdata, fdata_lookup, st_sel)[FV.CT]
            fdata_lookup[FV.REWS] = rews3
            out = self._lookup.calculate(algo, mdata, fdata_lookup, st_sel)
            out[FV.CT] = ct

        if factor_P is not None:
            out[FV.P][st_sel] *= factor_P
        if factor_ct is not None:
            out[FV.CT][st_sel] *= factor_ct

        return out
