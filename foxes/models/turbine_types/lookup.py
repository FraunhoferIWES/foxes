import numpy as np
import pandas as pd

from foxes.core import TurbineType, FData
from foxes.data import parse_Pct_file_name
from foxes.models.turbine_models import LookupTable
import foxes.variables as FV


class FromLookupTable(TurbineType):
    """
    Calculate power and ct by interpolating
    by using a lookup-table

    Attributes
    ----------
    source: str or pandas.DataFrame
        The file path, static name, or data
    rho: float
        The air density for which the data is valid
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
        input_vars,
        varmap={},
        lookup_pars={},
        rho=None,
        p_ct=1.0,
        p_P=1.88,
        var_ws_ct=FV.REWS2,
        var_ws_P=FV.REWS3,
        pd_file_read_pars={},
        interpn_args={},
        **parameters,
    ):
        """
        Constructor.

        Parameters
        ----------
        data_source: str or pandas.DataFrame
            The file path, static name, or data
        input_vars: list of str
            The foxes input variables
        varmap: dict
            Mapping from foxes variable names
            to column names in the data_source
        lookup_pars: dict
            Additional parameters for the LookupTable model
        rho: float, optional
            The air density for which the data is valid
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
        interpn_args: dict
            Parameters for scipy intern or interp1d
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
        self.rho = rho
        self.p_ct = p_ct
        self.p_P = p_P
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

    def __repr__(self):
        a = f"D={self.D}, H={self.H}, P_nominal={self.P_nominal}, P_unit={self.P_unit}, rho={self.rho}"
        a += f", var_ws_ct={self.WSCT}, var_ws_P={self.WSP}"
        return f"{type(self).__name__}({a})"

    def needs_rews2(self):
        """
        Returns flag for requiring REWS2 variable

        Returns
        -------
        flag: bool
            True if REWS2 is required

        """
        return self.WSCT == FV.REWS2 or self.WSP == FV.REWS2

    def needs_rews3(self):
        """
        Returns flag for requiring REWS3 variable

        Returns
        -------
        flag: bool
            True if REWS3 is required

        """
        return self.WSCT == FV.REWS3 or self.WSP == FV.REWS3

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            All sub models

        """
        return [self._lookup]

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

    def initialize(self, algo, verbosity=0, force=False):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent
        force: bool
            Overwrite existing data

        """
        super().initialize(algo, verbosity, force)
        if self.P_nominal is None:
            col_P = self._lookup.varmap.get(FV.P, FV.P)
            self.P_nominal = np.max(self._lookup._data[col_P].to_numpy())

    def modify_cutin(
        self,
        modify_ct,
        modify_P,
        steps=20,
        iterations=100,
        a=0.55,
        b=0.55,
    ):
        """
        Modify the data such that a discontinuity
        at cutin wind speed is avoided

        Parameters
        ----------
        variable: str
            The target variable
        modify_ct: bool
            Flag for modification of the ct curve
        modify_P: bool
            Flag for modification of the power curve
        steps: int
            The number of wind speed steps between 0 and
            the cutin wind speed
        iterations: int
            The number of iterations
        a: float
            Coefficient for iterative mixing
        b: float
            Coefficient for iterative mixing

        """
        if modify_ct or modify_P:
            raise NotImplementedError

        else:
            super().modify_cutin(modify_ct, modify_P)

    def calculate(self, algo, mdata, fdata, st_sel):
        """
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
        # prepare data for lookup:
        input_vars = self._lookup.input_vars
        fdata_lookup = FData(
            data={v: fdata[v] for v in input_vars},
            dims={v: fdata.dims[v] for v in input_vars},
            loop_dims=fdata.loop_dims,
        )
        for v in self.output_farm_vars(algo):
            fdata_lookup.add(v, fdata[v], fdata.dims[v])

        rews2 = None
        rews3 = None
        if self.WSP != FV.REWS or self.WSCT != FV.REWS:
            rews2 = fdata[self.WSCT].copy()
            rews3 = fdata[self.WSP].copy()

        # apply air density correction:
        if self.rho is not None:
            if rews2 is None:
                rews2 = fdata[self.WSCT].copy()
                rews3 = fdata[self.WSP].copy()

            # correct wind speed by air density, such
            # that in the partial load region the
            # correct value is reconstructed:
            rho = fdata[FV.RHO][st_sel]
            rews3[st_sel] *= (self.rho / rho) ** (1.0 / 3.0)
            del rho

        # in yawed case, calc yaw corrected wind speed:
        if FV.YAWM in fdata and (self.p_P is not None or self.p_ct is not None):
            if rews2 is None:
                rews2 = fdata[self.WSCT].copy()
                rews3 = fdata[self.WSP].copy()

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
                rews2[st_sel] *= (cosm**self.p_ct) ** 0.5
            if self.p_P is not None:
                rews3[st_sel] *= (cosm**self.p_P) ** (1.0 / 3.0)
            del yawm, cosm

        # run lookup:
        if rews2 is None:
            out = self._lookup.calculate(algo, mdata, fdata_lookup, st_sel)
        else:
            fdata_lookup[FV.REWS] = rews2
            ct = self._lookup.calculate(algo, mdata, fdata_lookup, st_sel)[FV.CT]
            fdata_lookup[FV.REWS] = rews3
            out = self._lookup.calculate(algo, mdata, fdata_lookup, st_sel)
            out[FV.CT] = ct

        return out
