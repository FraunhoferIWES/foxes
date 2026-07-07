from __future__ import annotations

import numpy as np
from typing import Any

from foxes.core import Model, States, VerticalProfile
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC


class SingleStateStates(States):
    """
    A single uniform state.

    Attributes
    ----------
    ws: float
        The wind speed
    wd: float
        The wind direction
    ti: float
        The TI value
    rho: float
        The air density
    profdicts: dict
        Key: output variable name str, Value: str or dict
        or `foxes.core.VerticalProfile`
    profdata: dict,
        Additional data for profiles

    :group: input.states

    """

    def __init__(
        self,
        ws: float | None = None,
        wd: float | None = None,
        ti: float | None = None,
        rho: float | None = None,
        profiles: dict[str, str | dict[str, Any] | VerticalProfile] | None = None,
        **profdata: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        ws: float, optional
            The wind speed
        wd: float, optional
            The wind direction
        ti: float, optional
            The TI value
        rho: float, optional
            The air density
        profiles: dict, optional
            Key: output variable name str, Value: str or dict
            or `foxes.core.VerticalProfile`
        profdata: dict, optional
            Additional data for profiles

        """
        super().__init__()
        self.ws = ws
        self.wd = wd
        self.ti = ti
        self.rho = rho
        self.profdicts = {} if profiles is None else profiles
        self.profdata = profdata

        if (
            ws is None
            and wd is None
            and ti is None
            and rho is None
            and not len(self.profdicts)
        ):
            raise KeyError(
                "Expecting at least one parameter: ws, wd, ti, rho, profiles"
            )

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return list(self._profiles.values())

    def initialize(
        self, algo, loaded_data=None, force: bool = False, verbosity: int = 0
    ):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        loaded_data: dict, optional
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force: bool
            Overwrite existing data
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        loaded_data: dict
            The loaded data, containing keys "coords", "data_vars", and "extra_data".
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.

        """
        self._profiles = {}
        for v, d in self.profdicts.items():
            if isinstance(d, str):
                self._profiles[v] = VerticalProfile.new(d)
            elif isinstance(d, VerticalProfile):
                self._profiles[v] = d
            elif isinstance(d, dict):
                t = d.pop("type")
                self._profiles[v] = VerticalProfile.new(t, **d)
            else:
                raise TypeError(
                    f"States '{self.name}': Wrong profile type '{type(d).__name__}' for variable '{v}'. Expecting VerticalProfile, str or dict"
                )
        return super().initialize(
            algo, loaded_data=loaded_data, force=force, verbosity=verbosity
        )

    def size(self) -> int:
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return 1

    def output_point_vars(self, algo) -> list[str]:
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
        out = set()
        if self.ws is not None:
            out.add(FV.WS)
        if self.wd is not None:
            out.add(FV.WD)
        if self.ti is not None:
            out.add(FV.TI)
        if self.rho is not None:
            out.add(FV.RHO)
        out.update(list(self._profiles.keys()))

        return list(out)

    def calculate(self, algo, mdata, fdata, tdata) -> dict[str, np.ndarray]:  # type: ignore[override]
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
        tdata: foxes.core.TData
            The target point data

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints)

        """
        self.ensure_output_vars(algo, tdata)

        if self.ws is not None:
            tdata[FV.WS] = np.full(
                (tdata.n_states, tdata.n_targets, tdata.n_tpoints),
                self.ws,
                dtype=config.dtype_double,
            )
        if self.wd is not None:
            tdata[FV.WD] = np.full(
                (tdata.n_states, tdata.n_targets, tdata.n_tpoints),
                self.wd,
                dtype=config.dtype_double,
            )
        if self.ti is not None:
            tdata[FV.TI] = np.full(
                (tdata.n_states, tdata.n_targets, tdata.n_tpoints),
                self.ti,
                dtype=config.dtype_double,
            )
        if self.rho is not None:
            tdata[FV.RHO] = np.full(
                (tdata.n_states, tdata.n_targets, tdata.n_tpoints),
                self.rho,
                dtype=config.dtype_double,
            )

        if len(self._profiles):
            z = tdata[FC.TARGETS][..., 2]
            for k, v in self.profdata.items():
                tdata[k] = v
            for v, p in self._profiles.items():
                pres = p.calculate(tdata, z)
                tdata[v] = pres

        tdata[FV.WEIGHT] = np.full(
            (mdata.n_states, 1, 1), 1.0, dtype=config.dtype_double
        )
        tdata.dims[FV.WEIGHT] = (FC.STATE, FC.TARGET, FC.TPOINT)

        return {v: tdata[v] for v in self.output_point_vars(algo)}
