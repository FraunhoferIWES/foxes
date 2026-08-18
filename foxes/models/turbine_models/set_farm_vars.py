from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
from typing import TYPE_CHECKING, Any

from foxes.core import TurbineModel
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData
    from foxes.core.model import LoadedData


class SetFarmVars(TurbineModel):
    """
    Set farm data variables to given data.

    Attributes
    ----------
    vars
        The variables to be set
    once
        Flag for running only once


    """

    def __init__(self, once: bool = False) -> None:
        """
        Constructor.

        Parameters
        ----------
        once
            Flag for running only once

        """
        super().__init__()
        self.once = once
        self.vars: list[str] = []
        self.__vdata: list[np.ndarray] = []
        self.__once_done: set[int] = set()
        self.reset()

    def add_var(self, var: str, data: np.ndarray) -> None:
        """
        Add data for a variable.

        Parameters
        ----------
        var
            The variable name
        data
            The data, shape: (n_states, n_turbines)

        """
        if self.initialized:
            raise ValueError(
                f"Model '{self.name}': Cannot add_var after initialization"
            )
        if self.running:
            raise ValueError(f"Model '{self.name}': Cannot add_var while running")
        self.vars.append(var)
        self.__vdata.append(np.asarray(data, dtype=config.dtype_double))

    def reset(self) -> None:
        """
        Remove all variables.
        """
        if self.running:
            raise ValueError(f"Model '{self.name}': Cannot reset while running")
        self.vars = []
        self.__vdata = []

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

        """
        loaded_data = super().initialize(
            algo, loaded_data=loaded_data, force=force, verbosity=verbosity
        )
        self.__once_done = set()
        return loaded_data

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
        return self.vars

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
        super().load_data(algo, loaded_data, force=force, verbosity=verbosity)
        n_states = algo.n_states
        n_turbines = algo.n_turbines
        assert n_states is not None and n_turbines is not None

        for i, v in enumerate(self.vars):
            data: np.ndarray = np.full(
                (n_states, n_turbines), np.nan, dtype=config.dtype_double
            )
            vdata = self.__vdata[i]

            # handle special case of call during vectorized optimization:
            if (
                np.ndim(vdata)
                and vdata.shape[0] != n_states
                and hasattr(algo.states, "n_pop")
            ):
                n_pop = algo.states.n_pop
                n_ost = algo.states.states.size()
                n_trb = n_turbines
                vdata = np.zeros((n_pop, n_ost, n_trb), dtype=config.dtype_double)
                vdata[:] = self.__vdata[i][None, :]
                vdata = vdata.reshape(n_pop * n_ost, n_trb)

            data[:] = vdata
            loaded_data["data_vars"][self.var(v)] = ((FC.STATE, FC.TURBINE), data)

            # special case of turbine positions:
            if v in [FV.X, FV.Y]:
                i = [FV.X, FV.Y].index(v)
                for ti in range(n_turbines):
                    t = algo.farm.turbines[ti]
                    if len(t.xy.shape) == 1:
                        xy: np.ndarray = np.zeros(
                            (n_states, 2), dtype=config.dtype_double
                        )
                        xy[:] = t.xy[None, :]
                        t.xy = xy
                    t.xy[:, i] = np.where(
                        np.isnan(data[:, ti]), t.xy[:, i], data[:, ti]
                    )

            # special case of rotor diameter and hub height:
            if v in [FV.D, FV.H]:
                for ti in range(n_turbines):
                    t = algo.farm.turbines[ti]
                    x: np.ndarray = np.zeros(n_states, dtype=config.dtype_double)
                    if v == FV.D:
                        d0: np.ndarray = np.asarray(t.D, dtype=config.dtype_double)
                        x[:] = d0 if np.ndim(d0) else float(d0)
                        t.D = x
                    else:
                        h0: np.ndarray = np.asarray(t.H, dtype=config.dtype_double)
                        x[:] = h0 if np.ndim(h0) else float(h0)
                        t.H = x
                    x[:] = np.where(np.isnan(data[:, ti]), x, data[:, ti])

    def set_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        isel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """
        Sets this model status to running, and moves
        all large data to stash.

        The stashed data will be returned by the
        unset_running() function after running calculations.

        Parameters
        ----------
        algo
            The calculation algorithm
        data_stash
            Large data stash, this function adds data here, if given.
            Key: model name. Value: dict, large model data
        sel
            The subset selection dictionary
        isel
            The index subset selection dictionary
        verbosity
            The verbosity level, 0 = silent

        """
        super().set_running(algo, data_stash, sel, isel, verbosity)

        if data_stash is not None:
            data_stash[self.name]["vdata"] = self.__vdata
        del self.__vdata

    def unset_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, Any]] | None,
        sel: dict[str, Any] | None = None,
        isel: dict[str, Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """
        Sets this model status to not running, recovering large data
        from stash

        Parameters
        ----------
        algo
            The calculation algorithm
        data_stash
            Reconstruct model data from this stash, if given.
            Key: model name. Value: dict, large model data
        sel
            The subset selection dictionary
        isel
            The index subset selection dictionary
        verbosity
            The verbosity level, 0 = silent

        """
        super().unset_running(algo, data_stash, sel, isel, verbosity)

        if data_stash is not None:
            self.__vdata = data_stash[self.name].pop("vdata")

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
        st_sel: slice or array of bool
            The state-turbine selection,
            for shape: (n_states, n_turbines)

        Returns
        -------
        results
            The resulting data, keys: output variable str.
            Values

        """
        self.ensure_output_vars(algo, fdata)

        i0 = mdata.states_i0(counter=True)
        assert i0 is not None, "Missing states_i0 in mdata"
        if not self.once or i0 not in self.__once_done:
            n_turbines = fdata.n_turbines
            assert n_turbines is not None
            bsel: np.ndarray = np.zeros((fdata.n_states, n_turbines), dtype=np.bool_)
            bsel[st_sel] = True

            for v in self.vars:
                data = mdata[self.var(v)]
                hsel = ~np.isnan(data)
                tsel = bsel & hsel

                fdata[v][tsel] = data[tsel]

            self.__once_done.add(i0)

        return {v: fdata[v] for v in self.vars}
