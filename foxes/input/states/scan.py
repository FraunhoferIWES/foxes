from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from typing import cast

from foxes.core import Algorithm, FData, LoadedData, MData, States, TData
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC


class ScanStates(States):
    """
    Scan over selected variables

    Parameters
    ----------
    scans: dict[str, numpy.typing.ArrayLike]
        The scans, key: variable name,
        value: scan values

    :group: input.states

    """

    def __init__(self, scans: dict[str, ArrayLike], **kwargs: object) -> None:
        """
        Constructor.

        Parameters
        ----------
        scans: dict[str, numpy.typing.ArrayLike]
            The scans, key: variable name,
            value: scan values
        kwargs: object
            Parameters for the base class

        """
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.scans = {v: np.asarray(d) for v, d in scans.items()}

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
        algo: foxes.core.Algorithm
            The calculation algorithm
        loaded_data: LoadedData
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force: bool
            Overwrite existing data
        verbosity: int
            The verbosity level, 0 = silent

        """
        n_v = len(self.scans)
        shp = [len(v) for v in self.scans.values()]
        self._N = int(np.prod(shp))
        self._vars = list(self.scans.keys())

        data = np.zeros(shp + [n_v], dtype=config.dtype_double)
        for i, d in enumerate(self.scans.values()):
            s = [None] * n_v
            s[i] = np.s_[:]
            st = tuple(s)
            data[..., i] = d[st]
        data = data.reshape(self._N, n_v)

        self.VARS = self.var("vars")
        self.DATA = self.var("data")
        super().load_data(algo, loaded_data, force=force, verbosity=verbosity)
        loaded_data["coords"][self.VARS] = self._vars
        loaded_data["data_vars"][self.DATA] = ((FC.STATE, self.VARS), data)

    def set_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, object]] | None,
        sel: dict[str, object] | None = None,
        isel: dict[str, object] | None = None,
        verbosity: int = 0,
    ) -> None:
        """
        Sets this model status to running, and moves
        all large data to stash.

        The stashed data will be returned by the
        unset_running() function after running calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data_stash: dict[str, dict[str, object]] or None
            Large data stash, this function adds data here, if given.
            Key: model name. Value: dict, large model data
        sel: dict[str, object], optional
            The subset selection dictionary
        isel: dict[str, object], optional
            The index subset selection dictionary
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().set_running(algo, data_stash, sel, isel, verbosity)

        if data_stash is not None:
            data_stash[self.name].update(dict(scans=self.scans))
        del self.scans

    def unset_running(
        self,
        algo: Algorithm,
        data_stash: dict[str, dict[str, object]] | None,
        sel: dict[str, object] | None = None,
        isel: dict[str, object] | None = None,
        verbosity: int = 0,
    ) -> None:
        """
        Sets this model status to not running, recovering large data
        from stash

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data_stash: dict[str, dict[str, object]] or None
            Reconstruct model data from this stash, if given.
            Key: model name. Value: dict, large model data
        sel: dict[str, object], optional
            The subset selection dictionary
        isel: dict[str, object], optional
            The index subset selection dictionary
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().unset_running(algo, data_stash, sel, isel, verbosity)

        if data_stash is not None:
            data = data_stash[self.name]
            self.scans = cast(dict[str, np.ndarray], data.pop("scans"))

    def size(self) -> int:
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return self._N

    def output_point_vars(self, algo: Algorithm) -> list[str]:
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
        return self._vars

    def calculate(  # type: ignore[override]
        self, algo: Algorithm, mdata: MData, fdata: FData, tdata: TData
    ) -> dict[str, np.ndarray]:
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
        results: dict[str, numpy.ndarray]
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints)

        """
        self.ensure_output_vars(algo, tdata)

        for i, v in enumerate(self._vars):
            if v not in tdata:
                tdata[v] = np.zeros_like(tdata[FC.TARGETS][..., 0])
            tdata[v][:] = mdata[self.DATA][:, None, None, i]

        # add weights:
        tdata[FV.WEIGHT] = np.full(
            (mdata.n_states, 1, 1), 1 / self._N, dtype=config.dtype_double
        )
        tdata.dims[FV.WEIGHT] = (FC.STATE, FC.TARGET, FC.TPOINT)

        return {v: tdata[v] for v in self.output_point_vars(algo)}
