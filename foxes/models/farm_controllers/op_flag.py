from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
from xarray import open_dataset, Dataset
from typing import TYPE_CHECKING, Any

from foxes.core import FarmController
from foxes.config import config
import foxes.constants as FC
import foxes.variables as FV

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData
    from foxes.core.model import LoadedData


class OpFlagController(FarmController):
    """
    A basic controller with a flag for
    turbine operation at each state.

    Parameters
    ----------
    non_op_values
        The non-operational values for variables,
        keys: variable str, values: float
    var2ncvar
        The mapping of variable names to NetCDF variable names,
        only needed if data_source is a path to a NetCDF file


    """

    def __init__(
        self,
        data_source: np.ndarray | str | Dataset,
        non_op_values: dict[str, float] | None = None,
        var2ncvar: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        data_source
            The operating flag data, shape: (n_states, n_turbines),
            or path to a NetCDF file
        non_op_values
            The non-operational values for variables,
            keys: variable str, values: float
        var2ncvar
            The mapping of variable names to NetCDF variable names,
            only needed if data_source is a path to a NetCDF file
        kwargs
            Additional keyword arguments for the
            base class constructor

        """
        super().__init__(**kwargs)
        self.data_source = data_source
        self.var2ncvar = {} if var2ncvar is None else var2ncvar

        self.non_op_values = {
            FV.P: 0.0,
            FV.CT: 0.0,
        }
        if non_op_values is not None:
            self.non_op_values.update(non_op_values)

        self._op_flags: np.ndarray | None = None

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
        vrs = set(super().output_farm_vars(algo))
        vrs.update([FV.OPERATING])
        return list(vrs)

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

        if isinstance(self.data_source, np.ndarray):
            self._op_flags = self.data_source

        elif isinstance(self.data_source, Dataset):
            cop = self.var2ncvar.get(FV.OPERATING, FV.OPERATING)
            self._op_flags = self.data_source[cop].to_numpy()

        else:
            if verbosity > 0:
                print(f"OpFlagController: Reading data from {self.data_source}")
            ds = open_dataset(self.data_source, engine=config.nc_engine)
            cop = self.var2ncvar.get(FV.OPERATING, FV.OPERATING)
            self._op_flags = ds[cop].to_numpy()
            del ds

        op_flags_data = self._op_flags
        assert op_flags_data is not None
        assert op_flags_data.ndim == 2, (
            f"OpFlagController data must be 2D representing (n_states, n_turbines), got shape {op_flags_data.shape}"
        )
        if op_flags_data.shape == (algo.n_turbines, algo.n_states):
            pass
        elif op_flags_data.shape in [(1, algo.n_turbines), (algo.n_states, 1)]:
            op_flags_data = (
                np.zeros((algo.n_states, algo.n_turbines), dtype=bool) + op_flags_data
            )
        else:
            raise ValueError(
                f"OpFlagController data shape {op_flags_data.shape} does not broadcast to "
                f"(n_states, n_turbines)=({algo.n_states}, {algo.n_turbines})"
            )
        op_flags = op_flags_data.astype(bool)

        off = np.where(~op_flags)
        turbine_model_names = self.turbine_model_names
        tmall = self._tmall
        assert turbine_model_names is not None and tmall is not None
        for mi in range(len(turbine_model_names)):
            vsel = self._tmodel_sels_var(mi)
            if vsel in loaded_data["data_vars"]:
                tsel = loaded_data["data_vars"][vsel][1]
            else:
                tsel = np.ones((algo.n_states, algo.n_turbines), dtype=bool)
            tsel[off[0], off[1]] = False

            if np.all(tsel):
                loaded_data["data_vars"].pop(vsel, None)
                tmall[mi] = True
            else:
                loaded_data["data_vars"][vsel] = ((FC.STATE, FC.TURBINE), tsel)
                tmall[mi] = False

        loaded_data["data_vars"].pop(FC.TMODEL_SELS, None)
        loaded_data["data_vars"][FV.OPERATING] = (
            (FC.STATE, FC.TURBINE),
            op_flags,
        )

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        pre_rotor: bool,
        downwind_index: int | None = None,
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
        pre_rotor
            Flag for running pre-rotor or post-rotor
            models
        downwind_index
            The index in the downwind order

        Returns
        -------
        results
            The resulting data, keys: output variable str.
            Values have shape (n_states, n_turbines)

        """
        self.ensure_output_vars(algo, fdata)

        # compute data for all operating turbines:
        op = mdata[FV.OPERATING].astype(bool)
        fdata[FV.OPERATING] = op
        results = super().calculate(algo, mdata, fdata, pre_rotor, downwind_index)
        results[FV.OPERATING] = fdata[FV.OPERATING]

        # set non-operating values:
        if downwind_index is None:
            off = np.where(~op)
            for v in self.output_farm_vars(algo):
                if v != FV.OPERATING:
                    fdata[v][off[0], off[1]] = self.non_op_values.get(v, np.nan)
        else:
            off = np.where(~op[:, downwind_index])
            for v in self.output_farm_vars(algo):
                if v != FV.OPERATING:
                    fdata[v][off[0], downwind_index] = self.non_op_values.get(v, np.nan)

        return results
