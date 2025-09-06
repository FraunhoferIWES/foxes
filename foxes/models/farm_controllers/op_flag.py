import numpy as np
from xarray import open_dataset, Dataset

from foxes.core import FarmController
import foxes.constants as FC
import foxes.variables as FV


class OpFlagController(FarmController):
    """
    A basic controller with a flag for
    turbine operation at each state.

    Parameters
    ----------
    non_op_values: dict
        The non-operational values for variables,
        keys: variable str, values: float
    var2ncvar: dict
        The mapping of variable names to NetCDF variable names,
        only needed if data_source is a path to a NetCDF file

    :group: models.farm_controllers

    """

    def __init__(
        self,
        data_source,
        non_op_values=None,
        var2ncvar={},
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        data_source: numpy.ndarray or str
            The operating flag data, shape: (n_states, n_turbines),
            or path to a NetCDF file
        non_op_values: dict, optional
            The non-operational values for variables,
            keys: variable str, values: float
        var2ncvar: dict
            The mapping of variable names to NetCDF variable names,
            only needed if data_source is a path to a NetCDF file
        kwargs: dict, optional
            Additional keyword arguments for the
            base class constructor

        """
        super().__init__(**kwargs)
        self.data_source = data_source
        self.var2ncvar = var2ncvar

        self.non_op_values = {
            FV.P: 0.0,
            FV.CT: 0.0,
        }
        if non_op_values is not None:
            self.non_op_values.update(non_op_values)

        self._op_flags = None

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
        vrs = set(super().output_farm_vars(algo))
        vrs.update([FV.OPERATING])
        return list(vrs)

    def load_data(self, algo, verbosity=0):
        """
        Load and/or create all model data that is subject to chunking.

        Such data should not be stored under self, for memory reasons. The
        data returned here will automatically be chunked and then provided
        as part of the mdata object during calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        idata: dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """

        idata = super().load_data(algo, verbosity)

        if isinstance(self.data_source, np.ndarray):
            self._op_flags = self.data_source

        elif isinstance(self.data_source, Dataset):
            cop = self.var2ncvar.get(FV.OPERATING, FV.OPERATING)
            self._op_flags = self.data_source[cop].to_numpy()

        else:
            if verbosity > 0:
                print(f"OpFlagController: Reading data from {self.data_source}")
            ds = open_dataset(self.data_source)
            cop = self.var2ncvar.get(FV.OPERATING, FV.OPERATING)
            self._op_flags = ds[cop].to_numpy()
            del ds

        assert self._op_flags.shape == (algo.n_states, algo.n_turbines), (
            f"OpFlagController data shape {self._op_flags.shape} does not match "
            f"(n_states, n_turbines)=({algo.n_states}, {algo.n_turbines})"
        )
        op_flags = self._op_flags.astype(bool)

        off = np.where(~op_flags)
        tmsels = idata["data_vars"][FC.TMODEL_SELS][1]
        tmsels[off[0], off[1], :] = False
        self._tmall = [np.all(t) for t in tmsels]

        idata["data_vars"][FC.TMODEL_SELS] = (
            (FC.STATE, FC.TURBINE, FC.TMODELS),
            tmsels,
        )
        idata["data_vars"][FV.OPERATING] = (
            (FC.STATE, FC.TURBINE),
            op_flags,
        )

        return idata

    def calculate(self, algo, mdata, fdata, pre_rotor, downwind_index=None):
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
        pre_rotor: bool
            Flag for running pre-rotor or post-rotor
            models
        downwind_index: int, optional
            The index in the downwind order

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

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
