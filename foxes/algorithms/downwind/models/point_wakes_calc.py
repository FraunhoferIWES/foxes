from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
from typing import TYPE_CHECKING, Any, cast

from foxes.core import PointDataModel
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core import PointDataModelList, WakeModel
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import LoadedData


class PointWakesCalculation(PointDataModel):
    """
    This model calculates wake effects at points of interest.

    Attributes
    ----------
    pvars: list of str
        The variables of interest
    emodels: foxes.core.PointDataModelList
        The extra evaluation models
    emodels_cpars: list of dict
        The calculation parameters for extra models
    wake_models: list of foxes.core.WakeModel
        The wake models to be used

    :group: algorithms.downwind.models

    """

    def __init__(
        self,
        emodels: PointDataModelList | None = None,
        emodels_cpars: list[dict[str, Any]] | None = None,
        wake_models: list[WakeModel] | None = None,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        emodels: foxes.core.PointDataModelList, optional
            The extra evaluation models
        emodels_cpars: list of dict, optional
            The calculation parameters for extra models
        wake_models: list of foxes.core.WakeModel, optional
            Specific wake models to be used

        """
        super().__init__()
        self.pvars: list[str] = []
        self.emodels = emodels
        self.emodels_cpars = [] if emodels_cpars is None else emodels_cpars
        self.wake_models = wake_models

    def sub_models(self) -> list[Any]:
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return [self.emodels] if self.emodels is not None else []

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
        loaded_data = super().initialize(algo, loaded_data, force, verbosity)
        self.pvars = algo.states.output_point_vars(algo)
        return loaded_data

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
        return self.pvars

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int | None = None,
    ) -> dict[str, np.ndarray]:
        """
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        tdata: foxes.core.Data
            The target point data
        downwind_index: int
            The index in the downwind order of the wake
            causing turbine

        Returns
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints)

        """

        def _contribute(
            gmodel: Any,
            tdata: TData,
            oi: int,
            wdeltas: dict[str, np.ndarray],
            wmodel: Any,
        ) -> None:
            """Helper function for contribution of wake deltas to wake results"""

            # reduce to targets within max wake length, if applicable:
            if algo.has_max_wake_length:
                tpts = tdata[FC.TARGETS]
                opts = fdata[FV.TXYH][:, oi]
                tsel = np.any(
                    np.linalg.norm(tpts - opts[:, None, None, :], axis=-1)
                    <= algo.max_wake_length_km * 1e3,
                    axis=(0, 2),
                )
                if not np.any(tsel):
                    return
                wdeltas0 = wdeltas
                tdata = tdata.get_targets_subset(tsel)
                wdeltas = {v: d[:, tsel, ...] for v, d in wdeltas0.items()}

            # compute contributions:
            gmodel.contribute_to_point_wakes(
                algo, mdata, fdata, tdata, oi, wdeltas, wmodel
            )

            # restore full data, if applicable:
            if algo.has_max_wake_length:
                for v in wdeltas0.keys():
                    wdeltas0[v][:, tsel, ...] = wdeltas[v]

        wmodels = (
            list(algo.wake_models.values())
            if self.wake_models is None
            else self.wake_models
        )
        pvrs = self.pvars + [FV.UV]
        for wmodel in wmodels:
            gmodel = algo.ground_models[wmodel.name]

            wdeltas = gmodel.new_point_wake_deltas(algo, mdata, fdata, tdata, wmodel)

            if len(set(pvrs).intersection(wdeltas.keys())):
                if downwind_index is None:
                    assert fdata.n_turbines is not None
                    for oi in range(fdata.n_turbines):
                        _contribute(gmodel, tdata, oi, wdeltas, wmodel)
                else:
                    _contribute(gmodel, tdata, downwind_index, wdeltas, wmodel)

                gmodel.finalize_point_wakes(algo, mdata, fdata, tdata, wdeltas, wmodel)

                for v in tdata.keys():
                    if v in wdeltas:
                        tdata[v] += wdeltas[v]

        if self.emodels is not None:
            self.emodels.calculate(
                algo,
                mdata,
                fdata,
                tdata,
                cast(list[dict[str, Any]], self.emodels_cpars),
            )

        return {v: tdata[v] for v in self.pvars}
