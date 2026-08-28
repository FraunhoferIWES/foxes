from __future__ import annotations

import numpy as np
from xarray import Dataset
from typing import TYPE_CHECKING, Any

import foxes.variables as FV

from .output import Output

if TYPE_CHECKING:
    from foxes.core import Algorithm


class SingleTurbineWakeFields(Output):
    """
    Computes and stores the wake fields for individual turbines.

    Attributes
    ----------
    algo
        The algorithm for point calculation
    farm_results
        The farm results
    turbine_wakes
        A dictionary storing the wake field for each turbine, keyed by turbine index.
        Values are `xarray.Dataset` objects representing the wake field for each turbine,
        or `None` if not yet computed.

    """

    def __init__(
        self,
        algo: Algorithm,
        farm_results: Dataset,
        **kwargs: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        algo
            The algorithm for point calculation
        farm_results
            The farm results
        **kwargs
            Additional keyword arguments passed to the base class.

        """
        super().__init__(**kwargs)
        self.algo = algo
        self.farm_results = farm_results
        self.turbine_wakes: dict[int, Dataset | None] = {
            ti: None for ti in range(algo.n_turbines)
        }

    def calculate_wake_field(self, turbine_index: int, store: bool = True) -> Dataset:
        """
        Calculates the wake field for a specific turbine.

        Parameters
        ----------
        turbine_index
            The index of the turbine for which to calculate the wake field.
        store
            Whether to store the calculated wake field in the `turbine_wakes` dictionary.

        Returns
        -------
        Dataset
            The calculated wake field as an `xarray.Dataset`.

        """

        # prepare:
        coords = {}
        dvars = {}

        # extract origin as turbine position:
        p0 = []
        for v in [FV.X, FV.Y, FV.H]:
            d = self.farm_results[v].values[:, turbine_index]
            if np.min(d) != np.max(d):
                raise ValueError(
                    f"{self.name}: Require state independent values for '{v}', found range {np.min(d)}-{np.max(d)} for turbine {turbine_index}"
                )
            p0.append(d[0])
        p0 = np.stack(p0, axis=-1)
        dvars["origin"] = ((FV.TXYH,), p0)
        TODO

        # Create dataset:
        wake_field = Dataset(
            coords=coords,
            data_vars=dvars,
            attrs={
                "turbine_index": turbine_index,
            },
        )

        if store:
            self.turbine_wakes[turbine_index] = wake_field

        return wake_field
