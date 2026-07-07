from __future__ import annotations

from xarray import DataArray, Dataset, merge
from typing import Any

import foxes.constants as FC

from .farms_eval import WindFarmsEval


class ClusterEval(WindFarmsEval):
    """
    Output class for cluster-aware aggregation and area mapping plots.

    :group: output

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Positional arguments for the base class
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(*args, **kwargs)
        if self.farm_results is not None:
            self.farm_results = merge(
                (
                    self.farm_results,
                    DataArray(
                        self.farm.cluster_list, dims=[FC.TURBINE], name=FC.CLUSTER
                    ),
                ),
                join="exact",
            )

        self._LEVEL = FC.CLUSTER

    @property
    def results(self) -> Dataset:
        """
        Get the aggregated cluster results.

        Returns
        -------
        xarray.Dataset
            The aggregated cluster results

        """
        if self._results is None:
            mapping = self.farm.get_cluster_mapping()
            self._results = self._aggregate(mapping)
        return self._results

    def get_mapping(self) -> dict:
        """
        Get the mapping from cluster to turbine indices.

        Returns
        -------
        dict
            The mapping from cluster to turbine indices

        """
        return self.farm.get_cluster_mapping()

    def split(self) -> dict:
        """
        Split the results by cluster.

        Returns
        -------
        dict
            A dictionary with the cluster names as keys and the corresponding results as values

        """
        assert self.farm_results is not None, "farm_results are required for splitting"
        return {
            cluster: self.farm_results.where(
                self.farm_results[FC.CLUSTER] == cluster, drop=True
            )
            for cluster in self.farm.cluster_names
        }
