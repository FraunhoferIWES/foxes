from xarray import DataArray, merge

from .output import Output
from foxes.utils import write_nc
import foxes.constants as FC
import foxes.variables as FV


class MultipleFarmsOutput(Output):
    """
    Output class for multiple wind farms.

     Attributes
    ----------
    farm: foxes.core.WindFarm
        The wind farm object
    results: xarray.Dataset
        The farm results

    :group: output

    """

    def __init__(self, farm, farm_results, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        farm: foxes.core.WindFarm
            The wind farm object
        farm_results: xarray.Dataset
            The farm results
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(**kwargs)
        self.farm = farm
        self.results = merge(
            (
                farm_results,
                DataArray(farm.wind_farm_list, dims=[FC.TURBINE], name=FC.FARM),
                DataArray(farm.cluster_list, dims=[FC.TURBINE], name=FC.CLUSTER),
            ),
            join="exact",
        )

        self._agg_farm_results = None
        self._agg_cluster_results = None

    @property
    def agg_farm_results(self):
        """
        Get the aggregated farm results.

        Returns
        -------
        xarray.Dataset
            The aggregated farm results

        """
        if self._agg_farm_results is None:
            aggsum = self.results.groupby(FC.FARM).sum()
            aggmean = self.results.groupby(FC.FARM).mean()
            self._agg_farm_results = []
            for v in self.results.data_vars.keys():
                if v in FV.extensive_farm and v in aggsum:
                    self._agg_farm_results.append(aggsum[v])
                elif v in FV.intensive_farm and v in aggmean:
                    self._agg_farm_results.append(aggmean[v])
            self._agg_farm_results = merge(self._agg_farm_results, join="exact")
            self._agg_farm_results = self._agg_farm_results.transpose(FC.STATE, FC.FARM)

        return self._agg_farm_results

    @property
    def agg_cluster_results(self):
        """
        Get the aggregated cluster results.

        Returns
        -------
        xarray.Dataset
            The aggregated cluster results

        """
        if self._agg_cluster_results is None:
            aggsum = self.results.groupby(FC.CLUSTER).sum()
            aggmean = self.results.groupby(FC.CLUSTER).mean()
            self._agg_cluster_results = []
            for v in self.results.data_vars.keys():
                if v in FV.extensive_farm and v in aggsum:
                    self._agg_cluster_results.append(aggsum[v])
                elif v in FV.intensive_farm and v in aggmean:
                    self._agg_cluster_results.append(aggmean[v])
            self._agg_cluster_results = merge(self._agg_cluster_results, join="exact")
            self._agg_cluster_results = self._agg_cluster_results.transpose(
                FC.STATE, FC.CLUSTER
            )

        return self._agg_cluster_results

    def write_agg_nc(self, fname, agg, **kwargs):
        """
        Write the aggregated results to a netCDF file.

        Parameters
        ----------
        fname: str
            The file name to write the netCDF file
        agg: str
            The aggregation level, either "wind_farm" or "cluster"
        kwargs: dict, optional
            Additional parameters for the xarray.Dataset.to_netcdf() method

        Returns
        -------
        xarray.Dataset
            The aggregated results that were written to the netCDF file

        """
        if agg == FC.FARM:
            ds = self.agg_farm_results
        elif agg == FC.CLUSTER:
            ds = self.agg_cluster_results
        else:
            raise ValueError(
                f"Unknown aggregation level: {agg}, choice is either '{FC.FARM}' or '{FC.CLUSTER}'"
            )

        fpath = self.get_fpath(fname)
        write_nc(ds, fpath, **kwargs)

        return ds

    def split_by_farm(self):
        """
        Split the results by wind farm.

        Returns
        -------
        dict
            A dictionary with the wind farm names as keys and the corresponding results as values

        """
        return {
            farm: self.results.where(self.results[FC.FARM] == farm, drop=True)
            for farm in self.farm.wind_farm_names
        }

    def split_by_cluster(self):
        """
        Split the results by cluster.

        Returns
        -------
        dict
            A dictionary with the cluster names as keys and the corresponding results as values

        """
        return {
            cluster: self.results.where(self.results[FC.CLUSTER] == cluster, drop=True)
            for cluster in self.farm.cluster_names
        }

    def write_farm_nc_files(self, base_name, **kwargs):
        """
        Write the results for each wind farm to separate netCDF files.

        Parameters
        ----------
        base_name: str
            The base name for the netCDF files, the actual file name will be "{base_name}_{farm}.nc"
        kwargs: dict, optional
            Additional parameters for the xarray.Dataset.to_netcdf() method

        Returns
        -------
        dict
            A dictionary with the wind farm names as keys and the corresponding results that
            were written to the netCDF files as values

        """
        fdict = self.split_by_farm()
        for farm, ds in fdict.items():
            fname = f"{base_name}_{farm}.nc"
            fpath = self.get_fpath(fname)
            write_nc(ds, fpath, **kwargs)

        return fdict

    def write_cluster_nc_files(self, base_name, **kwargs):
        """
        Write the results for each cluster to separate netCDF files.

        Parameters
        ----------
        base_name: str
            The base name for the netCDF files, the actual file name will be "{base_name}_{cluster}.nc"
        kwargs: dict, optional
            Additional parameters for the xarray.Dataset.to_netcdf() method

        Returns
        -------
        dict
            A dictionary with the cluster names as keys and the corresponding results that
            were written to the netCDF files as values

        """
        cdict = self.split_by_cluster()
        for cluster, ds in cdict.items():
            fname = f"{base_name}_{cluster}.nc"
            fpath = self.get_fpath(fname)
            write_nc(ds, fpath, **kwargs)

        return cdict
