import numpy as np
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

    def __init__(self, farm, farm_results, algo=None, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        farm: foxes.core.WindFarm
            The wind farm object
        farm_results: xarray.Dataset
            The farm results
        algo: foxes.core.Algorithm, optional
            The algorithm object, used to get the nominal power for the farms if available
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(**kwargs)
        self.farm = farm
        self.algo = algo
        self.results = merge(
            (
                farm_results,
                DataArray(farm.wind_farm_list, dims=[FC.TURBINE], name=FC.FARM),
                DataArray(farm.cluster_list, dims=[FC.TURBINE], name=FC.CLUSTER),
            ),
            join="exact",
        )

        if algo is not None:
            assert self.farm == self.algo.farm, (
                "The algorithm's farm does not match the output's farm"
            )

        self._agg_farm_results = None
        self._agg_cluster_results = None

    def _aggregate(self, group_key, mapping=None):
        gb = self.results.groupby(group_key)
        aggsum = gb.sum().transpose(FC.STATE, group_key).sortby(group_key)
        aggmean = gb.mean().transpose(FC.STATE, group_key).sortby(group_key)
        results = []
        for v in self.results.data_vars.keys():
            if v in FV.extensive_farm and v in aggsum:
                results.append(aggsum[v])
            elif v in FV.intensive_farm and v in aggmean:
                results.append(aggmean[v])

        trbns = {f: g.sizes[FC.TURBINE] for f, g in gb}
        results.append(
            DataArray(
                list(trbns.values()),
                coords={group_key: list(trbns.keys())},
                dims=(group_key,),
                name="n_turbines",
            ).sortby(group_key)
        )

        if self.algo is not None:
            assert mapping is not None, (
                f"Mapping from {group_key} to turbine indices must be provided when algo is not None"
            )
            Pnom = self.algo.farm.get_P_nominal_array(self.algo)
            Pnomf = {f: np.sum(Pnom[i]) for f, i in mapping.items()}
            results.append(
                DataArray(
                    list(Pnomf.values()),
                    coords={group_key: list(Pnomf.keys())},
                    dims=(group_key,),
                    name=FV.P_NOMINAL,
                ).sortby(group_key)
            )

            if FV.P in self.results.data_vars:
                cap = {
                    f: self.results[FV.AMB_P].values[:, i].sum(axis=1) / Pnom[i].sum()
                    for f, i in mapping.items()
                }
                keys = list(cap.keys())
                cap = np.stack(list(cap.values()), axis=-1)
                results.append(
                    DataArray(
                        cap,
                        coords={
                            FC.STATE: self.results[FC.STATE].values,
                            group_key: keys,
                        },
                        dims=(FC.STATE, group_key),
                        name=FV.AMB_CAP,
                    ).sortby(group_key)
                )

                cap = {
                    f: self.results[FV.P].values[:, i].sum(axis=1) / Pnom[i].sum()
                    for f, i in mapping.items()
                }
                keys = list(cap.keys())
                cap = np.stack(list(cap.values()), axis=-1)
                results.append(
                    DataArray(
                        cap,
                        coords={
                            FC.STATE: self.results[FC.STATE].values,
                            group_key: keys,
                        },
                        dims=(FC.STATE, group_key),
                        name=FV.CAP,
                    ).sortby(group_key)
                )

            eff = {
                f: self.results[FV.P].values[:, i].sum(axis=1)
                / np.maximum(self.results[FV.AMB_P].values[:, i].sum(axis=1), 1e-10)
                for f, i in mapping.items()
            }
            keys = list(eff.keys())
            eff = np.stack(list(eff.values()), axis=-1)
            results.append(
                DataArray(
                    eff,
                    coords={FC.STATE: self.results[FC.STATE].values, group_key: keys},
                    dims=(FC.STATE, group_key),
                    name=FV.EFF,
                ).sortby(group_key)
            )

        if FV.WEIGHT in self.results.data_vars:
            weight = self.results[FV.WEIGHT]
            if weight.dims == (FC.STATE,):
                results.append(weight)
            elif weight.dims == (FC.STATE, FC.TURBINE):
                wgts = {}
                for f, i in mapping.items():
                    wgts[f] = weight.values[:, i].mean(axis=1)
                    wgts[f] /= wgts[f].sum()
                keys = list(wgts.keys())
                wgts = np.stack(list(wgts.values()), axis=-1)
                results.append(
                    DataArray(
                        wgts,
                        coords={
                            FC.STATE: self.results[FC.STATE].values,
                            group_key: keys,
                        },
                        dims=(FC.STATE, group_key),
                        name=FV.WEIGHT,
                    ).sortby(group_key)
                )
            else:
                raise ValueError(
                    f"Unexpected dimensions for weight variable: {weight.dims}"
                )

        return merge(results, join="exact")

    @property
    def agg_farm_results(self):
        """
        Get the aggregated farm results.

        Parameters
        ----------
        algo: foxes.core.Algorithm, optional
            The algorithm object

        Returns
        -------
        xarray.Dataset
            The aggregated farm results

        """
        if self._agg_farm_results is None:
            mapping = self.farm.get_wind_farm_mapping()
            self._agg_farm_results = self._aggregate(FC.FARM, mapping)
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
            mapping = self.farm.get_cluster_mapping()
            self._agg_cluster_results = self._aggregate(FC.CLUSTER, mapping)
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
