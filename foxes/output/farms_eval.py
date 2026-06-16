import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from xarray import DataArray, merge

from foxes.utils import write_nc
import foxes.constants as FC
import foxes.variables as FV

from .farm_results_eval import FarmResultsEval


class WindFarmsEval(FarmResultsEval):
    """
    Output class for aggregating and splitting multiple wind farms.

     Attributes
    ----------
    farm: foxes.core.WindFarm
        The wind farm object
    results: xarray.Dataset
        The farm results

    :group: output

    """

    def __init__(self, farm, farm_results=None, results=None, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        farm: foxes.core.WindFarm
            The wind farm object
        farm_results: xarray.Dataset, optional
            The farm results
        results: xarray.Dataset, optional
            The aggregated results, if already computed
        algo: foxes.core.Algorithm, optional
            The algorithm object, used to get the nominal power for the farms if available
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(farm_results=farm_results, **kwargs)
        self.farm = farm
        self.farm_results = None
        if farm_results is not None:
            self.farm_results = merge(
                (
                    farm_results,
                    DataArray(farm.wind_farm_list, dims=[FC.TURBINE], name=FC.FARM),
                ),
                join="exact",
            )

        if self.algo is not None:
            assert self.farm == self.algo.farm, (
                "The algorithm's farm does not match the output's farm"
            )

        self._LEVEL = FC.FARM
        self._results = results

    def get_power_units(self):
        """
        Gets the power units in Watts for all elements

        Returns
        -------
        P_unit_W: np.ndarray
            The power units in Watts for all elements, shape: (n_elements,)

        """
        data = super().get_power_units()

        mapping = self.get_mapping()
        if mapping is None:
            raise ValueError(
                f"Mapping from {self._LEVEL} to turbine indices is required to get power units for each element"
            )

        P_unit_W = []
        for f in self.results[self._LEVEL].values:
            u = np.unique(data[mapping[f]])
            assert len(u) == 1, (
                f"Multiple power units found for {self._LEVEL} '{f}': {u}"
            )
            P_unit_W.append(u[0])
        P_unit_W = np.array(P_unit_W, dtype=data.dtype)

        return P_unit_W

    def _aggregate(self, mapping=None):
        assert self.farm_results is not None, (
            "farm_results are required for aggregation"
        )

        needs_mapping = self.algo is not None
        if (
            not needs_mapping
            and FV.WEIGHT in self.farm_results.data_vars
            and self.farm_results[FV.WEIGHT].dims == (FC.STATE, FC.TURBINE)
        ):
            needs_mapping = True

        if mapping is None and needs_mapping:
            mapping = self.get_mapping()

        if needs_mapping and mapping is None:
            raise ValueError(
                f"Mapping from {self._LEVEL} to turbine indices is required for aggregation"
            )

        gb = self.farm_results.groupby(self._LEVEL)
        aggsum = gb.sum().transpose(FC.STATE, self._LEVEL).sortby(self._LEVEL)
        aggmean = gb.mean().transpose(FC.STATE, self._LEVEL).sortby(self._LEVEL)
        results = []
        for v in self.farm_results.data_vars.keys():
            if v in FV.extensive_farm and v in aggsum:
                results.append(aggsum[v])
            elif v in FV.intensive_farm and v in aggmean:
                results.append(aggmean[v])

        trbns = {f: g.sizes[FC.TURBINE] for f, g in gb}
        results.append(
            DataArray(
                list(trbns.values()),
                coords={self._LEVEL: list(trbns.keys())},
                dims=(self._LEVEL,),
                name="n_turbines",
            ).sortby(self._LEVEL)
        )

        if self.algo is not None:
            Pnom = self.algo.farm.get_capacity_array(self.algo)
            Pnomf = {f: np.sum(Pnom[i]) for f, i in mapping.items() if f is not None}
            results.append(
                DataArray(
                    list(Pnomf.values()),
                    coords={self._LEVEL: list(Pnomf.keys())},
                    dims=(self._LEVEL,),
                    name=FV.CAP,
                ).sortby(self._LEVEL)
            )

        if FV.WEIGHT in self.farm_results.data_vars:
            weight = self.farm_results[FV.WEIGHT]
            if weight.dims == (FC.STATE,):
                results.append(weight)
            elif weight.dims == (FC.STATE, FC.TURBINE):
                wgts = {}
                for f, i in mapping.items():
                    if f is not None:
                        wgts[f] = weight.values[:, i].mean(axis=1)
                        wsum = wgts[f].sum()
                        if wsum <= 0.0:
                            raise ValueError(
                                f"Cannot normalize '{FV.WEIGHT}' for '{f}': sum is {wsum}"
                            )
                        wgts[f] /= wsum
                keys = list(wgts.keys())
                wgts = np.stack(list(wgts.values()), axis=-1)
                results.append(
                    DataArray(
                        wgts,
                        coords={
                            FC.STATE: self.farm_results[FC.STATE].values,
                            self._LEVEL: keys,
                        },
                        dims=(FC.STATE, self._LEVEL),
                        name=FV.WEIGHT,
                    ).sortby(self._LEVEL)
                )
            else:
                raise ValueError(
                    f"Unexpected dimensions for weight variable: {weight.dims}"
                )

        return merge(results, join="exact")

    @property
    def results(self):
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
        if self._results is None:
            mapping = self.farm.get_wind_farm_mapping()
            self._results = self._aggregate(mapping)
        return self._results

    def get_mapping(self):
        """
        Get the mapping from farm to turbine indices.

        Returns
        -------
        dict
            The mapping from farm to turbine indices

        """
        return self.farm.get_wind_farm_mapping()

    def get_capacity(self):
        """
        Gets the capacity values for each turbine, equals nominal power

        Returns
        -------
        capacity_array: numpy.ndarray
            The capacity array (nominal power) for all turbines, shape: (n_turbines,)

        """
        mapping = self.get_mapping()
        Pnom = super().get_capacity()
        cap = np.array(
            [Pnom[mapping[f]].sum() for f in self.results[self._LEVEL].values]
        )
        return cap

    def split(self):
        """
        Split the results by wind farm.

        Returns
        -------
        dict
            A dictionary with the wind farm names as keys and the corresponding results as values

        """
        assert self.farm_results is not None, "farm_results are required for splitting"
        return {
            farm: self.farm_results.where(self.farm_results[FC.FARM] == farm, drop=True)
            for farm in self.farm.wind_farm_names
        }

    def write_split_nc_files(self, base_name, **kwargs):
        """
        Write the results for each sub element to separate netCDF files.

        Parameters
        ----------
        base_name: str
            The base name for the netCDF files
        kwargs: dict, optional
            Additional parameters for the foxes.utils.write_nc() method

        Returns
        -------
        dict
            A dictionary with the wind farm names as keys and the corresponding results that
            were written to the netCDF files as values

        """
        fdict = self.split()
        for farm, ds in fdict.items():
            fname = f"{base_name}_{farm}.nc"
            fpath = self.get_fpath(fname)
            write_nc(ds, fpath, **kwargs)

        return fdict

    def get_area_mapping_plot_layout(self, area_by_name, n_areas):
        """
        Compute an extent-aware figure size and marker size for area mapping plots.

        Parameters
        ----------
        area_by_name: dict
            Mapping from area names to area geometries. Geometries that
            expose ``p_min()`` and ``p_max()`` are used to derive plot extent.
        n_areas: int
            Number of legend entries (mapped areas) used to scale
            vertical space for readable legends.

        Returns
        -------
        figsize: tuple of float
            Figure size as ``(width, height)`` in inches.
        scatter_size: float
            Scatter marker size passed to matplotlib ``s``.

        :group: output

        """
        bounds = []
        if self.farm.n_turbines:
            bounds.append(self.farm.xy_array)

        for area in area_by_name.values():
            if area is None or not hasattr(area, "p_min") or not hasattr(area, "p_max"):
                continue
            bounds.append(np.stack((area.p_min(), area.p_max()), axis=0))

        if bounds:
            xy = np.concatenate(bounds, axis=0)
            span = np.maximum(np.ptp(xy, axis=0), 1.0)
        else:
            span = np.array([1.0, 1.0], dtype=np.float64)

        aspect = np.clip(span[0] / span[1], 0.75, 2.0)
        n_turbines = max(self.farm.n_turbines, 1)
        plot_height = np.clip(
            max(5.5 + 0.4 * np.log10(n_turbines), 2.5 + 0.28 * n_areas),
            5.5,
            9.5,
        )
        plot_width = np.clip(plot_height * aspect, 6.0, 11.0)
        fig_width = plot_width + 2.4
        scatter_size = float(np.clip(54.0 / np.sqrt(n_turbines), 5.0, 18.0))

        return (fig_width, plot_height), scatter_size

    def write_area_mapping_plot(
        self,
        plot_file,
        areas=None,
        mapping=None,
        geojson_name_key="name",
        title=None,
        verbosity=1,
    ):
        """
        Writes a plot visualizing turbine-to-area mapping.

        Parameters
        ----------
        plot_file: str or pathlib.Path
            The output plot path.
        areas: list or str or pathlib.Path or dict, optional
            The areas to visualize. If None, farm.cluster_areas is used.
        mapping: dict, optional
            Mapping from area names to turbine indices. If None, mapping is
            chosen from self._LEVEL.
        geojson_name_key: str or list of str
            Preferred GeoJSON feature property key(s) used to read area
            names from GeoJSON inputs
        title: str, optional
            The plot title. If None, a default title based on self._LEVEL is used.
        verbosity: int
            The verbosity level. 0 = silent

        :group: output

        """
        from foxes.utils.geojson_utils import normalize_areas_input

        plot_file = self.get_fpath(plot_file)
        if verbosity > 0:
            print(f"{type(self).__name__}: Creating {plot_file}")

        if areas is None:
            if self._LEVEL == FC.CLUSTER:
                area_by_name = self.farm.cluster_areas
                if area_by_name is None:
                    raise ValueError(
                        "No areas provided for plotting and farm.cluster_areas is None"
                    )
            else:
                raise ValueError(
                    f"Areas must be provided when plotting level '{self._LEVEL}'"
                )
        else:
            area_by_name = normalize_areas_input(areas, geojson_name_key)

        if mapping is None:
            if self._LEVEL == FC.CLUSTER:
                mapping = self.farm.get_cluster_mapping()
            elif self._LEVEL == FC.FARM:
                mapping = self.farm.get_wind_farm_mapping()
            else:
                raise ValueError(
                    f"Unknown level '{self._LEVEL}', choice is '{FC.CLUSTER}' or '{FC.FARM}'"
                )

            if mapping is None:
                raise ValueError(
                    "No mapping provided and selected level mapping is None, cannot plot turbine-to-area mapping"
                )

        area_names = list(mapping.keys())
        palette = []
        for cmap_name in ("tab20", "tab20b", "tab20c"):
            cmap = plt.get_cmap(cmap_name)
            palette.extend([cmap(i) for i in range(cmap.N)])

        if len(area_names) > len(palette):
            cmap = plt.get_cmap("gist_ncar")
            palette = [
                cmap(v) for v in np.linspace(0.0, 1.0, len(area_names), endpoint=False)
            ]

        colors = {n: palette[i] for i, n in enumerate(area_names)}
        figsize, scatter_size = self.get_area_mapping_plot_layout(
            area_by_name, len(area_names)
        )

        fig, ax = plt.subplots(figsize=figsize)

        for name in area_names:
            area = area_by_name.get(name, None)
            if area is None:
                continue
            area.add_to_figure(
                ax,
                show_boundary=True,
                fill_mode=None,
                pars_boundary={
                    "edgecolor": colors[name],
                    "linewidth": 1.2,
                    "zorder": 20,
                },
            )

        turbine_area = [None] * len(self.farm.turbines)
        for name, inds in mapping.items():
            for i in inds:
                turbine_area[i] = name

        has_unmapped = any(name is None for name in turbine_area)

        for i, t in enumerate(self.farm.turbines):
            name = turbine_area[i]
            color = colors.get(name, "lightgray") if name is not None else "lightgray"
            ax.scatter(
                t.xy[0],
                t.xy[1],
                color=color,
                s=scatter_size,
                alpha=0.65,
                zorder=10,
            )

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=colors[n],
                alpha=0.8,
                label=f"{n}: {len(mapping[n])}",
            )
            for n in area_names
            if n is not None
        ]
        if has_unmapped or None in area_names:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color="lightgray",
                    alpha=0.8,
                    label=f"(none): {turbine_area.count(None)}",
                )
            )
        ax.legend(
            handles=handles,
            title="Areas: turbines",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )
        if title is None:
            title = f"Turbine to {self._LEVEL} mapping"
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, which="major", color="0.85", linestyle="-", linewidth=0.8)
        fig.subplots_adjust(right=max(0.62, 1.0 - 2.1 / figsize[0]))

        fpath = self.get_fpath(plot_file)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
