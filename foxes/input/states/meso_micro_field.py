import numpy as np
from typing import Any, cast

from foxes.config import config
from foxes.core import (
    Algorithm,
    LoadedData,
    States,
    MData,
    FData,
    TData,
    Model,
    run_with_engine,
    WindFarm,
    Turbine,
)
from foxes.utils import get_utm_zone, from_lonlat, delta_wd, wd2uv, uv2wd
from foxes.algorithms import Downwind
import foxes.constants as FC
import foxes.variables as FV

from .dataset_states import DatasetStates


class MesoMicroField(States):
    """
    Combines field data representing micro scale wind direction sectors
    and meso scale results at multiple reference points into a timeseries of fields.
    """

    def __init__(
        self,
        micro_states: DatasetStates,
        meso_states: DatasetStates,
        ref_points: np.ndarray | list[list[float]] | None = None,
        ref_points_are_lonlat: bool = False,
        ref_height: float | None = None,
        utm_zone: str | tuple[float, float] | None = None,
        output_vars: list[str] | None = None,
        fixed_vars: dict[str, float] = {},
        check_nans: bool = True,
        apply_blending: bool = True,
        **kwargs: object,
    ) -> None:
        """
        Parameters
        ----------
        micro_states
            Micro-scale field data states. Their states must represent
            different wind direction sectors and must be in "preload" mode.
        meso_states
            Meso-scale states evaluated at reference points. These define the final
            states and state weights and are used to scale the micro states.
        ref_points
            The [x, y, h] reference point coordinates, shape (n_ref_points, 3),
            or micro-state grid points with ref_height as height if None.
        ref_points_are_lonlat
            Whether the reference point coordinates are in longitude/latitude.
        ref_height
            The height of the reference points when ref_points is None.
            Defaults to the highest reference point.
        utm_zone
            The UTM zone for the reference point coordinates, if applicable.
            Either a string like "32N" or None to infer it automatically.
        output_vars
            The output variables. If None, all micro_states variables are used.
        fixed_vars
            Fixed variables, e.g. {"var_name": var_value}.
        apply_blending
            Whether to blend between wind direction sectors.
        check_nans
            Whether to check for NaN values.
        """
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.micro_states = micro_states
        self.meso_states = meso_states
        self.output_vars = output_vars
        self.fixed_vars = fixed_vars
        self.check_nans = check_nans
        self.ref_height = ref_height
        self.apply_blending = apply_blending

        self.ref_points = None
        if ref_points is not None:
            self.ref_points = np.asarray(ref_points)
            if len(self.ref_points.shape) != 2 or self.ref_points.shape[1] != 3:
                raise ValueError(
                    f"States '{self.name}': Expecting ref_points shape (N, 3), got {self.ref_points.shape}"
                )

        self.__ref_points_are_lonlat = ref_points_are_lonlat
        self.__utm_zone = utm_zone

    def output_point_vars(self, algo: Algorithm) -> list[str]:
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
        assert self.output_vars is not None
        return self.output_vars

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls
            All sub models

        """
        return [self.meso_states]  # keep micro_states out of the loop

    def _lonlat_to_utm(self, verbosity: int = 0) -> None:
        """Helper function to convert lonlat reference point to UTM coordinates"""
        if self.__ref_points_are_lonlat:
            assert self.ref_points is not None
            if not config.utm_zone_set and self.__utm_zone is None:
                zone = get_utm_zone(self.ref_points[None, :2])
            elif self.__utm_zone is None:
                zone = config.utm_zone
            elif isinstance(self.__utm_zone, str):
                zone = (int(self.__utm_zone[:-1]), self.__utm_zone[-1])
            elif len(self.__utm_zone) == 2:
                lonlat = np.asarray(self.__utm_zone)
                zone = get_utm_zone(lonlat[None, :])
            else:
                raise ValueError(
                    f"States '{self.name}': invalid utm_zone argument: {self.__utm_zone}"
                )
            if not config.utm_zone_set:
                config.set_utm_zone(*zone)
            elif config.utm_zone != zone:
                raise ValueError(
                    f"States '{self.name}': ref_point_is_lonlat is True, but config.utm_zone = {config.utm_zone} differs from determined zone {zone}"
                )
            lonlat = self.ref_points[:, :2].copy()
            self.ref_points[:, :2] = from_lonlat(self.ref_points[:, :2])
            if verbosity > 0:
                print(
                    f"States '{self.name}': ref_point lon/lat {lonlat} converted to UTM coordinates {self.ref_points[:, :2]} using zone {zone}"
                )
            self.__ref_points_are_lonlat = False

        elif self.__utm_zone is not None:
            raise ValueError(
                f"States '{self.name}': ref_points_are_lonlat is False, but utm_zone is given: {self.__utm_zone}. This is not allowed."
            )

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

        self.REF_DATA = self.var("ref_data")
        if self.REF_DATA not in loaded_data["data_vars"] or force:
            self.COORDS0 = self.var("coords0")
            self.VARS0 = self.var("vars0")
            self.EXTRA0 = self.var("extra0")
            self.STATE0 = self.var(FC.STATE + "0")
            self.REF_POINTS = self.var("ref_points")
            self.REF_POINT = self.var("ref_point")
            self.REF_VARS = self.var("ref_vars")
            self.WD_BIN_DATA = self.var("wd_bin_data")
            self.WD_BIN_DATA_VARS = self.var("wd_bin_data_vars")

            # update ref points:
            if self.ref_points is None:
                self.ref_points = self.meso_states.get_grid_points(
                    loaded_data=loaded_data, all_heights=False, height=self.ref_height
                )
                self.ref_height = self.ref_points[0, 2]
                if verbosity > 0:
                    print(
                        f"States '{self.name}': Using micro states grid point locations as reference points, shape: {self.ref_points.shape}, ref_height: {self.ref_height} m"
                    )
            self._lonlat_to_utm(verbosity=verbosity)
            n_points = len(self.ref_points)
            assert n_points > 0, (
                f"States '{self.name}': No reference points found, ref_points: {self.ref_points}, ref_height: {self.ref_height}"
            )
            loaded_data["data_vars"][self.REF_POINTS] = (
                (self.REF_POINT, FC.XYH),
                self.ref_points,
            )

            if verbosity > 0:
                print(
                    f"States '{self.name}': Computing states '{self.micro_states.name}' at {n_points} reference points"
                )

            assert self.micro_states.load_mode == "preload", (
                f"States '{self.name}': micro_states must be in 'preload' mode, got '{self.micro_states.load_mode}'"
            )
            if self.micro_states.initialized:
                self.micro_states.finalize(algo=algo, verbosity=verbosity - 1)

            # create local algorithm for loading field states:
            farm = WindFarm(name="farm plus ref points")
            for t in algo.farm.turbines:
                farm.add_turbine(
                    Turbine(xy=t.xy, turbine_models=["null_type"]),
                    verbosity=verbosity - 1,
                )
            for rp in self.ref_points:
                farm.add_turbine(
                    Turbine(xy=rp[:2], turbine_models=["null_type"], H=self.ref_height),
                    verbosity=verbosity - 1,
                )
            halgo = Downwind(
                farm=farm,
                states=self.micro_states,
                rotor_model="centre",
                partial_wakes="centre",
                wake_models=[],
                verbosity=verbosity - 1,
            )

            # initialize field states and local algorithm:
            halgo.init_states(force=True)
            ld = halgo.loaded_data
            self.STATE0 = self.var(FC.STATE + "0")
            loaded_coords = cast(dict[str, Any], loaded_data["coords"])
            loaded_data_vars = cast(dict[str, Any], loaded_data["data_vars"])
            loaded_data["extra_data"][self.COORDS0] = list(ld["coords"].keys())
            loaded_data["extra_data"][self.VARS0] = list(ld["data_vars"].keys())
            loaded_data["extra_data"][self.EXTRA0] = ld["extra_data"]
            for k, v in ld["coords"].items():
                if k == FC.STATE:
                    loaded_coords[self.STATE0] = v
                else:
                    loaded_coords[k] = v
            for k, v in ld["data_vars"].items():
                if FC.STATE in v[0]:
                    loaded_data_vars[k] = (
                        tuple(self.STATE0 if d == FC.STATE else d for d in v[0]),
                        v[1],
                    )
                else:
                    loaded_data_vars[k] = v

            # create mdata from ld data:
            mdict: dict[str, np.ndarray] = {}
            mdims: dict[str, tuple[str, ...]] = {}
            for k, v in ld["coords"].items():
                if isinstance(v, tuple):
                    mdims[k] = v[0]
                    mdict[k] = np.asarray(v[1])
                else:
                    mdims[k] = (k,)
                    mdict[k] = np.asarray(v)
            for k, v in ld["data_vars"].items():
                mdims[k] = v[0]
                mdict[k] = v[1]
            mdata = MData(
                data=mdict,
                dims=mdims,
                states_i0=0,
                extra_data=ld["extra_data"],
                name="mdata0",
            )
            del mdict, mdims

            # create fdata and tdata:
            n_states = mdata.n_states
            assert n_states is not None
            fdata = FData.from_sizes(n_states=n_states, n_turbines=halgo.n_turbines)
            points = np.zeros((n_states, n_points, 3), dtype=self.ref_points.dtype)
            points[:] = self.ref_points[None, :, :]
            tdata = TData.from_points(points=points, mdata=mdata)
            del points

            # compute results at reference point:
            halgo.initialize()
            results = run_with_engine(
                halgo.states.calculate,
                algo=halgo,
                mdata=mdata,
                fdata=fdata,
                tdata=tdata,
            )
            assert np.isclose(np.min(tdata[FV.WEIGHT]), np.max(tdata[FV.WEIGHT])), (
                f"States '{self.name}': Field states '{self.micro_states.name}' must provide equal weights for all states, got {np.min(tdata[FV.WEIGHT])} - {np.max(tdata[FV.WEIGHT])}"
            )
            if self.output_vars is None:
                self.output_vars = list(results.keys())
            del halgo, mdata, fdata, tdata

            assert FV.WD in results.keys(), (
                f"States '{self.name}': Field states '{self.micro_states.name}' must provide '{FV.WD}', got {list(results.keys())}"
            )
            assert FV.WS in results.keys(), (
                f"States '{self.name}': Field states '{self.micro_states.name}' must provide '{FV.WS}', got {list(results.keys())}"
            )

            if self.check_nans:
                for k, v in results.items():
                    if np.any(np.isnan(v)):
                        raise ValueError(
                            f"States '{self.name}': Field states '{self.micro_states.name}' output variable '{k}' contains {np.sum(np.isnan(v))} NaN values, state indices: {np.where(np.isnan(v))[0].tolist()}"
                        )

            # find wind direction bins at reference point:
            wd_bin_data: np.ndarray = np.zeros(
                (n_states, n_points, 3), dtype=config.dtype_double
            )  #  centre, minus, plus
            for pi in range(n_points):
                wd_sorted, wd_map, wd_imap = np.unique(
                    results[FV.WD][:, pi, 0], return_index=True, return_inverse=True
                )
                if not np.all(wd_map == np.arange(len(wd_map))):
                    for k in results.keys():
                        if k != FV.WD:
                            results[k][:, pi, ...] = results[k][wd_map, pi, ...]
                if len(np.unique(wd_imap)) < len(wd_imap):
                    for i in wd_imap:
                        w = np.where(wd_imap == i)[0]
                        if len(w) > 1:
                            break
                    raise ValueError(
                        f"States '{self.name}': Field states '{self.micro_states.name}' at state indices {w.tolist()} have identical wind direction {wd_sorted[w[0]]} at reference point {pi} = {self.ref_points[pi, :].tolist()} m"
                    )
                wdp = np.append(wd_sorted, wd_sorted[0] + 360.0)
                wdp = (wdp[1:] - wdp[:-1]) / 2
                wdm = np.insert(wd_sorted, 0, wd_sorted[-1] - 360.0)
                wdm = (wdm[:-1] - wdm[1:]) / 2
                wd_bin_data[:, pi, :] = np.stack([wd_sorted, wdm, wdp], axis=-1)

                del wdp, wdm, wd_sorted, wd_map, wd_imap
            del results[FV.WD]

            loaded_coords[self.WD_BIN_DATA_VARS] = [
                "wd_centre",
                "wd_minus",
                "wd_plus",
            ]
            loaded_data_vars[self.WD_BIN_DATA] = (
                (self.STATE0, self.REF_POINT, self.WD_BIN_DATA_VARS),
                wd_bin_data,
            )
            del wd_bin_data

            # store ref point results in loaded_data:
            loaded_coords[self.REF_VARS] = list(results.keys())
            loaded_data_vars[self.REF_DATA] = (
                (self.STATE0, self.REF_POINT, self.REF_VARS),
                np.stack([d[:, :, 0] for d in results.values()], axis=-1),
            )

            if verbosity > 0:
                print(
                    f"States '{self.name}': Finished computing states '{self.micro_states.name}' at reference point, results: {list(results.keys())}"
                )

    def size(self) -> int:
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return self.meso_states.size()

    def index(self) -> list[int]:
        """
        The index list

        Returns
        -------
        indices
            The index labels of states, or None for default integers

        """
        return list(self.meso_states.index())

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
            data_stash[self.name] = dict(
                ref_points=self.ref_points,
            )
        self.ref_points = None

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
            data = data_stash[self.name]
            self.ref_points = cast(np.ndarray, data.pop("ref_points"))

    def calculate(
        self, algo: Algorithm, *data: Any, **parameters: Any
    ) -> dict[str, np.ndarray]:
        if len(data) != 3:
            raise TypeError(
                f"States '{self.name}': Expecting 3 data arguments (mdata, fdata, tdata), got {len(data)}"
            )
        mdata, fdata, tdata = data
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
        tdata
            The target point data

        Returns
        -------
        results
            The resulting data, keys: output variable str.
            Values
            (n_states, n_targets, n_tpoints)

        """

        # prepare
        super().calculate(algo, mdata, fdata, tdata)
        n_states = mdata.n_states
        assert n_states is not None
        micro_coords0 = cast(list[str], mdata.extra_data[self.COORDS0])
        micro_vars0 = cast(list[str], mdata.extra_data[self.VARS0])
        micro_extra0 = cast(dict[str, Any], mdata.extra_data[self.EXTRA0])
        micro_ref_vars = cast(list[str], mdata[self.REF_VARS].tolist())
        micro_ref_results = cast(np.ndarray, mdata[self.REF_DATA])
        wd_bin_centre = mdata[self.WD_BIN_DATA][:, :, 0]
        wd_bin_minus = mdata[self.WD_BIN_DATA][:, :, 1]
        wd_bin_plus = mdata[self.WD_BIN_DATA][:, :, 2]
        ref_points = mdata[self.REF_POINTS]
        n_bins = wd_bin_centre.shape[0]
        n_points = len(ref_points)
        n_tpts = tdata.n_targets * tdata.n_tpoints
        ovars = self.output_point_vars(algo)

        assert (
            (FV.WD in ovars and FV.WS in ovars) or (FV.U in ovars and FV.V in ovars)
        ) and FV.UV not in ovars, (
            f"States '{self.name}': Output variables must include either '{FV.WD}' and '{FV.WS}' or '{FV.U}' and '{FV.V}', and must not include '{FV.UV}', got {ovars}"
        )

        # evaluate reference point:
        points = np.zeros((n_states, n_points, 3), dtype=ref_points.dtype)
        points[:] = ref_points[None, :, :]
        htdata = TData.from_points(points=points, mdata=mdata)
        raw_ref_results: dict[str, np.ndarray] = cast(
            dict[str, np.ndarray],
            self.meso_states.calculate(algo, mdata, fdata, htdata),
        )
        ref_results: dict[str, np.ndarray] = {
            str(k): d[:, :, 0] for k, d in raw_ref_results.items()
        }
        tdata[FV.WEIGHT] = htdata[FV.WEIGHT]
        tdata.dims[FV.WEIGHT] = (FC.STATE, FC.TARGET, FC.TPOINT)
        del points, htdata

        if self.check_nans:
            for result_name, result_data in ref_results.items():
                if np.any(np.isnan(result_data)):
                    raise ValueError(
                        f"States '{self.name}': Reference point states '{self.meso_states.name}' output variable '{result_name}' contains {np.sum(np.isnan(result_data))} NaN values"
                    )

        assert FV.WD in ref_results.keys(), (
            f"States '{self.name}': Reference point states '{self.meso_states.name}' must provide '{FV.WD}', got {list(ref_results.keys())}"
        )
        assert FV.WS in ref_results.keys(), (
            f"States '{self.name}': Reference point states '{self.meso_states.name}' must provide '{FV.WS}', got {list(ref_results.keys())}"
        )

        def _print_wd_error_info(statesw: tuple[np.ndarray, ...]) -> None:
            us = np.unique(statesw[0])
            up = np.unique(statesw[1])
            print(f"\nWD MISMATCH STATES: {len(us)}, [{us[0]} - {us[-1]}]")
            print(f"WD MISMATCH POINTS: {len(up)}, [{up[0]} - {up[-1]}]")
            print(
                f"--> ({statesw[0][0]}, {statesw[1][0]}) - ({statesw[0][-1]}, {statesw[1][-1]})\n"
            )

        # find field data in same sector as reference point data and average weights:
        dwd = delta_wd(wd_bin_centre[None, ...], ref_results[FV.WD][:, None, ...])
        sel = (dwd > wd_bin_minus[None, ...]) & (dwd <= wd_bin_plus[None, ...])
        if np.max(np.sum(sel, axis=1)) > 1:
            _print_wd_error_info(np.where(np.sum(sel, axis=1) > 1))
            raise ValueError(
                f"States '{self.name}': Reference point states '{self.meso_states.name}' have states that match multiple local wind direction sectors of field states '{self.micro_states.name}'"
            )
        if np.min(np.sum(sel, axis=1)) == 0:
            _print_wd_error_info(np.where(np.sum(sel, axis=1) == 0))
            raise ValueError(
                f"States '{self.name}': Reference point states '{self.meso_states.name}' have states that do not match any local wind direction sectors of field states '{self.micro_states.name}'"
            )

        # prepare states mapping, either with or without blending between wind direction sectors:
        if self.apply_blending:
            # compute blending weights:
            bs, b0, bp = np.where(sel)
            dwd_sel = dwd[sel]
            b1 = (
                b0 + np.where(dwd_sel >= 0.0, 1, -1).astype(config.dtype_int)
            ) % n_bins
            dbins = np.abs(delta_wd(wd_bin_centre[b0, bp], wd_bin_centre[b1, bp]))
            blend: np.ndarray = np.zeros_like(dwd_sel, dtype=config.dtype_double)
            np.divide(np.abs(dwd_sel), dbins, out=blend, where=dbins > 0.0)
            bf0: np.ndarray = np.zeros((n_states, n_points), dtype=config.dtype_double)
            bf0[bs, bp] = 1.0 - blend
            del dwd_sel, dbins, blend, dwd, b0

            # select second sector states:
            sel2 = np.zeros_like(sel)
            sel2[bs, b1, bp] = True
            del bs, b1, bp

            # blending requires evaluation of two sectors:
            fstates = np.where(np.any(sel | sel2, axis=(0, 2)))[0]
            fs2s_0 = [np.where(sel[:, fstates, pi])[1] for pi in range(n_points)]
            fs2s_1 = [np.where(sel2[:, fstates, pi])[1] for pi in range(n_points)]
            sector_maps = [fs2s_0, fs2s_1]
            del fs2s_0, fs2s_1, sel, sel2

        else:
            # single sector case:
            fstates = np.where(np.any(sel, axis=(0, 2)))[0]
            fs2s = [np.where(sel[:, fstates, pi])[1] for pi in range(n_points)]
            sector_maps = [fs2s]
            bf0 = np.ones((n_states, n_points), dtype=config.dtype_double)
            del dwd, sel, fs2s

        # map meso to micro states:
        micro_ref_results = micro_ref_results[fstates, :, :]
        n_bins = len(fstates)

        # create mdata:
        mdata_dict: dict[str, np.ndarray] = {c: mdata[c] for c in micro_coords0}
        mdata_dims: dict[str, tuple[str, ...]] = {c: (c,) for c in micro_coords0}
        mdata_dict.update({v: mdata[v] for v in micro_vars0})
        mdata_dims.update({v: mdata.dims[v] for v in micro_vars0})
        if FC.STATE in mdata_dict:
            mdata_dict[FC.STATE] = mdata_dict[FC.STATE][fstates]
        else:
            mdata_dict[FC.STATE] = fstates
        mdata_dims[FC.STATE] = (FC.STATE,)
        for k in mdata_dims.keys():
            if len(mdata_dims[k]) > 0 and mdata_dims[k][0] == self.STATE0:
                mdata_dims[k] = (FC.STATE,) + mdata_dims[k][1:]
                mdata_dict[k] = mdata_dict[k][fstates, ...]
        hmdata = MData(
            data=mdata_dict,
            dims=mdata_dims,
            states_i0=0,
            extra_data=micro_extra0,
            name="mdata_field",
        )
        del mdata_dict, mdata_dims

        # create fdata:
        hfdata = FData.from_sizes(n_states=n_bins, n_turbines=algo.n_turbines)

        # create tdata:
        tpoints: np.ndarray = np.zeros(
            (n_bins, tdata.n_targets, tdata.n_tpoints, 3), dtype=config.dtype_double
        )
        tpoints[:] = tdata[FC.TARGETS][0, None, ...]
        htdata = TData.from_tpoints(
            tpoints=tpoints, tweights=tdata[FC.TWEIGHTS], mdata=hmdata
        )
        del tpoints

        # run field states calculation:
        micro_data = self.micro_states.calculate(
            algo,
            hmdata,
            cast(FData, hfdata),
            cast(TData, htdata),
        )
        micro_results_vrs: list[str] = list(micro_data.keys())
        micro_results: np.ndarray = np.stack(
            list(micro_data.values()), axis=-1
        )  # dims (n_bins, n_tpts, n_points, n_vrs)
        n_vrs = len(micro_results_vrs)
        del hmdata, hfdata, htdata, micro_data

        # replace WS, WD by U, V:
        if FV.U in micro_results_vrs or FV.V in micro_results_vrs:
            assert FV.U in micro_results_vrs and FV.V in micro_results_vrs, (
                f"States '{self.name}': Field states '{self.micro_states.name}' must provide both '{FV.U}' and '{FV.V}', got {micro_results_vrs}"
            )
            iu = micro_results_vrs.index(FV.U)
            iv = micro_results_vrs.index(FV.V)
        else:
            assert FV.WS in micro_results_vrs and FV.WD in micro_results_vrs, (
                f"States '{self.name}': Field states '{self.micro_states.name}' must provide both '{FV.WS}' and '{FV.WD}', got {micro_results_vrs}"
            )
            iwd = micro_results_vrs.index(FV.WD)
            iws = micro_results_vrs.index(FV.WS)
            iu = iws
            iv = iwd
            uv = wd2uv(
                micro_results[..., iwd],
                micro_results[..., iws],
            )
            micro_results[..., iu] = uv[..., 0]
            micro_results[..., iv] = uv[..., 1]
            micro_results_vrs[iu] = FV.U
            micro_results_vrs[iv] = FV.V
            del uv

        # evaluate sectors:
        mires: np.ndarray = np.zeros(
            (n_states, n_tpts, n_points, n_vrs), dtype=config.dtype_double
        )
        for bi, fs2s in enumerate(sector_maps):
            # sector weight:
            weight = bf0 if bi == 0 else (1.0 - bf0)

            # compute speedups:
            speedups: dict[str, list[np.ndarray]] = {}
            for v in ref_results.keys():
                if v in micro_ref_vars:
                    i = micro_ref_vars.index(v)
                    speedups[v] = []
                    for pi in range(n_points):
                        mres = micro_ref_results[fs2s[pi], pi, i]
                        speedups[v].append(
                            np.where(
                                np.abs(mres) > 1.0e-10,
                                ref_results[v][:, pi] / mres,
                                0.0,
                            )
                        )
                        del mres
            if FV.WS in speedups.keys():
                speedups[FV.U] = speedups[FV.WS]
                speedups[FV.V] = speedups[FV.WS]

            # apply speedups wrt each reference point:
            for i, v in enumerate(micro_results_vrs):
                d = micro_results[..., i].reshape(n_bins, n_tpts)
                for pi in range(n_points):
                    a = d[fs2s[pi], :]
                    if v in speedups.keys():
                        a *= speedups[v][pi][:, None]
                    w = weight[:, pi, None] if self.apply_blending else weight
                    mires[:, :, pi, i] += w * a
                    del a, w
                del d
            del speedups
        micro_results = mires  # now with dims (n_states, n_tpts, n_points, n_vrs)
        del mires

        # prepare ref point selection:
        refw: np.ndarray = np.zeros((n_points, n_points), dtype=config.dtype_double)
        np.fill_diagonal(refw, 1.0)
        refv = [f"ref_point_{pi}" for pi in range(n_points)]

        # prepare target points for interpolation:
        assert n_states is not None
        points = tdata[FC.TARGETS][..., :2].reshape((n_states, n_tpts, 2))
        pmin = np.min(points, axis=0)
        pmax = np.max(points, axis=0)
        if np.any(pmax - pmin > 1e-4):
            points, up2p = np.unique(
                points.reshape(n_states * n_tpts, 2), axis=0, return_inverse=True
            )
        else:
            points = points[0, :, :]
            up2p = None

        # interpolate to target points:
        refw = self.meso_states.interpolate_data(
            mdata=mdata,
            idims=[FV.X, FV.Y],
            d=refw,
            pts=points,
            vrs=refv,
            state_indices=mdata.get(FC.STATE, None),
        )
        if up2p is not None:
            refw = refw[up2p, :].reshape(n_states, n_tpts, n_points)
            sinds = np.arange(n_states)
            refw = refw[sinds, ...]
            del sinds, up2p
        else:
            refw = refw[None, ...]

        # apply mixing weights:
        micro_results = np.einsum("sprv,spr->spv", micro_results, refw)

        def _get_data(v: str, out: dict[str, np.ndarray]) -> np.ndarray:
            """Helper function for output data extraction"""
            if v in out.keys():
                return out[v]
            elif v in micro_results_vrs:
                i = micro_results_vrs.index(v)
                return micro_results[..., i].reshape(
                    n_states, tdata.n_targets, tdata.n_tpoints
                )
            elif v in [FV.WS, FV.WD]:
                uv = np.stack([micro_results[..., iu], micro_results[..., iv]], axis=-1)
                if v == FV.WD:
                    return uv2wd(uv).reshape(n_states, tdata.n_targets, tdata.n_tpoints)
                else:
                    return np.linalg.norm(uv, axis=-1).reshape(
                        n_states, tdata.n_targets, tdata.n_tpoints
                    )
            elif v in ref_results.keys():
                d = np.einsum("sr,spr->sp", ref_results[v], refw)
                return d.reshape(n_states, tdata.n_targets, tdata.n_tpoints)
            else:
                raise KeyError(
                    f"States '{self.name}': Output variable '{v}' not found in field states variables {micro_results_vrs} or reference point states variables {list(ref_results.keys())}"
                )

        # collect output:
        out: dict[str, np.ndarray] = {}
        for v in ovars:
            if v in out:
                pass
            elif v in micro_results_vrs:
                i = micro_results_vrs.index(v)
                out[v] = micro_results[..., i].reshape(
                    n_states, tdata.n_targets, tdata.n_tpoints
                )
            elif v in [FV.WS, FV.WD]:
                uv = np.stack([micro_results[..., iu], micro_results[..., iv]], axis=-1)
                out[FV.WD] = uv2wd(uv).reshape(
                    n_states, tdata.n_targets, tdata.n_tpoints
                )
                out[FV.WS] = np.linalg.norm(uv, axis=-1).reshape(
                    n_states, tdata.n_targets, tdata.n_tpoints
                )
                del uv
            elif v in ref_results.keys():
                out[v] = _get_data(v, out)
            elif v == FV.TI and (
                FV.TKE in micro_results_vrs or FV.TKE in ref_results.keys()
            ):
                tke = _get_data(FV.TKE, out)
                ws = _get_data(FV.WS, out)
                out[v] = np.sqrt(2.0 / 3.0 * tke) / ws
                del tke, ws
            elif (
                v == FV.RHO
                and (FV.P in micro_results_vrs or FV.P in ref_results.keys())
                and (FV.T in micro_results_vrs or FV.T in ref_results.keys())
            ):
                p = _get_data(FV.P, out)
                T = _get_data(FV.T, out)
                out[v] = p / (FC.Rd * T)
                del p, T
            else:
                raise KeyError(
                    f"States '{self.name}': Output variable '{v}' not found in field states variables {micro_results_vrs} or reference point states variables {list(ref_results.keys())}"
                )

        return out
