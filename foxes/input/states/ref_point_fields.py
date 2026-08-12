import numpy as np

from foxes.config import config
from foxes.core import States, MData, FData, TData, run_with_engine, WindFarm, Turbine
from foxes.utils import get_utm_zone, from_lonlat, delta_wd, wd2uv, uv2wd
from foxes.algorithms import Downwind
import foxes.constants as FC
import foxes.variables as FV


class SectorSimRefPointField(States):
    """
    Combines field data representing wind direction sectors and timeseries
    data at one reference point into a timeseries of fields.

    Attributes
    ----------
    field_states: foxes.core.States
        Field data states
    ref_point_states: foxes.core.States
        Reference point states
    ref_point: array_like
        The [x, y, h] reference point coordinates
    output_vars: list of str
        The output variables, if None, all field_states variables are used
    fixed_vars: dict
        Fixed variables, e.g. {"var_name": var_value}
    apply_blending: bool
        Whether to blend between wind direction sectors
    check_nans: bool
        Whether to check for NaN values

    :group: input.states

    """

    def __init__(
        self,
        field_states,
        ref_point_states,
        ref_point,
        ref_point_is_lonlat=False,
        utm_zone=None,
        output_vars=None,
        fixed_vars={},
        apply_blending=True,
        check_nans=True,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        field_states: foxes.core.States
            Field data states
        ref_point_states: foxes.core.States
            Reference point states
        ref_point: array_like
            The [x, y, h] reference point coordinates
        ref_point_is_lonlat: bool, optional
            Whether the reference point coordinates are in longitude/latitude
        utm_zone: str, optional
            The UTM zone for the reference point coordinates, if applicable.
            Either a string like "32N" or None for definition by field or ref point states
            or automatic detection based on the reference point coordinates.
        output_vars: list of str, optional
            The output variables, if None, all field_states variables are used
        fixed_vars: dict, optional
            Fixed variables, e.g. {"var_name": var_value}
        apply_blending: bool, optional
            Whether to blend between wind direction sectors
        check_nans: bool, optional
            Whether to check for NaN values

        """
        super().__init__(**kwargs)
        self.field_states = field_states
        self.ref_point_states = ref_point_states
        self.ref_point = np.asarray(ref_point)
        if self.ref_point.shape != (3,):
            raise ValueError(
                f"States '{self.name}': Expecting ref_point shape (3,), got {self.ref_point.shape}"
            )
        self.output_vars = output_vars
        self.fixed_vars = fixed_vars
        self.apply_blending = apply_blending
        self.check_nans = check_nans

        self.__ref_point_is_lonlat = ref_point_is_lonlat
        self.__utm_zone = utm_zone

    def output_point_vars(self, algo):
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
        return self.output_vars

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            All sub models

        """
        return [self.ref_point_states]  # keep field_states out of the loop

    def _lonlat_to_utm(self, verbosity=0):
        """Helper function to convert lonlat reference point to UTM coordinates"""
        if self.__ref_point_is_lonlat:
            if not config.utm_zone_set and self.__utm_zone is None:
                zone = get_utm_zone(self.ref_point[None, :2])
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
            lonlat = self.ref_point[:2].copy()
            self.ref_point[:2] = from_lonlat(self.ref_point[None, :2])[0]
            if verbosity > 0:
                print(
                    f"States '{self.name}': ref_point lon/lat {lonlat} converted to UTM coordinates {self.ref_point[:2]} using zone {zone}"
                )
            self.__ref_point_is_lonlat = False

        elif self.__utm_zone is not None:
            raise ValueError(
                f"States '{self.name}': ref_point_is_lonlat is False, but utm_zone is given: {self.__utm_zone}. This is not allowed."
            )

    def load_data(self, algo, loaded_data, force=False, verbosity=0):
        """
        Load and/or create all model data that is subject to chunking.

        Such data should not be stored under self, for memory reasons. The
        data returned here will automatically be chunked and then provided
        as part of the mdata object during calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        loaded_data: dict
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force: bool
            Overwrite existing data
        verbosity: int
            The verbosity level, 0 = silent

        """

        self.REF_DATA = self.var("ref_data")
        if self.REF_DATA not in loaded_data["data_vars"] or force:
            self.COORDS0 = self.var("coords0")
            self.VARS0 = self.var("vars0")
            self.EXTRA0 = self.var("extra0")
            self.STATE0 = self.var(FC.STATE + "0")
            self.REF_VARS = self.var("ref_vars")
            self.WD_BIN_DATA = self.var("wd_bin_data")
            self.WD_BIN_DATA_VARS = self.var("wd_bin_data_vars")

            assert self.field_states.load_mode == "preload", (
                f"States '{self.name}': field_states must be in 'preload' mode, got '{self.field_states.load_mode}'"
            )
            if self.field_states.initialized:
                self.field_states.finalize(algo=algo, verbosity=verbosity - 1)

            if verbosity > 0:
                print(
                    f"States '{self.name}': Computing states '{self.field_states.name}' at reference point '{self.ref_point}'"
                )

            # create local algorithm for loading field states:
            self._lonlat_to_utm(verbosity=verbosity)
            farm = WindFarm(name="farm plus ref point")
            for t in algo.farm.turbines:
                farm.add_turbine(
                    Turbine(xy=t.xy, turbine_models=["null_type"]),
                    verbosity=verbosity - 1,
                )
            farm.add_turbine(
                Turbine(xy=self.ref_point[:2], turbine_models=["null_type"]),
                verbosity=verbosity - 1,
            )
            halgo = Downwind(
                farm=farm,
                states=self.field_states,
                rotor_model="centre",
                partial_wakes="centre",
                wake_models=[],
                verbosity=verbosity - 1,
            )
            halgo.initialize()

            # initialize field states and local algorithm:
            ld = halgo.loaded_data
            self.STATE0 = self.var(FC.STATE + "0")
            loaded_data["extra_data"][self.COORDS0] = list(ld["coords"].keys())
            loaded_data["extra_data"][self.VARS0] = list(ld["data_vars"].keys())
            loaded_data["extra_data"][self.EXTRA0] = ld["extra_data"]
            for k, v in ld["coords"].items():
                if k == FC.STATE:
                    loaded_data["coords"][self.STATE0] = v
                else:
                    loaded_data["coords"][k] = v
            for k, v in ld["data_vars"].items():
                if FC.STATE in v[0]:
                    loaded_data["data_vars"][k] = (
                        tuple(self.STATE0 if d == FC.STATE else d for d in v[0]),
                        v[1],
                    )
                else:
                    loaded_data["data_vars"][k] = v

            # create mdata from ld data:
            mdict = {}
            mdims = {}
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
            fdata = FData.from_sizes(n_states=n_states, n_turbines=halgo.n_turbines)
            points = np.zeros((n_states, 1, 3), dtype=self.ref_point.dtype)
            points[:] = self.ref_point[None, None, :]
            tdata = TData.from_points(points=points, mdata=mdata)
            del points

            # compute results at reference point:
            results = run_with_engine(
                halgo.states.calculate,
                algo=halgo,
                mdata=mdata,
                fdata=fdata,
                tdata=tdata,
            )
            assert np.isclose(np.min(tdata[FV.WEIGHT]), np.max(tdata[FV.WEIGHT])), (
                f"States '{self.name}': Field states '{self.field_states.name}' must provide equal weights for all states, got {np.min(tdata[FV.WEIGHT])} - {np.max(tdata[FV.WEIGHT])}"
            )
            if self.output_vars is None:
                self.output_vars = list(results.keys())
            del halgo, mdata, fdata, tdata

            assert FV.WD in results.keys(), (
                f"States '{self.name}': Field states '{self.field_states.name}' must provide '{FV.WD}', got {list(results.keys())}"
            )
            assert FV.WS in results.keys(), (
                f"States '{self.name}': Field states '{self.field_states.name}' must provide '{FV.WS}', got {list(results.keys())}"
            )

            if self.check_nans:
                for k, v in results.items():
                    if np.any(np.isnan(v)):
                        raise ValueError(
                            f"States '{self.name}': Field states '{self.field_states.name}' output variable '{k}' contains {np.sum(np.isnan(v))} NaN values, state indices: {np.where(np.isnan(v))[0].tolist()}"
                        )

            # find wind direction bins at reference point:
            wd_sorted, wd_map, wd_imap = np.unique(
                results[FV.WD][:, 0, 0], return_index=True, return_inverse=True
            )
            if not np.all(wd_map == np.arange(len(wd_map))):
                results = {k: v[wd_map, ...] for k, v in results.items() if k != FV.WD}
            else:
                del results[FV.WD]
            if len(np.unique(wd_imap)) < len(wd_imap):
                for i in wd_imap:
                    w = np.where(wd_imap == i)[0]
                    if len(w) > 1:
                        break
                raise ValueError(
                    f"States '{self.name}': Field states '{self.field_states.name}' at state indices {w.tolist()} have identical wind direction {wd_sorted[w[0]]} at target point"
                )
            wd_plus = np.append(wd_sorted, wd_sorted[0] + 360.0)
            wd_plus = (wd_plus[1:] - wd_plus[:-1]) / 2
            wd_minus = np.insert(wd_sorted, 0, wd_sorted[-1] - 360.0)
            wd_minus = (wd_minus[:-1] - wd_minus[1:]) / 2
            wd_bins = np.stack([wd_sorted, wd_minus, wd_plus], axis=-1)
            loaded_data["coords"][self.WD_BIN_DATA_VARS] = [
                "wd_centre",
                "wd_minus",
                "wd_plus",
            ]
            loaded_data["data_vars"][self.WD_BIN_DATA] = (
                (self.STATE0, self.WD_BIN_DATA_VARS),
                wd_bins,
            )
            del wd_sorted, wd_plus, wd_minus, wd_bins

            # store ref point results in loaded_data:
            loaded_data["coords"][self.REF_VARS] = list(results.keys())
            loaded_data["data_vars"][self.REF_DATA] = (
                (self.STATE0, self.REF_VARS),
                np.stack([d[:, 0, 0] for d in results.values()], axis=-1),
            )

            if verbosity > 0:
                print(
                    f"States '{self.name}': Finished computing states '{self.field_states.name}' at reference point, results: {list(results.keys())}"
                )

    def size(self):
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return self.ref_point_states.size()

    def index(self):
        """
        The index list

        Returns
        -------
        indices: array_like
            The index labels of states, or None for default integers

        """
        return self.ref_point_states.index()

    def calculate(self, algo, mdata, fdata, tdata):
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
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints)

        """

        # prepare
        super().calculate(algo, mdata, fdata, tdata)
        n_states = mdata.n_states
        field_coords0 = mdata.extra_data[self.COORDS0]
        field_vars0 = mdata.extra_data[self.VARS0]
        field_extra0 = mdata.extra_data[self.EXTRA0]
        field_ref_vars = mdata[self.REF_VARS].tolist()
        field_ref_results = mdata[self.REF_DATA]
        wd_bin_centre = mdata[self.WD_BIN_DATA][:, 0]
        wd_bin_minus = mdata[self.WD_BIN_DATA][:, 1]
        wd_bin_plus = mdata[self.WD_BIN_DATA][:, 2]
        n_bins = len(wd_bin_centre)
        ovars = self.output_point_vars(algo)
        out = {v: np.zeros_like(tdata[v]) for v in ovars}

        assert (
            FV.WD in ovars
            and FV.WS in ovars
            and FV.U not in ovars
            and FV.V not in ovars
            and FV.UV not in ovars
        ), (
            f"States '{self.name}': Output variables must include '{FV.WD}', '{FV.WS}' and '{FV.UV}', and must not include '{FV.U}' or '{FV.V}', got {ovars}"
        )

        # evaluate reference point:
        points = np.zeros((n_states, 1, 3), dtype=self.ref_point.dtype)
        points[:] = self.ref_point[None, None, :]
        htdata = TData.from_points(points=points, mdata=mdata)
        ref_results = self.ref_point_states.calculate(algo, mdata, fdata, htdata)
        ref_results = {k: d[:, 0, 0] for k, d in ref_results.items()}
        tdata[FV.WEIGHT] = htdata[FV.WEIGHT]
        tdata.dims[FV.WEIGHT] = (FC.STATE, FC.TARGET, FC.TPOINT)
        del points, htdata

        if self.check_nans:
            for k, v in ref_results.items():
                if np.any(np.isnan(v)):
                    raise ValueError(
                        f"States '{self.name}': Reference point states '{self.ref_point_states.name}' output variable '{k}' contains {np.sum(np.isnan(v))} NaN values"
                    )

        assert FV.WD in ref_results.keys(), (
            f"States '{self.name}': Reference point states '{self.ref_point_states.name}' must provide '{FV.WD}', got {list(ref_results.keys())}"
        )
        assert FV.WS in ref_results.keys(), (
            f"States '{self.name}': Reference point states '{self.ref_point_states.name}' must provide '{FV.WS}', got {list(ref_results.keys())}"
        )

        def _print_wd_error_info(statesi):
            print("\nLOCAL WIND DIRECTION SECTORS:")
            for i, (c, m, p) in enumerate(
                zip(wd_bin_centre, wd_bin_minus, wd_bin_plus)
            ):
                print(f"  {i:3d}: {c:7.2f} ({m:7.2f}, {p:7.2f})")
            print("\nREFERENCE POINT WIND DIRECTIONS:")
            for i in statesi:
                print(
                    f"  {i:4d}: WD = {ref_results[FV.WD][i]:7.2f}, WS = {ref_results[FV.WS][i]:7.2f}"
                )
            print()
            raise ValueError(
                f"States '{self.name}': States '{self.ref_point_states.name}' have {len(statesi)} states that do not match any local wind direction sectors of field states '{self.field_states.name}', state indices: {statesi.tolist()}"
            )

        # find field data in same sector as reference point data and average weights:
        dwd = delta_wd(wd_bin_centre[None, :], ref_results[FV.WD][:, None])
        sel = (dwd > wd_bin_minus[None, :]) & (dwd <= wd_bin_plus[None, :])
        if np.max(np.sum(sel, axis=1)) > 1:
            _print_wd_error_info(np.where(np.sum(sel, axis=1) > 1)[0])
            raise ValueError(
                f"States '{self.name}': Reference point states '{self.ref_point_states.name}' have {np.sum(sel, axis=1)} states that match multiple local wind direction sectors of field states '{self.field_states.name}'"
            )
        if np.min(np.sum(sel, axis=1)) == 0:
            _print_wd_error_info(np.where(np.sum(sel, axis=1) == 0)[0])
            raise ValueError(
                f"States '{self.name}': Reference point states '{self.ref_point_states.name}' have {np.sum(sel, axis=1)} states that do not match any local wind direction sectors of field states '{self.field_states.name}'"
            )

        # prepare states mapping, either with or without blending between wind direction sectors:
        if self.apply_blending:
            # replace WD and WS with UV for blending:
            del out[FV.WS]
            del out[FV.WD]
            out[FV.UV] = np.zeros(
                (n_states, tdata.n_targets, tdata.n_tpoints, 2),
                dtype=config.dtype_double,
            )

            # compute blending weights:
            b0 = np.where(sel)[1]
            b1 = (b0 + np.sign(dwd[sel]).astype(config.dtype_int)) % n_bins
            bf0 = np.abs(dwd[sel]) / np.abs(
                delta_wd(wd_bin_centre[b0], wd_bin_centre[b1])
            )
            del dwd, b0

            # select second sector states:
            sel2 = np.zeros_like(sel)
            sel2[np.arange(sel.shape[0]), b1] = True
            del b1

            # blending requires evaluation of two sectors:
            fstates = np.where(np.any(sel | sel2, axis=0))[0]
            fs2s_0 = np.where(sel[:, fstates])[1]
            fs2s_1 = np.where(sel2[:, fstates])[1]
            sector_maps = [fs2s_0, fs2s_1]
            del fs2s_0, fs2s_1, sel, sel2

        else:
            # single sector case:
            fstates = np.where(np.any(sel, axis=0))[0]
            fs2s = np.where(sel[:, fstates])[1]
            sector_maps = [fs2s]
            bf0 = 1.0
            del dwd, sel, fs2s

        # filter to relevant field states:
        field_ref_results = field_ref_results[fstates]

        # compute field states at target points:
        field_n_states = len(fstates)
        if field_n_states > 0:
            # create mdata:
            mdict = {c: mdata[c] for c in field_coords0}
            mdims = {c: (c,) for c in field_coords0}
            mdict.update({v: mdata[v] for v in field_vars0})
            mdims.update({v: mdata.dims[v] for v in field_vars0})
            if FC.STATE in mdict:
                mdict[FC.STATE] = mdict[FC.STATE][fstates]
            else:
                mdict[FC.STATE] = fstates
            mdims[FC.STATE] = (FC.STATE,)
            for k in mdict.keys():
                if len(mdims[k]) > 0 and mdims[k][0] == self.STATE0:
                    mdims[k] = (FC.STATE,) + mdims[k][1:]
                    mdict[k] = mdict[k][fstates]
            hmdata = MData(
                data=mdict,
                dims=mdims,
                states_i0=0,
                extra_data=field_extra0,
                name="mdata_field",
            )
            del mdict, mdims

            # create fdata:
            hfdata = FData.from_sizes(
                n_states=field_n_states, n_turbines=algo.n_turbines
            )

            # create tdata:
            tpoints = np.zeros(
                (field_n_states, tdata.n_targets, tdata.n_tpoints, 3),
                dtype=config.dtype_double,
            )
            tpoints[:] = tdata[FC.TARGETS][0, None, ...]
            htdata = TData.from_tpoints(
                tpoints=tpoints, tweights=tdata[FC.TWEIGHTS], mdata=hmdata
            )
            del tpoints

            # run field states calculation:
            field_results = self.field_states.calculate(algo, hmdata, hfdata, htdata)
            del hmdata, hfdata, htdata

            # evaluate sectors:
            for bi, fs2s in enumerate(sector_maps):
                # sector weight:
                weight = bf0 if bi == 0 else (1.0 - bf0)

                # compute speedups:
                speedups = {}
                for v in ref_results.keys():
                    if v in field_ref_vars:
                        i = field_ref_vars.index(v)
                        fres = field_ref_results[fs2s, i]
                        speedups[v] = np.where(
                            np.abs(fres) > 1.0e-10,
                            ref_results[v] / fres,
                            0.0,
                        )
                        del fres
                    elif v in out:
                        out[v][:] = ref_results[v][:, None, None]
                    elif self.apply_blending and v == FV.WD:
                        pass
                    else:
                        raise KeyError(
                            f"States '{self.name}': Reference point states '{self.ref_point_states.name}' output variable '{v}' not found in field states variables {field_ref_vars} or output variables {ovars}"
                        )

                def _get_data(v):
                    if v in field_results.keys():
                        return field_results[v][fs2s, :, :]
                    elif v in ref_results.keys():
                        return ref_results[v][:, None, None]
                    else:
                        raise KeyError(
                            f"States '{self.name}': Output variable '{v}' not found in field states variables {list(field_results.keys())} or reference point states variables {list(ref_results.keys())}"
                        )

                # add to output variables:
                for v in ovars:
                    if v == FV.WD:
                        pass
                    elif v in field_results.keys():
                        if v in speedups.keys():
                            if v != FV.WS or not self.apply_blending:
                                w = (
                                    weight[:, None, None]
                                    if isinstance(weight, np.ndarray)
                                    else weight
                                )
                                out[v][:] += (
                                    w
                                    * speedups[v][:, None, None]
                                    * field_results[v][fs2s, :, :]
                                )
                                del w
                            else:
                                uv = wd2uv(field_results[FV.WD], field_results[FV.WS])[
                                    fs2s, :, :
                                ]
                                w = (
                                    weight[:, None, None, None]
                                    if isinstance(weight, np.ndarray)
                                    else weight
                                )
                                out[FV.UV][:] += (
                                    w * speedups[FV.WS][:, None, None, None] * uv
                                )
                                del uv, w
                        else:
                            raise KeyError(
                                f"States '{self.name}': Field states variable '{v}' not found in speedups, got {list(speedups.keys())}"
                            )
                    elif v in ref_results.keys():
                        out[v][:] = ref_results[v][:, None, None]
                    elif v == FV.TI and (
                        FV.TKE in field_results.keys() or FV.TKE in ref_results.keys()
                    ):
                        tke = _get_data(FV.TKE)
                        ws = _get_data(FV.WS)
                        w = (
                            weight[:, None, None]
                            if isinstance(weight, np.ndarray)
                            else weight
                        )
                        out[v][:] += w * np.sqrt(2.0 / 3.0 * tke) / ws
                        del tke, ws, w
                    elif (
                        v == FV.RHO
                        and (FV.P in field_results.keys() or FV.P in ref_results.keys())
                        and (FV.T in field_results.keys() or FV.T in ref_results.keys())
                    ):
                        p = _get_data(FV.P)
                        T = _get_data(FV.T)
                        w = (
                            weight[:, None, None]
                            if isinstance(weight, np.ndarray)
                            else weight
                        )
                        out[v][:] += w * p / (FC.Rd * T)
                        del p, T, w
                    else:
                        raise KeyError(
                            f"States '{self.name}': Output variable '{v}' not found in field states variables {list(field_results.keys())} or reference point states variables {list(ref_results.keys())}"
                        )

        if self.apply_blending:
            uv = out.pop(FV.UV)
            out[FV.WD] = uv2wd(uv)
            out[FV.WS] = np.linalg.norm(uv, axis=-1)
            del uv

        return out
