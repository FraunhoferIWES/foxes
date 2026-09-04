from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import xarray as xr

import foxes.constants as FC
import foxes.variables as FV
from foxes.data import MODEL_DATA
from foxes.models.wake_models.gaussian import GaussianWakeModel
from foxes.utils.gaussian_pwakes_utils import evaluate_lookup_geometry
from foxes.utils.gaussian_pwakes_utils import load_lookup_dataset
from foxes.utils.gaussian_pwakes_utils import validate_lookup_dataset
from foxes.utils.gaussian_pwakes_utils import gaussian_disc_weight_analytical

from .centre import PartialCentre

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import LoadedData
    from foxes.core.wake_model import WakeModel


class PartialGaussianLookup(PartialCentre):
    """
    Gaussian-only partial wake model using lookup-artifact rotor weights.
    """

    def __init__(
        self,
        lookup_data: xr.Dataset | str | Path | None = None,
        bounds_policy: Literal["clip", "nan", "raise"] = "clip",
    ) -> None:
        """
        Parameters
        ----------
        lookup_data
            Lookup dataset source, either in-memory xarray dataset or path to
            a NetCDF artifact. If ``None``, the packaged
            ``model_data/gaussian_lookup.nc`` artifact is loaded through the
            algorithm data book during initialization.
        bounds_policy
            Out-of-range behavior for radial lookup queries. Choices are
            ``"clip"`` (default), ``"nan"``, and ``"raise"``.
            For ``"clip"``, radial out-of-range points are clipped to table bounds;
            if resulting clipped weights are larger than ``min_weight``,
            ``ValueError`` is raised. Sigma values above the generated upper
            sigma bound always use the large-sigma asymptote.
        """
        super().__init__()
        if bounds_policy not in {"clip", "nan", "raise"}:
            raise ValueError("bounds_policy must be one of {'clip', 'nan', 'raise'}")
        self.lookup_data = lookup_data
        self.bounds_policy = bounds_policy
        self.lookup_dataset_key = self.var("weights")

    def __repr__(self) -> str:
        src = (
            type(self.lookup_data).__name__ if self.lookup_data is not None else "None"
        )
        return (
            f"{type(self).__name__}(lookup_data={src}, "
            f"bounds_policy={self.bounds_policy})"
        )

    def check_wmodel(self, wmodel: WakeModel, error: bool = True) -> bool:
        """
        Check wake model compatibility.

        Parameters
        ----------
        wmodel
            Wake model to test.
        error
            Flag for raising ``TypeError`` when incompatible.

        Returns
        -------
        chk
            True when wake model is compatible.
        """
        if not isinstance(wmodel, GaussianWakeModel):
            if error:
                raise TypeError(
                    f"Partial wakes '{self.name}': Cannot be applied to wake model '{wmodel.name}', since not a GaussianWakeModel"
                )
            return False
        return True

    def load_data(
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        force: bool = False,
        verbosity: int = 0,
    ) -> None:
        """
        Load the validated lookup artifact into model extra data.

        Parameters
        ----------
        algo
            The calculation algorithm.
        loaded_data
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries ``dim_name_str -> dim_array``;
            "data_vars", a dict with entries ``name_str -> (dim_tuple, data_ndarray)``;
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data.
        verbosity
            The verbosity level, where 0 is silent.

        """
        if isinstance(self.lookup_data, xr.Dataset):
            validate_lookup_dataset(self.lookup_data)
            lookup_dataset = self.lookup_data.copy(deep=True)
        elif isinstance(self.lookup_data, (str, Path)):
            lookup_dataset = load_lookup_dataset(self.lookup_data)
        elif self.lookup_data is None:
            lookup_path = algo.dbook.get_file_path(
                MODEL_DATA, "gaussian_lookup.nc", errors=False
            )
            if lookup_path is None:
                raise FileNotFoundError(
                    "Default Gaussian lookup artifact 'gaussian_lookup.nc' was "
                    f"not found in data-book context '{MODEL_DATA}'"
                )
            lookup_dataset = load_lookup_dataset(lookup_path)
        else:
            raise TypeError(
                "lookup_data must be xarray.Dataset, str, pathlib.Path, or None"
            )

        loaded_data["extra_data"][self.lookup_dataset_key] = lookup_dataset
        super().load_data(algo, loaded_data, force=force, verbosity=verbosity)

    def contribute(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
        downwind_index: int,
        wake_deltas: dict[str, np.ndarray],
        wmodel: WakeModel,
    ) -> None:
        """
        Modify wake deltas using lookup-based rotor weights.

        Parameters
        ----------
        algo
            The calculation algorithm.
        mdata
            The model data.
        fdata
            The farm data.
        tdata
            The target point data.
        downwind_index
            The index of the wake-causing turbine in downwind order.
        wake_deltas
            The accumulated wake deltas, keyed by variable name.
        wmodel
            The wake model that provides the wake deltas.

        """
        self.check_wmodel(wmodel, error=True)
        wmodel = cast(GaussianWakeModel, wmodel)

        # Geometry at target rotor centres in wake-frame coordinates:
        wcoos = algo.wake_frame.get_wake_coos(algo, mdata, fdata, tdata, downwind_index)
        x = wcoos[..., 0, 0]
        yz = wcoos[..., 0, 1:3]
        R = np.linalg.norm(yz, axis=-1)
        del wcoos, yz

        # Rotor diameters at state-target points:
        D = self.get_data(
            FV.D,
            FC.STATE_TARGET,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            upcast=True,
        )

        amsi, st_sel = wmodel.calc_amplitude_sigma(
            algo,
            mdata,
            fdata,
            tdata,
            downwind_index,
            x,
        )
        if not np.any(st_sel):
            return

        wdeltas: dict[str, np.ndarray] = {}
        for v, (ampld, sigma) in amsi.items():
            sigma_full = np.ones_like(R, dtype=R.dtype)
            sigma_full[st_sel] = sigma
            weights_full = self._get_weights(mdata, R, D, sigma_full, st_sel)

            wdeltas[v] = ampld * weights_full[st_sel]

        # apply wake-deflection auxiliary effects for WS, consistent with GaussianWakeModel
        if wmodel.affects_ws and FV.WS in wdeltas:
            if FC.WDEFL_ROT_ANGLE in tdata:
                dwd_defl = tdata.pop(FC.WDEFL_ROT_ANGLE)
                dwd_defl = dwd_defl[st_sel].reshape(-1)
                if FV.WD not in wdeltas:
                    wdeltas[FV.WD] = np.zeros_like(wdeltas[FV.WS])
                    wdeltas[FV.WD][:] = dwd_defl
                else:
                    wdeltas[FV.WD] += dwd_defl

            if FC.WDEFL_DWS_FACTOR in tdata:
                dws_defl = tdata.pop(FC.WDEFL_DWS_FACTOR)
                wdeltas[FV.WS] *= dws_defl[st_sel]

        # run superposition models:
        if wmodel.affects_ws and wmodel.has_uv:
            assert wmodel.has_vector_wind_superp, (
                f"{self.name}: Expecting vector wind superposition in wake model '{wmodel.name}', got '{wmodel.wind_superposition}'"
            )
            vec_superp = wmodel.vec_superp
            assert vec_superp is not None
            if FV.UV in wdeltas:
                duv = wdeltas.pop(FV.UV)
            else:
                clwe = {v: d[:, None] for v, d in wdeltas.items()}
                vec_superp.wdeltas_ws2uv(
                    algo,
                    fdata,
                    tdata,
                    downwind_index,
                    clwe,
                    st_sel,
                )
                duv = clwe.pop(FV.UV)[:, 0]
                del clwe, wdeltas[FV.WS]
                if FV.WD in wdeltas:
                    del wdeltas[FV.WD]

            wake_deltas[FV.UV] = vec_superp.add_wake_vector(
                algo,
                mdata,
                fdata,
                tdata,
                downwind_index,
                st_sel,
                wake_deltas[FV.UV],
                duv[:, None],
            )

        for v, d in wdeltas.items():
            try:
                superp = wmodel.superp[v]
            except KeyError:
                raise KeyError(
                    f"Model '{self.name}': Missing wake superposition entry for variable '{v}' in wake model '{wmodel.name}', found {sorted(list(wmodel.superp.keys()))}"
                )

            wake_deltas[v] = superp.add_wake(
                algo,
                mdata,
                fdata,
                tdata,
                downwind_index,
                st_sel,
                v,
                wake_deltas[v],
                d[:, None],
            )

    def _get_weights(
        self,
        mdata: MData,
        r: np.ndarray,
        d: np.ndarray,
        sigma: np.ndarray,
        is_waked: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate lookup weights for the target rotor geometry.

        Parameters
        ----------
        mdata
            The model data containing the lookup dataset.
        r
            Radial distance between wake and target rotor centres.
        d
            Target rotor diameters.
        sigma
            Gaussian wake widths.
        is_waked
            Mask selecting waked state-target points.

        Returns
        -------
        weights
            Rotor-disc averaged Gaussian weights.

        """
        try:
            lookup_dataset = cast(xr.Dataset, mdata.extra_data[self.lookup_dataset_key])
        except KeyError:
            raise ValueError(
                f"Partial wakes '{self.name}': Missing lookup dataset. "
                "Provide 'lookup_data' as Dataset or NetCDF path."
            ) from None
        try:
            min_weight = float(lookup_dataset.attrs["min_weight"])
        except KeyError:
            raise ValueError(
                f"Partial wakes '{self.name}': Lookup dataset is missing "
                "the 'min_weight' attribute."
            ) from None
        if min_weight <= 0.0:
            raise ValueError(
                f"Partial wakes '{self.name}': Lookup dataset attribute "
                "'min_weight' must be > 0."
            )
        return evaluate_lookup_geometry(
            lookup_dataset,
            r=r,
            d=d,
            sigma=sigma,
            is_waked=is_waked,
            bounds_policy=self.bounds_policy,
            masked_value=0.0,
            min_weight=min_weight,
            clip_check_min_weight=min_weight if self.bounds_policy == "clip" else None,
        )


class PartialGaussian(PartialGaussianLookup):
    """
    Gaussian partial wakes using an analytical rotor-disc average.
    """

    def __init__(self, min_weight: float = 1.0e-8) -> None:
        """
        Parameters
        ----------
        min_weight
            Minimal retained rotor-disc weight. Lower values are zeroed.
        """
        PartialCentre.__init__(self)
        if min_weight < 0.0:
            raise ValueError("min_weight must be >= 0")
        self.min_weight = min_weight

    def __repr__(self) -> str:
        return f"{type(self).__name__}(min_weight={self.min_weight})"

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """
        Initialize the analytical partial wakes model.

        Parameters
        ----------
        algo
            The calculation algorithm.
        loaded_data
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries ``dim_name_str -> dim_array``;
            "data_vars", a dict with entries ``name_str -> (dim_tuple, data_ndarray)``;
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data.
        verbosity
            The verbosity level, where 0 is silent.

        Returns
        -------
        loaded_data
            The loaded data, containing keys "coords", "data_vars", and
            "extra_data".

        """
        return PartialCentre.initialize(
            self,
            algo,
            loaded_data=loaded_data,
            force=force,
            verbosity=verbosity,
        )

    def load_data(
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        force: bool = False,
        verbosity: int = 0,
    ) -> None:
        """
        Initialize without loading a lookup artifact.

        Parameters
        ----------
        algo
            The calculation algorithm.
        loaded_data
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries ``dim_name_str -> dim_array``;
            "data_vars", a dict with entries ``name_str -> (dim_tuple, data_ndarray)``;
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data.
        verbosity
            The verbosity level, where 0 is silent.

        """
        PartialCentre.load_data(
            self, algo, loaded_data, force=force, verbosity=verbosity
        )

    def _get_weights(
        self,
        mdata: MData,
        r: np.ndarray,
        d: np.ndarray,
        sigma: np.ndarray,
        is_waked: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate analytical Gaussian rotor-disc weights.

        Parameters
        ----------
        mdata
            The model data. It is unused by the analytical evaluation.
        r
            Radial distance between wake and target rotor centres.
        d
            Target rotor diameters.
        sigma
            Gaussian wake widths.
        is_waked
            Mask selecting waked state-target points.

        Returns
        -------
        weights
            Rotor-disc averaged Gaussian weights.

        """
        return gaussian_disc_weight_analytical(
            r,
            d,
            sigma,
            is_waked=is_waked,
            masked_value=0.0,
            min_weight=self.min_weight,
        )
