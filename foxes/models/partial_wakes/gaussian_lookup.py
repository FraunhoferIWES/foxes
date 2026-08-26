from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import xarray as xr

import foxes.constants as FC
import foxes.variables as FV
from foxes.data import MODEL_DATA
from foxes.models.wake_models.gaussian import GaussianWakeModel
from foxes.utils.gaussian_lookup import evaluate_lookup_geometry
from foxes.utils.gaussian_lookup import load_lookup_dataset
from foxes.utils.gaussian_lookup import validate_lookup_dataset

from .centre import PartialCentre

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import LoadedData
    from foxes.core.wake_model import WakeModel


class PartialGaussianLookup(PartialCentre):
    """
    Gaussian-only partial wake model using lookup-artifact rotor weights.

    Attributes
    ----------
    lookup_data
        Lookup dataset source, either loaded dataset or NetCDF file path.
    bounds_policy
        Out-of-range behavior for lookup queries. Default is ``"clip"``.
    min_weight
        Minimal retained lookup weight. Lower values are zeroed.
    lookup_dataset
        Loaded and validated lookup dataset.
    """

    def __init__(
        self,
        lookup_data: xr.Dataset | str | Path | None = None,
        bounds_policy: Literal["clip", "nan", "raise"] = "clip",
        min_weight: float = 1.0e-8,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        lookup_data
            Lookup dataset source, either in-memory xarray dataset or path to
            a NetCDF artifact. If ``None``, the packaged
            ``model_data/gaussian_lookup.nc`` artifact is loaded through the
            algorithm data book during initialization.
        bounds_policy
            Out-of-range behavior for lookup queries. Choices are
            ``"clip"`` (default), ``"nan"``, and ``"raise"``.
            For ``"clip"``, out-of-range points are clipped to table bounds;
            if resulting clipped weights are larger than ``min_weight``,
            ``ValueError`` is raised. Sigma values above the generated upper
            sigma bound use the large-sigma asymptote.
        min_weight
            Minimal retained lookup weight. Interpolated weights below this
            value are set to zero. This runtime value may override the
            threshold used to generate an externally supplied artifact.
        """
        super().__init__()
        if bounds_policy not in {"clip", "nan", "raise"}:
            raise ValueError("bounds_policy must be one of {'clip', 'nan', 'raise'}")
        if min_weight < 0.0:
            raise ValueError("min_weight must be >= 0")
        self.lookup_data = lookup_data
        self.bounds_policy = bounds_policy
        self.min_weight = min_weight
        self.lookup_dataset: xr.Dataset | None = None

    def __repr__(self) -> str:
        src = type(self.lookup_data).__name__ if self.lookup_data is not None else "None"
        return (
            f"{type(self).__name__}(lookup_data={src}, "
            f"bounds_policy={self.bounds_policy}, min_weight={self.min_weight})"
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

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """
        Initialize model and resolve lookup artifact input.

        Parameters
        ----------
        algo
            The calculation algorithm.
        loaded_data
            Already loaded data to extend.
        force
            Overwrite existing data.
        verbosity
            Verbosity level.

        Notes
        -----
        When ``lookup_data`` is ``None``, the bundled
        ``model_data/gaussian_lookup.nc`` artifact is resolved through
        ``algo.dbook``.

        Returns
        -------
        loaded_data
            Extended loaded data.
        """
        loaded_data = super().initialize(
            algo,
            loaded_data=loaded_data,
            force=force,
            verbosity=verbosity,
        )

        if isinstance(self.lookup_data, xr.Dataset):
            validate_lookup_dataset(self.lookup_data)
            self.lookup_dataset = self.lookup_data.copy(deep=True)
        elif isinstance(self.lookup_data, (str, Path)):
            self.lookup_dataset = load_lookup_dataset(self.lookup_data)
        elif self.lookup_data is None:
            lookup_path = algo.dbook.get_file_path(
                MODEL_DATA, "gaussian_lookup.nc", errors=False
            )
            if lookup_path is None:
                raise FileNotFoundError(
                    "Default Gaussian lookup artifact 'gaussian_lookup.nc' was "
                    f"not found in data-book context '{MODEL_DATA}'"
                )
            self.lookup_dataset = load_lookup_dataset(lookup_path)
        else:
            raise TypeError(
                "lookup_data must be xarray.Dataset, str, pathlib.Path, or None"
            )

        return loaded_data

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
        Apply wake contributions using lookup-based rotor weights.
        """
        self.check_wmodel(wmodel, error=True)
        wmodel = cast(GaussianWakeModel, wmodel)

        if self.lookup_dataset is None:
            raise ValueError(
                f"Partial wakes '{self.name}': Missing lookup dataset. "
                "Provide 'lookup_data' as Dataset or NetCDF path."
            )

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
            weights_full = evaluate_lookup_geometry(
                self.lookup_dataset,
                r=R,
                d=D,
                sigma=sigma_full,
                is_waked=st_sel,
                bounds_policy=self.bounds_policy,
                masked_value=0.0,
                min_weight=self.min_weight,
                clip_check_min_weight=(
                    self.min_weight if self.bounds_policy == "clip" else None
                ),
            )

            wdeltas[v] = ampld * weights_full[st_sel]

        # apply wake-deflection auxiliary effects for WS, consistent with GaussianWakeModel
        if wmodel.affects_ws and FV.WS in wdeltas:
            if FC.WDEFL_ROT_ANGLE in tdata:
                dwd_defl = tdata.pop(FC.WDEFL_ROT_ANGLE)
                if FV.WD not in wdeltas:
                    wdeltas[FV.WD] = np.zeros_like(wdeltas[FV.WS])
                    wdeltas[FV.WD][:] = dwd_defl[st_sel]
                else:
                    wdeltas[FV.WD] += dwd_defl[st_sel]

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
