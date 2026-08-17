from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, cast

from foxes.models.wake_models.top_hat import TopHatWakeModel
from foxes.utils.two_circles import calc_area
import foxes.variables as FV
import foxes.constants as FC

from .centre import PartialCentre

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import LoadedData, Model
    from foxes.core.wake_model import WakeModel
    from foxes.core.rotor_model import RotorModel


class PartialTopHat(PartialCentre):
    """
    Partial wakes for top-hat models.

    The wake effect is weighted by the overlap of
    the wake circle and the rotor disc circle.

    Attributes
    ----------
    rotor_model
        The rotor model, default is the one from the algorithm

    :group: models.partial_wakes

    """

    def check_wmodel(self, wmodel: WakeModel, error: bool = True) -> bool:
        """
        Checks the wake model type

        Parameters
        ----------
        wmodel
            The wake model to be tested
        error
            Flag for raising TypeError

        Returns
        -------
        chk
            True if wake model is compatible

        """
        if not isinstance(wmodel, TopHatWakeModel):
            if error:
                raise TypeError(
                    f"Partial wakes '{self.name}': Cannot be applied to wake model '{wmodel.name}', since not a TopHatWakeModel"
                )
            return False
        return True

    def __init__(self, rotor_model: RotorModel | None = None) -> None:
        """
        Constructor.

        Parameters
        ----------
        rotor_model
            The rotor model, default is the one from the algorithm

        """
        super().__init__()
        self.rotor_model = rotor_model

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

        Returns
        -------
        loaded_data
            The loaded data, containing keys "coords", "data_vars", and "extra_data".
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.

        """
        if self.rotor_model is None:
            self.rotor_model = algo.rotor_model

        loaded_data = super().initialize(
            algo, loaded_data=loaded_data, force=force, verbosity=verbosity
        )

        self.WCOOS_ID = self.var("WCOOS_ID")
        self.WCOOS_X = self.var("WCOOS_X")
        self.WCOOS_R = self.var("WCOOS_R")
        return loaded_data

    def sub_models(self) -> list[Model]:
        """
        List of all sub-models

        Returns
        -------
        smdls
            Names of all sub models

        """
        smdls = super().sub_models()
        if self.rotor_model is not None:
            smdls.append(self.rotor_model)
        return smdls

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
        Modifies wake deltas at target points by
        contributions from the specified wake source turbines.

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
        downwind_index
            The index of the wake causing turbine
            in the downwind order
        wake_deltas
            The wake deltas. Key: variable name,
            value
            (n_states, n_targets, n_tpoints, ...)

        """
        self.check_wmodel(wmodel, error=True)
        wmodel = cast(TopHatWakeModel, wmodel)

        wcoos = algo.wake_frame.get_wake_coos(algo, mdata, fdata, tdata, downwind_index)
        x = wcoos[:, :, 0, 0]
        yz = wcoos[:, :, 0, 1:3]
        del wcoos

        ct = self.get_data(
            FV.CT,
            FC.STATE_TARGET,
            lookup="w",
            fdata=fdata,
            tdata=tdata,
            downwind_index=downwind_index,
            algo=algo,
            upcast=True,
        )

        sel0 = (ct > 1e-8) & (x > 1e-8)
        if np.any(sel0):
            R = np.linalg.norm(yz, axis=-1)
            del yz

            D = self.get_data(
                FV.D,
                FC.STATE_TARGET,
                lookup="w",
                fdata=fdata,
                tdata=tdata,
                downwind_index=downwind_index,
                algo=algo,
                upcast=True,
            )

            wr = wmodel.calc_wake_radius(
                algo, mdata, fdata, tdata, downwind_index, x, ct
            )

            st_sel = sel0 & (wr > R - D / 2)
            if np.any(st_sel):
                x = x[st_sel]
                ct = ct[st_sel]
                wr = wr[st_sel]
                R = R[st_sel]
                D = D[st_sel]

                clw = wmodel.calc_centreline(
                    algo, mdata, fdata, tdata, downwind_index, st_sel, x, wr, ct
                )

                weights = calc_area(D / 2, wr, R) / (np.pi * (D / 2) ** 2)

                # run superposition models:
                if wmodel.affects_ws and wmodel.has_uv:
                    assert wmodel.has_vector_wind_superp, (
                        f"{self.name}: Expecting vector wind superposition in wake model '{wmodel.name}', got '{wmodel.wind_superposition}'"
                    )
                    vec_superp = wmodel.vec_superp
                    assert vec_superp is not None
                    if FV.UV in clw:
                        duv = clw.pop(FV.UV)
                    else:
                        clwe = {v: d[:, None] for v, d in clw.items()}
                        vec_superp.wdeltas_ws2uv(
                            algo, fdata, tdata, downwind_index, clwe, st_sel
                        )
                        duv = np.einsum("sd,s->sd", clwe.pop(FV.UV)[:, 0], weights)
                        del clwe, clw[FV.WS]
                        if FV.WD in clw:
                            del clw[FV.WD]
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

                for v, d in clw.items():
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
                        weights[:, None] * d[:, None],
                    )
