import numpy as np

from foxes.models.wake_models.top_hat import TopHatWakeModel
from foxes.utils.two_circles import calc_area
import foxes.variables as FV
import foxes.constants as FC

from .centre import PartialCentre


class PartialTopHat(PartialCentre):
    """
    Partial wakes for top-hat models.

    The wake effect is weighted by the overlap of
    the wake circle and the rotor disc circle.

    Attributes
    ----------
    rotor_model: foxes.core.RotorModel
        The rotor model, default is the one from the algorithm

    :group: models.partial_wakes

    """

    def check_wmodel(self, wmodel, error=True):
        """
        Checks the wake model type

        Parameters
        ----------
        wmodel: foxes.core.WakeModel
            The wake model to be tested
        error: bool
            Flag for raising TypeError

        Returns
        -------
        chk: bool
            True if wake model is compatible

        """
        if not isinstance(wmodel, TopHatWakeModel):
            if error:
                raise TypeError(
                    f"Partial wakes '{self.name}': Cannot be applied to wake model '{wmodel.name}', since not a TopHatWakeModel"
                )
            return False
        return True

    def __init__(self, rotor_model=None):
        """
        Constructor.

        Parameters
        ----------
        rotor_model: foxes.core.RotorModel, optional
            The rotor model, default is the one from the algorithm

        """
        super().__init__()
        self.rotor_model = rotor_model

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        """
        if self.rotor_model is None:
            self.rotor_model = algo.rotor_model

        super().initialize(algo, verbosity)

        self.WCOOS_ID = self.var("WCOOS_ID")
        self.WCOOS_X = self.var("WCOOS_X")
        self.WCOOS_R = self.var("WCOOS_R")

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return super().sub_models() + [self.rotor_model]

    def contribute(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        wake_deltas,
        wmodel,
    ):
        """
        Modifies wake deltas at target points by
        contributions from the specified wake source turbines.

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
        downwind_index: int
            The index of the wake causing turbine
            in the downwind order
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints, ...)

        """
        self.check_wmodel(wmodel, error=True)

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
                    if FV.UV in clw:
                        duv = clw.pop(FV.UV)
                    else:
                        clwe = {v: d[:, None] for v, d in clw.items()}
                        wmodel.vec_superp.wdeltas_ws2uv(
                            algo, fdata, tdata, downwind_index, clwe, st_sel
                        )
                        duv = np.einsum("sd,s->sd", clwe.pop(FV.UV)[:, 0], weights)
                        del clwe, clw[FV.WS]
                        if FV.WD in clw:
                            del clw[FV.WD]
                    wake_deltas[FV.UV] = wmodel.vec_superp.add_wake_vector(
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
