import numpy as np

from foxes.config import config
from foxes.core import TurbineInductionModel
import foxes.variables as FV
import foxes.constants as FC


class VortexSheet(TurbineInductionModel):
    """
    The Vortex Sheet model implemented with a radial dependency

    Notes
    -----
    Reference:
    Medici, D., et al. "The upstream flow of a wind turbine: blockage effect." Wind Energy 14.5 (2011): 691-697.
    https://doi.org/10.1002/we.451

    Attributes
    ----------
    pre_rotor_only: bool
        Calculate only the pre-rotor region
    induction: foxes.core.AxialInductionModel or str
        The induction model

    :group: models.wake_models.induction

    """

    def __init__(
        self,
        superposition="ws_linear",
        induction="Madsen",
        pre_rotor_only=False,
    ):
        """
        Constructor.

        Parameters
        ----------
        superposition: str
            The wind speed superposition
        induction: foxes.core.AxialInductionModel or str
            The induction model
        pre_rotor_only: bool
            Calculate only the pre-rotor region

        """
        super().__init__(wind_superposition=superposition)
        self.induction = induction
        self.pre_rotor_only = pre_rotor_only

    def __repr__(self):
        iname = (
            self.induction if isinstance(self.induction, str) else self.induction.name
        )
        return f"{type(self).__name__}({self.wind_superposition}, induction={iname})"

    @property
    def affects_ws(self):
        """
        Flag for wind speed wake models

        Returns
        -------
        dws: bool
            If True, this model affects wind speed

        """
        return True
    
    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            All sub models

        """
        return super().sub_models() + [self.induction]

    def initialize(self, algo, verbosity=0, force=False):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent
        force: bool
            Overwrite existing data

        """
        if isinstance(self.induction, str):
            self.induction = algo.mbook.axial_induction[self.induction]
        super().initialize(algo, verbosity, force)

    def new_wake_deltas(self, algo, mdata, fdata, tdata):
        """
        Creates new empty wake delta arrays.

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
        wake_deltas: dict
            Key: variable name, value: The zero filled
            wake deltas, shape: (n_states, n_targets, n_tpoints, ...)

        """
        if self.has_uv:
            duv = np.zeros(
                (tdata.n_states, tdata.n_targets, tdata.n_tpoints, 2), 
                dtype=config.dtype_double,
            )
            return {FV.UV: duv}
        else:
            dws = np.zeros(
                (tdata.n_states, tdata.n_targets, tdata.n_tpoints), 
                dtype=config.dtype_double,
            )
            return {FV.WS: dws}

    def contribute(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        wake_coos,
        wake_deltas,
    ):
        """
        Modifies wake deltas at target points by
        contributions from the specified wake source turbines.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        tdata: foxes.core.Data
            The target point data
        downwind_index: int
            The index of the wake causing turbine
            in the downwind order
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints, ...)

        """

        # get x, y and z. Rounding for safe x < 0 condition
        x = np.round(wake_coos[..., 0], 12)
        y = wake_coos[..., 1]
        z = wake_coos[..., 2]
        r = np.sqrt(y**2 + z**2)
        r_sph = np.sqrt(r**2 + x**2)

        # get ct
        ct = self.get_data(
            FV.CT,
            FC.STATE_TARGET_TPOINT,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            upcast=True,
            downwind_index=downwind_index,
        )

        # get D
        D = self.get_data(
            FV.D,
            FC.STATE_TARGET_TPOINT,
            lookup="w",
            algo=algo,
            fdata=fdata,
            tdata=tdata,
            upcast=True,
            downwind_index=downwind_index,
        )

        sp_sel = (ct > 1e-8) & (x <= 0)
        ct_sel = ct[sp_sel]
        r_sph_sel = r_sph[sp_sel]
        R_sel = D[sp_sel] / 2
        xi = r_sph_sel / R_sel

        def add_wake(sp_sel, wake_deltas, blockage):
            """adds to wake deltas"""
            if self.has_vector_wind_superp:
                wdeltas = {FV.WS: blockage}
                self.vec_superp.wdeltas_ws2uv(algo, fdata, tdata, downwind_index, wdeltas, sp_sel)
                wake_deltas[FV.UV] = self.vec_superp.add_wake_vector(
                    algo,
                    mdata, 
                    fdata, 
                    tdata, 
                    downwind_index, 
                    sp_sel,
                    wake_deltas[FV.UV],
                    wdeltas.pop(FV.UV),
                )
            else:
                self.superp[FV.WS].add_wake(
                    algo,
                    mdata,
                    fdata,
                    tdata,
                    downwind_index,
                    sp_sel,
                    FV.WS,
                    wake_deltas[FV.WS],
                    blockage,
                )    

        if np.any(sp_sel):
            blockage = self.induction.ct2a(ct_sel) * (1 + -xi / np.sqrt(1 + xi**2))
            add_wake(sp_sel, wake_deltas, -blockage)

        if not self.pre_rotor_only:
            sp_sel = (
                (ct > 1e-8) & (x > 0) & (r > D / 2)
            )  # mirror in rotor plane and inverse blockage, but not directly behind rotor
            ct_sel = ct[sp_sel]
            r_sph_sel = r_sph[sp_sel]
            R_sel = D[sp_sel] / 2
            xi = r_sph_sel / R_sel
            if np.any(sp_sel):
                blockage = self.induction.ct2a(ct_sel) * (1 + -xi / np.sqrt(1 + xi**2))
                add_wake(sp_sel, wake_deltas, blockage)

        return wake_deltas
