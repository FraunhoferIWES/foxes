import numpy as np
from abc import abstractmethod

from foxes.core import PartialWakesModel, Data
from foxes.utils import wd2uv, uv2wd
import foxes.variables as FV
import foxes.constants as FC

from .distsliced import DistSlicedWakeModel

class WakeShapedPartials(PartialWakesModel):
    """
    Abstract class that makes use of wake shapes
    during point evaluation

    :group: models.partial_wakes

    """

    @abstractmethod
    def n_wpoints(self, algo):
        """
        The number of evaluation points per rotor

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
            
        Returns
        -------
        n: int
            The number of evaluation points per rotor
        
        """
        pass

    def _contr_generic(self, *args, **kwargs):
        """ Helper function for generic wake models """
        super().contribute_to_wake_deltas(*args, **kwargs)

    def _contr_dsliced(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        downwind_index,
        wake_deltas,
        wmodel,  
    ):
        """ Helper function for DistSlicedWakesModel """

        # evaluate grid rotor:
        n_states = fdata.n_states
        n_points = pdata.n_points
        n_wpoints = self.n_wpoints(algo)
        n_turbines = int(n_points/n_wpoints)
        wcoos = algo.wake_frame.get_wake_coos(
            algo, mdata, fdata, pdata, downwind_index
        ).reshape(n_states, n_turbines, n_wpoints, 3)
        x = wcoos[:, :, 0, 0]
        yz = wcoos[:, :, :, 1:3]
        del wcoos

        # reconstruct centre points:
        points = pdata[FC.POINTS].reshape(n_states, n_turbines, n_wpoints, 3)
        points = np.mean(points, axis=2)
        hpdata = Data.from_points(points=points)
        del points

        # evaluate wake model:
        wdeltas, sp_sel = wmodel.calc_wakes_spsel_x_yz(
            algo, mdata, fdata, hpdata, downwind_index, x, yz
        )

        wsps = np.zeros((n_states, n_turbines, n_wpoints), dtype=bool)
        wsps[:] = sp_sel[:, :, None]
        wsps = wsps.reshape(n_states, n_points)

        for v, wdel in wdeltas.items():
            d = np.zeros((n_states, n_turbines, n_wpoints), dtype=FC.DTYPE)
            d[sp_sel] = wdel
            d = d.reshape(n_states, n_points)[wsps]

            try:
                superp = wmodel.superp[v]
            except KeyError:
                raise KeyError(
                    f"Model '{self.name}': Missing wake superposition entry for variable '{v}' in wake model '{wmodel.name}', found {sorted(list(wmodel.superp.keys()))}"
                )

            wake_deltas[v] = superp.calc_wakes_plus_wake(
                algo,
                mdata,
                fdata,
                pdata,
                downwind_index,
                wsps,
                v,
                wake_deltas[v],
                d,
            )

    def contribute_to_wake_deltas(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        downwind_index,
        wake_deltas,
        wmodel,  
    ):
        """
        Modifies wake deltas by contributions from the
        specified wake source turbines.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        pdata: foxes.core.Data
            The evaluation point data
        downwind_index: int
            The index of the wake causing turbine
            in the downwnd order
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: numpy.ndarray with shape
            (n_states, n_points, ...)
        wmodel: foxes.core.WakeModel
            The wake model

        """

        if isinstance(wmodel, DistSlicedWakeModel):
            self._contr_dsliced(algo, mdata, fdata, pdata, downwind_index,
                                wake_deltas, wmodel)
        else:
            self._contr_generic(algo, mdata, fdata, pdata, downwind_index,
                                wake_deltas, wmodel)

    def evaluate_results(
        self,
        algo,
        mdata,
        fdata,
        wake_deltas,
        wmodel,
        downwind_index,
        amb_res=None,
    ):
        """
        Updates the farm data according to the wake
        deltas.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
            Modified in-place by this function
        wake_deltas: Any
            The wake deltas object at the selected downwind
            turbines
        wmodel: foxes.core.WakeModel
            The wake model
        downwind_index: int
            The index in the downwind order
        amb_res: dict, optional
            Ambient states results. Keys: var str, values:
            numpy.ndarray of shape (n_states, n_points)

        """
        rweights = algo.rotor_model.from_data_or_store(FC.RWEIGHTS, algo, mdata)
        rpoints = algo.rotor_model.from_data_or_store(FC.RPOINTS, algo, mdata)
        n_states, n_turbines, n_rpoints, __ = rpoints.shape

        amb_res_in = amb_res is not None
        if not amb_res_in:
            amb_res = algo.rotor_model.from_data_or_store(
                FC.AMB_RPOINT_RESULTS, algo, mdata
            )

        wweights = self.grotor.rotor_point_weights()
        n_wpoints = self.grotor.n_rotor_points()
        n_states, n_turbines, n_rpoints, __ = rpoints.shape

        uv = None
        if (FV.WS in amb_res and FV.WD not in amb_res) or (
            FV.WS not in amb_res and FV.WD in amb_res
        ):
            raise KeyError(
                f"Model '{self.name}': Missing one of the variables '{FV.WS}', '{FV.WD}' in ambient rotor results: {list(amb_res.keys())}"
            )

        elif FV.WD in amb_res and np.any(
            np.min(amb_res[FV.WD], axis=2) != np.max(amb_res[FV.WD], axis=2)
        ):
            wd = amb_res[FV.WD].reshape(n_states, n_turbines, n_rpoints)[:, downwind_index]
            ws = amb_res[FV.WS].reshape(n_states, n_turbines, n_rpoints)[:, downwind_index]
            uv = wd2uv(wd, ws, axis=-1)
            uv = np.einsum("spd,p->sd", uv, rweights)
            del ws, wd

        wres = {}
        for v, ares in amb_res.items():
            if v == FV.WS and uv is not None:
                wres[v] = np.linalg.norm(uv, axis=-1)
            elif v == FV.WD and uv is not None:
                wres[v] = uv2wd(uv, axis=-1)
            else:
                wres[v] = ares.reshape(n_states, n_turbines, n_rpoints)[:, downwind_index]
                wres[v] = np.einsum("sp,p->s", wres[v], rweights)
            wres[v] = wres[v][:, None]
        del uv

        wdel = {}
        for v, d in wake_deltas.items():
            wdel[v] = d.reshape(n_states, n_turbines, n_wpoints)
        for w in self.wake_models:
            w.finalize_wake_deltas(algo, mdata, fdata, pdata, wres, wdel)
        for v in wdel.keys():
            wdel[v] = np.einsum("sp,p->s", wdel[v], wweights)[:, None]

        for v in wres.keys():
            if v in wake_deltas:
                wres[v] += wdel[v]
                if amb_res_in:
                    amb_res[v][st_sel] = wres[v]
            wres[v] = wres[v][:, None]

        self.rotor_model.eval_rpoint_results(
            algo, mdata, fdata, wres, np.array([1.0]), states_turbine=states_turbine
        )
