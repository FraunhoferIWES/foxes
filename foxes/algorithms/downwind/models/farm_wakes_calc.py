import numpy as np
from copy import deepcopy

from foxes.core import FarmDataModel, TData


class FarmWakesCalculation(FarmDataModel):
    """
    This model calculates wakes effects on farm data.

    :group: algorithms.downwind.models

    """

    def output_farm_vars(self, algo):
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
        ovars = algo.rotor_model.output_farm_vars(
            algo
        ) + algo.farm_controller.output_farm_vars(algo)
        return list(dict.fromkeys(ovars))

    def calculate(self, algo, mdata, fdata):
        """ "
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        # collect ambient rotor results and weights:
        rotor = algo.rotor_model
        weights = rotor.from_data_or_store(rotor.RWEIGHTS, algo, mdata)
        amb_res = rotor.from_data_or_store(rotor.AMBRES, algo, mdata)

        # generate all wake evaluation points
        # (n_states, n_order, n_rpoints)
        pwake2tdata = {}
        for wname, wmodel in algo.wake_models.items():
            pwake = algo.partial_wakes[wname]
            if pwake.name not in pwake2tdata:
                tpoints, tweights = pwake.get_wake_points(algo, mdata, fdata)
                pwake2tdata[pwake.name] = TData.from_tpoints(tpoints, tweights)

        def _get_wdata(tdatap, wdeltas, s):
            """Helper function for wake data extraction"""
            tdata = tdatap.get_slice(s, keep=True)
            wdelta = {v: d[s] for v, d in wdeltas.items()}
            return tdata, wdelta

        def _evaluate(
            gmodel, tdata, amb_res, weights, wake_res, wdeltas, oi, wmodel, pwake
        ):
            """Helper function for data evaluation at turbines"""
            wres = gmodel.finalize_farm_wakes(
                algo, mdata, fdata, tdata, amb_res, weights, wdeltas, wmodel, oi, pwake
            )

            hres = {v: d[:, oi, None] for v, d in wake_res.items()}
            for v, d in wres.items():
                if v in wake_res:
                    hres[v] += d[:, None]

            rotor.eval_rpoint_results(
                algo, mdata, fdata, hres, weights, downwind_index=oi
            )

            res = algo.farm_controller.calculate(
                algo, mdata, fdata, pre_rotor=False, downwind_index=oi
            )
            fdata.update(res)

        wake_res = deepcopy(amb_res)
        n_turbines = mdata.n_turbines
        run_up = None
        run_down = None
        for wname, wmodel in algo.wake_models.items():
            pwake = algo.partial_wakes[wname]
            gmodel = algo.ground_models[wname]
            tdatap = pwake2tdata[pwake.name]
            wdeltas = gmodel.new_farm_wake_deltas(
                algo, mdata, fdata, tdatap, wmodel, pwake
            )

            # downwind:
            if wmodel.affects_downwind:
                run_up = wname
                for oi in range(n_turbines):
                    if oi > 0:
                        _evaluate(
                            gmodel,
                            tdatap,
                            amb_res,
                            weights,
                            wake_res,
                            wdeltas,
                            oi,
                            wmodel,
                            pwake,
                        )

                    if oi < n_turbines - 1:
                        tdata, wdelta = _get_wdata(tdatap, wdeltas, np.s_[:, oi + 1 :])
                        gmodel.contribute_to_farm_wakes(
                            algo, mdata, fdata, tdata, oi, wdelta, wmodel, pwake
                        )

            # upwind:
            else:
                run_down = wname
                for oi in range(n_turbines - 1, -1, -1):
                    if oi < n_turbines - 1:
                        _evaluate(
                            gmodel,
                            tdatap,
                            amb_res,
                            weights,
                            wake_res,
                            wdeltas,
                            oi,
                            wmodel,
                            pwake,
                        )

                    if oi > 0:
                        tdata, wdelta = _get_wdata(tdatap, wdeltas, np.s_[:, :oi])
                        gmodel.contribute_to_farm_wakes(
                            algo, mdata, fdata, tdata, oi, wdelta, wmodel, pwake
                        )

            if run_up is not None and run_down is not None:
                raise KeyError(
                    f"Wake model '{run_up}' is an upwind model, wake model '{run_down}' is a downwind model: Require iterative algorithm"
                )

        return {v: fdata[v] for v in self.output_farm_vars(algo)}
