from abc import abstractmethod

from foxes.core import WakeModel


class DistSlicedWakeModel(WakeModel):
    """
    Abstract base class for wake models for which
    the x-denpendency can be separated from the
    yz-dependency.

    The multi-yz ability is used by the `PartialDistSlicedWake`
    partial wakes model.

    Attributes
    ----------
    superpositions: dict
        The superpositions. Key: variable name str,
        value: The wake superposition model name,
        will be looked up in model book
    superp: dict
        The superposition dict, key: variable name str,
        value: `foxes.core.WakeSuperposition`

    :group: models.wake_models

    """

    def __init__(self, superpositions):
        """
        Constructor.

        Parameters
        ----------
        superpositions: dict
            The superpositions. Key: variable name str,
            value: The wake superposition model name,
            will be looked up in model book

        """
        super().__init__()
        self.superpositions = superpositions

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return list(self.superp.values())

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        This includes loading all required data from files. The model
        should return all array type data as part of the idata return
        dictionary (and not store it under self, for memory reasons). This
        data will then be chunked and provided as part of the mdata object
        during calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        idata: dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """
        self.superp = {
            v: algo.mbook.wake_superpositions[s] for v, s in self.superpositions.items()
        }
        super().initialize(algo, verbosity)

    @abstractmethod
    def calc_wakes_spsel_x_yz(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        states_source_turbine,
        x,
        yz,
    ):
        """
        Calculate wake deltas.

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
        states_source_turbine: numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        x: numpy.ndarray
            The x values, shape: (n_states, n_points)
        yz: numpy.ndarray
            The yz values for each x value, shape:
            (n_states, n_points, n_yz_per_x, 2)

        Returns
        -------
        wdeltas: dict
            The wake deltas. Key: variable name str,
            value: numpy.ndarray, shape: (n_sp_sel, n_yz_per_x)
        sp_sel: numpy.ndarray of bool
            The state-point selection, for which the wake
            is non-zero, shape: (n_states, n_points)

        """
        pass

    def contribute_to_wake_deltas(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        states_source_turbine,
        wake_coos,
        wake_deltas,
    ):
        """
        Calculate the contribution to the wake deltas
        by this wake model.

        Modifies wake_deltas on the fly.

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
        states_source_turbine: numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        wake_coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_points, 3)
        wake_deltas: dict
            The wake deltas, are being modified ob the fly.
            Key: Variable name str, for which the
            wake delta applies, values: numpy.ndarray with
            shape (n_states, n_points, ...)

        """
        x = wake_coos[:, :, 0]
        yz = wake_coos[:, :, None, 1:3]

        wdeltas, sp_sel = self.calc_wakes_spsel_x_yz(
            algo, mdata, fdata, pdata, states_source_turbine, x, yz
        )

        for v, hdel in wdeltas.items():
            try:
                superp = self.superp[v]
            except KeyError:
                raise KeyError(
                    f"Model '{self.name}': Missing wake superposition entry for variable '{v}', found {sorted(list(self.superp.keys()))}"
                )

            wake_deltas[v] = superp.calc_wakes_plus_wake(
                algo,
                mdata,
                fdata,
                pdata,
                states_source_turbine,
                sp_sel,
                v,
                wake_deltas[v],
                hdel[:, 0],
            )

    def finalize_wake_deltas(
        self,
        algo,
        mdata,
        fdata,
        pdata,
        amb_results,
        wake_deltas,
    ):
        """
        Finalize the wake calculation.

        Modifies wake_deltas on the fly.

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
        amb_results: dict
            The ambient results, key: variable name str,
            values: numpy.ndarray with shape (n_states, n_points)
        wake_deltas: dict
            The wake deltas, are being modified ob the fly.
            Key: Variable name str, for which the wake delta
            applies, values: numpy.ndarray with shape
            (n_states, n_points, ...) before evaluation,
            numpy.ndarray with shape (n_states, n_points) afterwards

        """
        for v, s in self.superp.items():
            wake_deltas[v] = s.calc_final_wake_delta(
                algo, mdata, fdata, pdata, v, amb_results[v], wake_deltas[v]
            )
