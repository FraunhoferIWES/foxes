from abc import abstractmethod

from foxes.core import WakeModel


class DistSlicedWakeModel(WakeModel):
    """
    Abstract base class for wake models for which
    the x-denpendency can be separated from the
    yz-dependency.

    The multi-yz ability is used by the `PartialDistSlicedWake`
    partial wakes model.

    Parameters
    ----------
    superpositions : dict
        The superpositions. Key: variable name str,
        value: The wake superposition model name,
        will be looked up in model book

    Attributes
    ----------
    superp : dict
        The superposition dict, key: variable name str,
        value: `foxes.core.WakeSuperposition`

    """

    def __init__(self, superpositions):
        super().__init__()
        self.superp = superpositions

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        verbosity : int
            The verbosity level

        """
        super().initialize(algo, verbosity=verbosity)

        self.superp = {
            v: algo.mbook.wake_superpositions[s] for v, s in self.superp.items()
        }
        for v, s in self.superp.items():
            if not s.initialized:
                s.initialize(algo, verbosity=verbosity)

    @abstractmethod
    def calc_wakes_spsel_x_yz(self, algo, mdata, fdata, states_source_turbine, x, yz):
        """
        Calculate wake deltas.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        states_source_turbine : numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        x : numpy.ndarray
            The x values, shape: (n_states, n_points)
        yz : numpy.ndarray
            The yz values for each x value, shape:
            (n_states, n_points, n_yz_per_x, 2)

        Returns
        -------
        wdeltas : dict
            The wake deltas. Key: variable name str,
            value: numpy.ndarray, shape: (n_sp_sel, n_yz_per_x)
        sp_sel : numpy.ndarray of bool
            The state-point selection, for which the wake
            is non-zero, shape: (n_states, n_points)

        """
        pass

    def contribute_to_wake_deltas(
        self, algo, mdata, fdata, states_source_turbine, wake_coos, wake_deltas
    ):
        """
        Calculate the contribution to the wake deltas
        by this wake model.

        Modifies wake_deltas on the fly.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        states_source_turbine : numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        wake_deltas : dict
            The wake deltas, are being modified ob the fly.
            Key: Variable name str, for which the
            wake delta applies, values: numpy.ndarray with
            shape (n_states, n_points, ...)

        """
        x = wake_coos[:, :, 0]
        yz = wake_coos[:, :, None, 1:3]

        wdeltas, sp_sel = self.calc_wakes_spsel_x_yz(
            algo, mdata, fdata, states_source_turbine, x, yz
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
                states_source_turbine,
                sp_sel,
                v,
                wake_deltas[v],
                hdel[:, 0],
            )

    def finalize_wake_deltas(self, algo, mdata, fdata, amb_results, wake_deltas):
        """
        Finalize the wake calculation.

        Modifies wake_deltas on the fly.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        amb_results : dict
            The ambient results, key: variable name str,
            values: numpy.ndarray with shape (n_states, n_points)
        wake_deltas : dict
            The wake deltas, are being modified ob the fly.
            Key: Variable name str, for which the wake delta
            applies, values: numpy.ndarray with shape
            (n_states, n_points, ...) before evaluation,
            numpy.ndarray with shape (n_states, n_points) afterwards

        """
        for v, s in self.superp.items():
            wake_deltas[v] = s.calc_final_wake_delta(
                algo, mdata, fdata, v, amb_results[v], wake_deltas[v]
            )
