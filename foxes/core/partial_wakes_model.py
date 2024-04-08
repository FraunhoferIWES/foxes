from abc import abstractmethod

from foxes.utils import all_subclasses

from .model import Model


class PartialWakesModel(Model):
    """
    Abstract base class for partial wakes models.

    Partial wakes models compute wake effects
    for rotor effective quantities.

    Attributes
    ----------
    wake_models: list of foxes.core.WakeModel
        The wake model selection
    wake_frame: foxes.core.WakeFrame, optional
        The wake frame

    :group: core

    """

    @abstractmethod
    def get_wake_points(self, algo, mdata, fdata):
        """
        Get the wake calculation points.

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
        rpoints: numpy.ndarray
            All rotor points, shape: (n_states, n_targets, n_rpoints, 3)

        """
        pass

    def new_wake_deltas(self, algo, mdata, fdata, wmodel, wpoints):
        """
        Creates new initial wake deltas, filled
        with zeros.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        wmodel: foxes.core.WakeModel
            The wake model
        wpoints: numpy.ndarray
            The wake evaluation points,
            shape: (n_states, n_turbines, n_rpoints, 3)

        Returns
        -------
        wake_deltas: dict
            Key: variable name, value: The zero filled 
            wake deltas, shape: (n_states, n_turbines, n_rpoints, ...)

        """
        return wmodel.new_wake_deltas(algo, mdata, fdata, wpoints)

    def contribute_at_rotors(
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
        Modifies wake deltas at rotor points by 
        contributions from the specified wake source turbines.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.Data
            The model data
        fdata: foxes.core.Data
            The farm data
        pdata: foxes.core.Data
            The evaluation point data at rotor points
        downwind_index: int
            The index of the wake causing turbine
            in the downwnd order
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: numpy.ndarray with shape
            (n_states, n_rotors, n_rpoints, ...)
        wmodel: foxes.core.WakeModel
            The wake model

        """
        wcoos = algo.wake_frame.wake_coos_at_rotors(
            algo, mdata, fdata, pdata, downwind_index
        )
        wmodel.contribute_at_rotors(
            algo, mdata, fdata, pdata, downwind_index, 
            wcoos, wake_deltas
        )

    def contribute_at_points(
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
        Modifies wake deltas at given points by 
        contributions from the specified wake source turbines.

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
        wcoos = algo.wake_frame.wake_coos_at_points(
            algo, mdata, fdata, pdata, downwind_index
        )
        wmodel.contribute_at_points(
            algo, mdata, fdata, pdata, downwind_index, 
            wcoos, wake_deltas
        )

    @abstractmethod
    def evaluate_results(
        self,
        algo,
        mdata,
        fdata,
        wake_deltas,
        wmodel,
        downwind_index,
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
        wake_deltas: dict
            The wake deltas object at the selected downwind
            turbines. Key: variable str, value: numpy.ndarray
            with shape (n_states, n_rpoints, ...)
        wmodel: foxes.core.WakeModel
            The wake model
        downwind_index: int
            The index in the downwind order

        """
        pass

    @classmethod
    def new(cls, pwake_type, **kwargs):
        """
        Run-time partial wakes factory.

        Parameters
        ----------
        pwake_type: str
            The selected derived class name

        """

        if pwake_type is None:
            return None

        allc = all_subclasses(cls)
        found = pwake_type in [scls.__name__ for scls in allc]

        if found:
            for scls in allc:
                if scls.__name__ == pwake_type:
                    return scls(**kwargs)

        else:
            estr = "Partial wakes model type '{}' is not defined, available types are \n {}".format(
                pwake_type, sorted([i.__name__ for i in allc])
            )
            raise KeyError(estr)
