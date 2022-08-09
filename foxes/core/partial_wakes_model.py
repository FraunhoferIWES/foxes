from abc import abstractmethod

from .model import Model
from foxes.utils import all_subclasses


class PartialWakesModel(Model):
    """
    Abstract base class for partial wakes models.

    Partial wakes models compute wake effects
    for rotor effective quantities.

    Parameters
    ----------
    wake_models : list of foxes.core.WakeModel
        The wake model selection, None for all
        from algorithm.
    wake_frame : foxes.core.WakeFrame, optional
        The wake frame, None takes from algorithm

    Attributes
    ----------
    wake_models : list of foxes.core.WakeModel
        The wake model selection
    wake_frame : foxes.core.WakeFrame, optional
        The wake frame

    """

    def __init__(self, wake_models=None, wake_frame=None):
        super().__init__()

        self.wake_models = wake_models
        self.wake_frame = wake_frame

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
        if self.wake_models is None:
            self.wake_models = algo.wake_models
        if self.wake_frame is None:
            self.wake_frame = algo.wake_frame

        if not self.wake_frame.initialized:
            self.wake_frame.initialize(algo, verbosity=verbosity)
        for w in self.wake_models:
            if not w.initialized:
                w.initialize(algo, verbosity=verbosity)

        super().initialize(algo, verbosity=verbosity)

    @abstractmethod
    def new_wake_deltas(self, algo, mdata, fdata):
        """
        Creates new initial wake deltas, filled
        with zeros.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data

        Returns
        -------
        wake_deltas : dict
            Keys: Variable name str, values: any

        """
        pass

    @abstractmethod
    def contribute_to_wake_deltas(
        self, algo, mdata, fdata, states_source_turbine, wake_deltas
    ):
        """
        Modifies wake deltas by contributions from the
        specified wake source turbines.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        states_source_turbine : numpy.ndarray of int
            For each state, one turbine index corresponding
            to the wake causing turbine. Shape: (n_states,)
        wake_deltas : Any
            The wake deltas object created by the
            `new_wake_deltas` function

        """
        pass

    @abstractmethod
    def evaluate_results(
        self, algo, mdata, fdata, wake_deltas, states_turbine, update_amb_res=False
    ):
        """
        Updates the farm data according to the wake
        deltas.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
            Modified in-place by this function
        wake_deltas : Any
            The wake deltas object, created by the
            `new_wake_deltas` function and filled
            by `contribute_to_wake_deltas`
        states_turbine : numpy.ndarray of int
            For each state, the index of one turbine
            for which to evaluate the wake deltas.
            Shape: (n_states,)
        update_amb_res : bool
            Flag for updating ambient results

        """
        pass

    @classmethod
    def new(cls, pwake_type, **kwargs):
        """
        Run-time partial wakes factory.

        Parameters
        ----------
        pwake_type : str
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
