from abc import abstractmethod

from foxes.utils import new_instance

from .model import Model


class WakeSuperposition(Model):
    """
    Abstract base class for wake superposition models.

    Note that it is a matter of the wake model
    if superposition models are used, or if the
    wake model computes the total wake result by
    other means.

    :group: core

    """

    @abstractmethod
    def add_wake(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        st_sel,
        variable,
        wake_delta,
        wake_model_result,
    ):
        """
        Add a wake delta to previous wake deltas,
        at rotor points.

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
        st_sel: numpy.ndarray of bool
            The selection of targets, shape: (n_states, n_targets)
        variable: str
            The variable name for which the wake deltas applies
        wake_delta: numpy.ndarray
            The original wake deltas, shape:
            (n_states, n_targets, n_tpoints, ...)
        wake_model_result: numpy.ndarray
            The new wake deltas of the selected rotors,
            shape: (n_st_sel, n_tpoints, ...)

        Returns
        -------
        wdelta: numpy.ndarray
            The updated wake deltas, shape:
            (n_states, n_targets, n_tpoints, ...)

        """
        pass

    @abstractmethod
    def calc_final_wake_delta(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        variable,
        wake_delta,
    ):
        """
        Calculate the final wake delta after adding all
        contributions.

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
        variable: str
            The variable name for which the wake deltas applies
        wake_delta: numpy.ndarray
            The wake deltas at targets, shape:
            (n_states, n_targets, n_tpoints)

        Returns
        -------
        final_wake_delta: numpy.ndarray
            The final wake delta, which will be added to the ambient
            results by simple plus operation. Shape:
            (n_states, n_targets, n_tpoints)

        """
        pass

    @classmethod
    def new(cls, superp_type, *args, **kwargs):
        """
        Run-time wake superposition model factory.

        Parameters
        ----------
        superp_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for constructor
        kwargs: dict, optional
            Additional parameters for constructor

        """
        return new_instance(cls, superp_type, *args, **kwargs)


class WindVectorWakeSuperposition(Model):
    """
    Base class for wind vector superposition.

    Note that it is a matter of the wake model
    if superposition models are used, or if the
    wake model computes the total wake result by
    other means.

    :group: core

    """

    @abstractmethod
    def add_wake_vector(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        downwind_index,
        st_sel,
        wake_delta_uv,
        wake_model_result_uv,
    ):
        """
        Add a wake delta vector to previous wake deltas,
        at rotor points.

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
        st_sel: numpy.ndarray of bool
            The selection of targets, shape: (n_states, n_targets)
        wake_delta_uv: numpy.ndarray
            The original wind vector wake deltas, shape:
            (n_states, n_targets, n_tpoints, 2)
        wake_model_result_uv: numpy.ndarray
            The new wind vector wake deltas of the selected rotors,
            shape: (n_st_sel, n_tpoints, 2, ...)

        Returns
        -------
        wdelta_uv: numpy.ndarray
            The updated wind vector wake deltas, shape:
            (n_states, n_targets, n_tpoints, ...)

        """
        pass

    @abstractmethod
    def calc_final_wake_delta_uv(
        self,
        algo,
        mdata,
        fdata,
        tdata,
        wake_delta_uv,
    ):
        """
        Calculate the final wind vector wake delta after adding all
        contributions.

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
        wake_delta_uv: numpy.ndarray
            The original wind vector wake deltas, shape:
            (n_states, n_targets, n_tpoints, 2)

        Returns
        -------
        final_wake_delta_ws: numpy.ndarray
            The final wind speed wake delta, which will be added to
            the ambient results by simple plus operation. Shape:
            (n_states, n_targets, n_tpoints)
        final_wake_delta_wd: numpy.ndarray
            The final wind direction wake delta, which will be added to
            the ambient results by simple plus operation. Shape:
            (n_states, n_targets, n_tpoints)

        """
        pass

    @classmethod
    def new(cls, superp_type, *args, **kwargs):
        """
        Run-time wind wake superposition model factory.

        Parameters
        ----------
        superp_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for constructor
        kwargs: dict, optional
            Additional parameters for constructor

        """
        return new_instance(cls, superp_type, *args, **kwargs)
