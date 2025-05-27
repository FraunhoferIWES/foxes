from foxes.utils import new_instance
from .model import Model


class WakeDeflection(Model):
    """
    Abstract base class for wake deflection models.

    :group: core

    """

    def update_coos(
        self,
        algo, 
        mdata,
        fdata, 
        tdata, 
        downwind_index, 
        wframe, 
        wmodel, 
        coos,
    ):
        """
        Updates the wake coordinates

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
        wframe: foxes.core.WakeFrame
            The wake frame
        wmodel: foxes.core.WakeModel
            The wake model
        coos: numpy.ndarray
            The wake frame coordinates of the evaluation
            points, shape: (n_states, n_targets, n_tpoints, 3)

        """
        raise NotImplementedError(f"Wake deflection '{self.name}' not implemented for wake frame '{wframe.name}' and wake model '{wmodel.name}'")

    @classmethod
    def new(cls, wframe_type, *args, **kwargs):
        """
        Run-time wake deflection model factory.

        Parameters
        ----------
        wframe_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for constructor
        kwargs: dict, optional
            Additional parameters for constructor

        """
        return new_instance(cls, wframe_type, *args, **kwargs)
