from foxes.core import PartialWakesModel
import foxes.variables as FV
import foxes.constants as FC


class WakeMirror(PartialWakesModel):
    """
    Adds mirror turbines that model wake reflection from a horizontal plane,
    e.g. the ground

    Attributes
    ----------
    base: str
        Name of the underlying partial wakes model
    heights: list of float
        The reflection heights

    :group: models.partial_wakes

    """

    def __init__(self, base, heights):
        """
        Constructor.

        Parameters
        ----------
        base: str
            Name of the underlying partial wakes model
        heights: list of float
            The reflection heights

        """
        super().__init__()
        self.heights = heights
        self.base = base
        self._base = None

    def __repr__(self):
        return f"{type(self).__name__}(heights={self.heights}) with base {self._base}"

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
        self._base = algo.mbook.partial_wakes[self.base]
        super().initialize(algo, verbosity, force)

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return [self._base]

    def get_wake_points(self, *args, **kwargs):
        """
        Get the wake calculation points, and their
        weights.

        Parameters
        ----------
        args: tuple, optional
            Parameters for the underlying partial wake model
        kwargs: dict, optional
            Parameters for the underlying partial wake model

        Returns
        -------
        rpoints: numpy.ndarray
            The wake calculation points, shape:
            (n_states, n_turbines, n_tpoints, 3)
        rweights: numpy.ndarray
            The target point weights, shape: (n_tpoints,)

        """
        return self._base.get_wake_points(*args, **kwargs)

    def new_wake_deltas(self, *args, **kwargs):
        """
        Creates new initial wake deltas, filled
        with zeros.

        Parameters
        ----------
        args: tuple, optional
            Parameters for the underlying partial wake model
        kwargs: dict, optional
            Parameters for the underlying partial wake model

        Returns
        -------
        wake_deltas: dict
            Key: variable name, value: The zero filled
            wake deltas, shape: (n_states, n_turbines, n_tpoints, ...)

        """
        return self._base.new_wake_deltas(*args, **kwargs)
    
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
            in the downwnd order
        wake_deltas: dict
            The wake deltas. Key: variable name,
            value: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints, ...)
        wmodel: foxes.core.WakeModel
            The wake model

        """
        # prepare:
        hh = fdata[FV.H][:, downwind_index].copy()

        # contribution from main wake:
        wcoos = algo.wake_frame.get_wake_coos(algo, mdata, fdata, tdata, downwind_index)
        wmodel.contribute(algo, mdata, fdata, tdata, downwind_index, wcoos, wake_deltas)

        # contribution from mirrors:
        tdata[FC.TARGETS] = tdata[FC.TARGETS].copy()  # making sure this is no ref
        for h in self.heights:

            fdata[FV.H][:, downwind_index] = hh + 2 * (h - hh)

            wcoos = algo.wake_frame.get_wake_coos(algo, mdata, fdata, tdata, downwind_index)
            wmodel.contribute(algo, mdata, fdata, tdata, downwind_index, wcoos, wake_deltas)

        # reset heights:
        fdata[FV.H][:, downwind_index] = hh

    def finalize_wakes(self,*args, **kwargs):
        """
        Updates the wake_deltas at the selected target
        downwind index.

        Modifies wake_deltas on the fly.

        Parameters
        ----------
        args: tuple, optional
            Parameters for the underlying partial wake model
        kwargs: dict, optional
            Parameters for the underlying partial wake model

        Returns
        -------
        final_wake_deltas: dict
            The final wake deltas at the selected downwind
            turbines. Key: variable name, value: np.ndarray
            of shape (n_states, n_rotor_points)

        """
        return self._base.finalize_wakes(*args, **kwargs)


class GroundMirror(WakeMirror):
    """
    Adds mirror turbines that model wake reflection from the ground

    :group: models.partial_wakes

    """

    def __init__(self, *args, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Additional parameters for WakeMirror
        kwargs: dict, optional
            Additional parameters for WakeMirror

        """
        super().__init__(*args, heights=[0], **kwargs)
