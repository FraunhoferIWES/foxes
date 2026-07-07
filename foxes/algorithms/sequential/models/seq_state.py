from __future__ import annotations
# mypy: disable-error-code=override

from typing import TYPE_CHECKING, Any

from foxes.core.states import States

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData
    from foxes.core.model import LoadedData


class SeqState(States):
    """
    A single state during sequential iteration, just serving
    as a structural placeholder

    Parameters
    ----------
    states: foxes.core.States
        The original states set

    :group: algorithms.sequential.models

    """

    def __init__(self, states: States) -> None:
        """
        Constructor.

        Attributes
        ----------
        states: foxes.core.States
            The original states set

        """
        super().__init__()
        self.states = states

        # updated by SequentialIter:
        self._size: int = states.size()
        self._indx: Any | None = None
        self._counter: int | None = None

    def sub_models(self) -> list[Any]:
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return [self.states]

    def initialize(
        self,
        algo: Algorithm,
        loaded_data: LoadedData | None = None,
        force: bool = False,
        verbosity: int = 0,
    ) -> LoadedData:
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        loaded_data: dict, optional
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force: bool
            Overwrite existing data
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        loaded_data: dict
            The loaded data, containing keys "coords", "data_vars", and "extra_data".
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.

        """
        loaded_data = super().initialize(
            algo, loaded_data=loaded_data, force=force, verbosity=verbosity
        )
        self._size = self.states.size()
        return loaded_data

    def size(self) -> int:
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return self._size

    def index(self) -> list[Any] | Any:
        """
        The index list

        Returns
        -------
        indices: array_like
            The index labels of states, or None for default integers

        """
        return [self._indx] if self._size == 1 else self.states.index()

    @property
    def counter(self) -> int | None:
        """
        The current index counter

        Returns
        -------
        i: int
            The current index counter

        """
        return self._counter

    def output_point_vars(self, algo: Algorithm) -> list[str]:
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
        return self.states.output_point_vars(algo)

    def calculate(
        self,
        algo: Algorithm,
        mdata: MData,
        fdata: FData,
        tdata: TData,
    ) -> dict[str, Any]:
        """
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
        tdata: foxes.core.Data
            The target point data

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints)

        """
        return self.states.calculate(algo, mdata, fdata, tdata)
