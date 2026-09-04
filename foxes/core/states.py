from __future__ import annotations
# mypy: disable-error-code=override

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generator, cast

from foxes.utils import new_instance

from .point_data_model import PointDataModel, PointDataModelList

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.data import FData, MData, TData


class States(PointDataModel):
    """
    Abstract base class for states.

    States describe ambient meteorological data,
    typically wind speed, wind direction, turbulence
    intensity and air density.
    """

    @abstractmethod
    def size(self) -> int:
        """
        The total number of states.

        Returns
        -------
        size
            The total number of states

        """
        pass

    def index(self) -> list[int]:
        """
        The index list

        Returns
        -------
        indices
            The index labels of states, or None for default integers

        """
        return list(range(self.size()))

    def reset(
        self,
        algo: Algorithm | None = None,
        states_sel: slice | range | list[int] | None = None,
        states_loc: list[int] | None = None,
        verbosity: int = 0,
    ) -> None:
        """
        Reset the states, optionally selecting a subset.

        Parameters
        ----------
        states_sel
            State subset selection.
        states_loc
            State index selection via the pandas loc function.
        verbosity
            The verbosity level, where 0 is silent.

        """
        raise NotImplementedError(f"States '{self.name}': Reset is not implemented")

    @abstractmethod
    def output_point_vars(self, algo: Algorithm) -> list[str]:
        """
        Return the variables modified by the model.

        Parameters
        ----------
        algo
            The calculation algorithm.

        Returns
        -------
        output_vars
            The output variable names.

        """
        pass

    def gen_states_split_size(self) -> Generator[int | None, None, None]:
        """
        Generator for suggested states split sizes for output writing.

        Yields
        ------
        split_size
            The suggested split size, or None for no splitting

        """
        yield None

    def __add__(
        self, s: PointDataModel | list[PointDataModel] | ExtendedStates
    ) -> ExtendedStates:
        if isinstance(s, list):
            return ExtendedStates(self, s)
        elif isinstance(s, ExtendedStates):
            if s.states is not self:
                raise ValueError(
                    "Cannot add extended states, since not based on same states"
                )
            return ExtendedStates(self, s.pmodels.models[1:])
        else:
            return ExtendedStates(self, [s])

    @classmethod
    def new(
        cls,
        states_type: str,
        *args: Any,
        **kwargs: Any,
    ) -> States:
        """
        Create a states instance at runtime.

        Parameters
        ----------
        states_type
            The selected derived class name.
        args
            Additional positional arguments for the constructor.
        kwargs
            Additional keyword arguments for the constructor.

        """
        return cast(States, new_instance(cls, states_type, *args, **kwargs))


class ExtendedStates(States):
    """
    States extended by point data models.
    """

    def __init__(
        self,
        states: States,
        point_models: list[PointDataModel] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        states
            The base states to start from.
        point_models
            The point models executed after states.
        """
        super().__init__()
        self.states = states
        point_models = [] if point_models is None else point_models
        self.pmodels = PointDataModelList(models=[states] + point_models)

    def append(self, model: PointDataModel) -> None:
        """
        Add a model to the list.

        Parameters
        ----------
        model
            The model to add.

        """
        self.pmodels.append(model)

    def sub_models(self) -> list[PointDataModelList]:
        """
        List of all sub-models

        Returns
        -------
        smdls
            Names of all sub models

        """
        return [self.pmodels]

    def size(self) -> int:
        """
        The total number of states.

        Returns
        -------
        size
            The total number of states

        """
        return self.states.size()

    def index(self) -> list[int]:
        """
        The index list

        Returns
        -------
        indices
            The index labels of states, or None for default integers

        """
        return self.states.index()

    def output_point_vars(self, algo: Algorithm) -> list[str]:
        """
        Return the variables modified by the model.

        Parameters
        ----------
        algo
            The calculation algorithm.

        Returns
        -------
        output_vars
            The output variable names.

        """
        return self.pmodels.output_point_vars(algo)

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
        algo
            The calculation algorithm
        mdata
            The model data
        fdata
            The farm data
        tdata
            The target point data

        Returns
        -------
        results
            The resulting data, keyed by output variable names.

        """
        return self.pmodels.calculate(algo, mdata, fdata, tdata)

    def __add__(self, m: Any) -> ExtendedStates:
        models = self.pmodels.models[1:]
        if isinstance(m, list):
            return ExtendedStates(self.states, models + m)
        elif isinstance(m, ExtendedStates):
            if m.states is not self.states:
                raise ValueError(
                    "Cannot add extended states, since not based on same states"
                )
            return ExtendedStates(self.states, models + m.pmodels.models[1:])
        else:
            return ExtendedStates(self.states, models + [m])
