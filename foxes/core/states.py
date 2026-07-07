from __future__ import annotations
# mypy: disable-error-code=override

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generator

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

    :group: core

    """

    @abstractmethod
    def size(self) -> int:
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        pass

    def index(self) -> Any:
        """
        The index list

        Returns
        -------
        indices: array_like
            The index labels of states, or None for default integers

        """
        return list(range(self.size()))

    def reset(
        self,
        algo: Algorithm | None = None,
        states_sel: slice | range | list[int] | None = None,
        states_loc: list[Any] | None = None,
        verbosity: int = 0,
    ) -> None:
        """
        Reset the states, optionally select states

        Parameters
        ----------
        states_sel: slice or range or list of int, optional
            States subset selection
        states_loc: list, optional
            State index selection via pandas loc function
        verbosity: int
            The verbosity level, 0 = silent

        """
        raise NotImplementedError(f"States '{self.name}': Reset is not implemented")

    @abstractmethod
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
        pass

    def gen_states_split_size(self) -> Generator[int | None, None, None]:
        """
        Generator for suggested states split sizes for output writing.

        Yields
        ------
        split_size: int or None
            The suggested split size, or None for no splitting

        """
        yield None

    def __add__(self, s: Any) -> ExtendedStates:
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
        Run-time states factory.

        Parameters
        ----------
        states_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for constructor
        kwargs: dict, optional
            Additional parameters for constructor

        """
        return new_instance(cls, states_type, *args, **kwargs)


class ExtendedStates(States):
    """
    States extended by point data models.

    Attributes
    ----------
    states: foxes.core.States
        The base states to start from
    pmodels: foxes.core.PointDataModelList
        The point models, including states as first model

    :group: core

    """

    def __init__(
        self,
        states: States,
        point_models: list[PointDataModel] = [],
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        states: foxes.core.States
            The base states to start from
        point_models: list of foxes.core.PointDataModel, optional
            The point models, executed after states

        """
        super().__init__()
        self.states = states
        self.pmodels = PointDataModelList(models=[states] + point_models)

    def append(self, model: PointDataModel) -> None:
        """
        Add a model to the list

        Parameters
        ----------
        model: foxes.core.PointDataModel
            The model to add

        """
        self.pmodels.append(model)

    def sub_models(self) -> list[PointDataModelList]:
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return [self.pmodels]

    def size(self) -> int:
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return self.states.size()

    def index(self) -> Any:
        """
        The index list

        Returns
        -------
        indices: array_like
            The index labels of states, or None for default integers

        """
        return self.states.index()

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
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        tdata: foxes.core.TData
            The target point data

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_points)

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
