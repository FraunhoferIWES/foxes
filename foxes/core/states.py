from abc import abstractmethod

from foxes.utils import new_instance
import foxes.constants as FC

from .point_data_model import PointDataModel, PointDataModelList


class States(PointDataModel):
    """
    Abstract base class for states.

    States describe ambient meteorological data,
    typically wind speed, wind direction, turbulence
    intensity and air density.

    :group: core

    """

    @abstractmethod
    def size(self):
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        pass

    def index(self):
        """
        The index list

        Returns
        -------
        indices: array_like
            The index labels of states, or None for default integers

        """
        return list(range(self.size()))

    def reset(self, algo=None, states_sel=None, states_loc=None, verbosity=0):
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

    def load_data(self, algo, verbosity=0):
        """
        Load and/or create all model data that is subject to chunking.

        Such data should not be stored under self, for memory reasons. The
        data returned here will automatically be chunked and then provided
        as part of the mdata object during calculations.

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
        idata = super().load_data(algo, verbosity)

        sinds = self.index()
        if sinds is not None:
            idata["coords"][FC.STATE] = sinds

        return idata

    @abstractmethod
    def output_point_vars(self, algo):
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

    def __add__(self, s):
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
    def new(cls, states_type, *args, **kwargs):
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

    def __init__(self, states, point_models=[]):
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

    def append(self, model):
        """
        Add a model to the list

        Parameters
        ----------
        model: foxes.core.PointDataModel
            The model to add

        """
        self.pmodels.append(model)

    def sub_models(self):
        """
        List of all sub-models

        Returns
        -------
        smdls: list of foxes.core.Model
            Names of all sub models

        """
        return [self.pmodels]

    def size(self):
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return self.states.size()

    def index(self):
        """
        The index list

        Returns
        -------
        indices: array_like
            The index labels of states, or None for default integers

        """
        return self.states.index()

    def output_point_vars(self, algo):
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

    def calculate(self, algo, mdata, fdata, tdata):
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

    def __add__(self, m):
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
