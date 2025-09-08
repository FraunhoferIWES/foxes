import numpy as np
from abc import abstractmethod

from foxes.config import config
from foxes.utils import new_instance
import foxes.constants as FC
import foxes.variables as FV

from .point_data_model import PointDataModel, PointDataModelList
from .data import MData, FData, TData


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


class PopulationStates(States):
    """
    States extended by a population factor.

    For each original state, n_pop states are created.
    This is useful for parameter studies, where each
    inserted state corresponds to a different value of the
    associated variable.

    Attributes
    ----------
    states: foxes.core.States
        The original states
    n_pop: int
        The population size

    :group: core

    """

    def __init__(self, states, n_pop):
        """
        Constructor.

        Parameters
        ----------
        states: foxes.core.States
            The original states
        n_pop: int
            The population size

        """
        super().__init__()
        self.states = states
        self.n_pop = n_pop

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
        self.STATE0 = self.var(FC.STATE + "0")
        self.SMAP = self.var("SMAP")

        idata = super().load_data(algo, verbosity)
        idata0 = algo.get_model_data(self.states)
        for cname, coord in idata0["coords"].items():
            if cname != FC.STATE:
                idata["coords"][cname] = coord
            else:
                idata["coords"][self.STATE0] = coord

        for dname, (dims0, data0) in idata0["data_vars"].items():
            # if dname != FV.WEIGHT:
            hdims = tuple(
                [d if d != FC.STATE else self.STATE0 for d in np.atleast_1d(dims0)]
            )
            idata["data_vars"][dname] = (hdims, data0)

        smap = np.zeros((self.n_pop, self.states.size()), dtype=np.int32)
        smap[:] = np.arange(self.states.size())[None, :]
        smap = smap.reshape(self.size())
        idata["data_vars"][self.SMAP] = ((FC.STATE,), smap)

        found = False
        for dname, (dims0, data0) in idata["data_vars"].items():
            if self.STATE0 in dims0:
                found = True
                break
        if not found:
            del idata["coords"][self.STATE0]

        return idata

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        """
        if not self.states.initialized:
            self.states.initialize(algo, verbosity)
        super().initialize(algo, verbosity)

    def size(self):
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return self.states.size() * self.n_pop

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
        return self.states.output_point_vars(algo)

    def calculate(self, algo, mdata, fdata, tdata):
        """ "
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
            The point data

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_points)

        """
        smap = mdata[self.SMAP]

        def _map(in_data, DClass):
            if in_data is None:
                return None

            hdata = {}
            hdims = {}
            for dname, data in in_data.items():
                dms = in_data.dims[dname]
                if dname == self.SMAP or dname == self.STATE0:
                    pass
                elif dms[0] == self.STATE0:
                    hdata[dname] = data[smap]
                    hdims[dname] = tuple([FC.STATE] + list(dms)[1:])
                elif self.STATE0 in dms:
                    raise ValueError(
                        f"States '{self.name}': Found states variable not at dimension 0 for mdata entry '{dname}': {dms}"
                    )
                else:
                    hdata[dname] = data
                    hdims[dname] = dms
            return DClass(hdata, hdims, name=in_data.name + "_pop")

        hmdata = _map(mdata, MData)
        hfdata = _map(fdata, FData)
        htdata = _map(tdata, TData)
        out = self.states.calculate(algo, hmdata, hfdata, htdata)
        del hmdata, hfdata

        assert FV.WEIGHT in htdata, (
            f"Missing '{FV.WEIGHT}' in tdata results from states '{self.states.name}'"
        )
        out[FV.WEIGHT] = np.zeros(
            (htdata.n_states, htdata.n_targets, htdata.n_tpoints),
            dtype=config.dtype_double,
        )
        out[FV.WEIGHT][:] = htdata[FV.WEIGHT]

        return out
