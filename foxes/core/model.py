import numpy as np
from abc import ABCMeta
from itertools import count


class Model(metaclass=ABCMeta):
    """
    Base class for all models.

    Attributes
    ----------
    name : str
        The model name

    """

    _ids = {}

    def __init__(self):

        t = type(self).__name__
        if t not in self._ids:
            self._ids[t] = count(0)

        self._id = next(self._ids[t])
        self.name = type(self).__name__
        self.__initialized = False

    def __repr__(self):
        t = type(self).__name__
        s = self.name if self.name == t else f"{self.name} ({t})"
        return s

    @property
    def model_id(self):
        """
        Unique id based on the model type.

        Returns
        -------
        int
            Unique id of the model object

        """
        return self._id

    def var(self, v):
        """
        Creates a model specific variable name.

        Parameters
        ----------
        v : str
            The variable name

        Returns
        -------
        str
            Model specific variable name

        """
        ext = "" if self._id == 0 else f"_id{self._id}"
        return f"{self.name}{ext}_{v}"

    @property
    def initialized(self):
        """
        Initialization flag.

        Returns
        -------
        bool :
            True if the model has been initialized.

        """
        return self.__initialized

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
        self.__initialized = True

    def model_input_data(self, algo):
        """
        The model input data, as needed for the
        calculation.

        This function should specify all data
        that depend on the loop variable (e.g. state),
        or that are intended to be shared between chunks.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        idata : dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """
        return {"coords": {}, "data_vars": {}}

    def finalize(self, algo, clear_mem=False, verbosity=0):
        """
        Finalizes the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        clear_mem : bool
            Flag for deleting model data and
            resetting initialization flag
        verbosity : int
            The verbosity level, 0 = silent

        """
        self.__initialized = False

    def get_data(
        self,
        variable,
        data,
        st_sel=None,
        upcast=None,
        data_prio=True,
        accept_none=False,
    ):
        """
        Getter for a data entry in either the given
        data source, or the model object.

        Parameters
        ----------
        variable : str
            The variable, serves as data key
        data : dict
            The data source
        st_sel : numpy.ndarray of bool, optional
            If given, get the specified state-turbine subset
        upcast : str, optional
            Either 'farm' or 'points', broadcasts potential
            scalar data to numpy.ndarray with dimensions
            (n_states, n_turbines) or (n_states, n_points),
            respectively
        data_prio: bool
            First search the data source, then the object
        accept_none: bool
            Do not throw an error if data entry is None or np.nan

        """

        sources = ("data", "self") if data_prio else ("self", "data")

        out = None
        for s in sources:
            if s == "self":
                try:
                    out = getattr(self, variable)
                    break
                except AttributeError:
                    pass
            else:
                try:
                    out = data[variable]
                    break
                except KeyError:
                    pass

        if out is None:
            raise KeyError(
                f"Model '{self.name}': Variable '{variable}' neither found in data {sorted(list(data.keys()))} nor among attributes"
            )

        if upcast is not None and not isinstance(out, np.ndarray):
            if upcast == "farm":
                out = np.full((data.n_states, data.n_turbines), out)
            elif upcast == "points":
                out = np.full((data.n_states, data.n_points), out)
            else:
                raise ValueError(
                    f"Model '{self.name}': Illegal upcast '{upcast}', select 'farm' or 'points'"
                )

        if st_sel is not None:
            try:
                out = out[st_sel]
            except TypeError:
                pass

        if not accept_none:
            try:
                if np.all(np.isnan(np.atleast_1d(out))):
                    raise ValueError(
                        f"Model '{self.name}': Variable '{variable}' requested but not provided."
                    )
            except TypeError:
                pass
        return out
