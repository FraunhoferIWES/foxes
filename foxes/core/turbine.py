import numpy as np


class Turbine:
    """
    An individual wind turbine.

    The turbine is merely a defined by basic data
    entries and a choice of turbine models.

    Attributes
    ----------
    xy: array_like
        The turbine ground position, shape: (2,)
    models: list of str
        The turbine model names, as they appear
        in the model book
    index: int, optional
        The index in the wind farm
    name: str, optional
        The turbine name/label
    mstates_sel: list of numpy.ndarray, optional
        For each turbine model, the state selection
        boolean array with shape (n_states,)
    D: float, optional
        The rotor diameter. Overwrites turbine type
        settings if given
    H: float, optional
        The hub height. Overwrites turbine type
        settings if given

    :group: core

    """

    def __init__(
        self,
        xy,
        turbine_models=[],
        index=None,
        name=None,
        models_state_sel=None,
        D=None,
        H=None,
    ):
        """
        Constructor.

        Parameters
        ----------
        xy: array_like
            The turbine ground position, shape: (2,)
        turbine_models: list of str
            The turbine model names, as they appear
            in the model book
        index: int, optional
            The index in the wind farm
        name: str, optional
            The turbine name/label
        models_state_sel: list of numpy.ndarray, optional
            For each turbine model, the state selection
            boolean array with shape (n_states,)
        D: float, optional
            The rotor diameter. Overwrites turbine type
            settings if given
        H: float, optional
            The hub height. Overwrites turbine type
            settings if given

        """
        self.index = index
        self.name = name
        self.xy = np.array(xy)
        self.models = turbine_models
        self.D = D
        self.H = H

        self.mstates_sel = models_state_sel
        if self.mstates_sel is None:
            self.mstates_sel = [None] * len(self.models)

    def add_model(self, model, states_sel=None):
        """
        Add a turbine model to the list.

        Parameters
        ----------
        model: foxes.core.TurbineModel
            The model
        states_sel: numpy.ndarray of bool, optional
            The states selection for the model, shape: (n_states,)

        """
        self.models.append(model)
        self.mstates_sel.append(states_sel)

    def insert_model(self, index, model, states_sel=None):
        """
        Insert a turbine model into the list of models.

        Parameters
        ----------
        index: int
            The position in the model list
        model: foxes.core.TurbineModel
            The model
        states_sel: numpy.ndarray of bool, optional
            The states selection for the model, shape: (n_states,)

        """
        self.models.insert(index, model)
        self.mstates_sel.insert(index, states_sel)
