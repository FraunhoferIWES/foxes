import numpy as np
from copy import deepcopy
from numpy.typing import ArrayLike
from typing import Any


class Turbine:
    """
    An individual wind turbine.

    The turbine is merely a defined by basic data
    entries and a choice of turbine models.
    """

    def __init__(
        self,
        xy: ArrayLike,
        turbine_models: list[str] | None = None,
        index: int | None = None,
        name: str | None = None,
        models_state_sel: list[np.ndarray[Any, Any] | None] | None = None,
        D: float | np.ndarray[Any, Any] | None = None,
        H: float | np.ndarray[Any, Any] | None = None,
        wind_farm_name: str | None = None,
        cluster_name: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        xy
            The turbine ground position with shape ``(2,)``.
        turbine_models
            The turbine model names as they appear in the model book.
        index
            The index in the wind farm.
        name
            The turbine name or label.
        models_state_sel
            For each turbine model, the state-selection boolean array with shape
            ``(n_states,)``.
        D
            The rotor diameter. This overwrites the turbine-type setting when
            provided.
        H
            The hub height. This overwrites the turbine-type setting when
            provided.
        wind_farm_name
            The name of the wind farm to which the turbine belongs.
        cluster_name
            The name of the cluster to which the wind farm belongs.
        """
        self.index = index
        self.name = name
        self.xy = np.array(xy)
        self.models = [] if turbine_models is None else deepcopy(turbine_models)
        self.D = D
        self.H = H
        self.wind_farm_name = wind_farm_name
        self.cluster_name = cluster_name

        self.mstates_sel: list[np.ndarray[Any, Any] | None]
        self.mstates_sel = models_state_sel if models_state_sel is not None else []
        if not self.mstates_sel:
            self.mstates_sel = [None] * len(self.models)

    def add_model(
        self, model: str, states_sel: np.ndarray[Any, Any] | None = None
    ) -> None:
        """
        Add a turbine model to the list.

        Parameters
        ----------
        model
            The model name from ``mbook.turbine_models``.
        states_sel
            The state-selection mask for the model with shape ``(n_states,)``.

        """
        self.models.append(model)
        self.mstates_sel.append(states_sel)

    def insert_model(
        self,
        index: int,
        model: str,
        states_sel: np.ndarray[Any, Any] | None = None,
    ) -> None:
        """
        Insert a turbine model into the model list.

        Parameters
        ----------
        index
            The position in the model list.
        model
            The model name from ``mbook.turbine_models``.
        states_sel
            The state-selection mask for the model with shape ``(n_states,)``.

        """
        self.models.insert(index, model)
        self.mstates_sel.insert(index, states_sel)
