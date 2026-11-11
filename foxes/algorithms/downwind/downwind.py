from __future__ import annotations

from typing import TYPE_CHECKING, Any

# mypy: disable-error-code=override
import numpy as np
from xarray import Dataset

import foxes.constants as FC
import foxes.models as fm
import foxes.variables as FV
from foxes.core import (
    Algorithm,
    FarmController,
    FarmDataModelList,
    Output,
    OutputList,
    PointDataModel,
    PointDataModelList,
)
from foxes.utils import Dict

from . import models as mdls

if TYPE_CHECKING:
    from foxes.core import (
        GroundModel,
        PartialWakesModel,
        RotorModel,
        States,
        WakeDeflection,
        WakeFrame,
        WakeModel,
        WindFarm,
    )
    from foxes.models import ModelBook

    from .models.population import PopulationModel


class Downwind(Algorithm):
    """
    The downwind algorithm.

    The turbines are evaluated once, in the order
    that is calculated by the provided `TurbineOrder`
    object.
    """

    def __init__(
        self,
        farm: WindFarm,
        states: States,
        wake_models: list[str],
        rotor_model: str = "centre",
        wake_frame: str = "rotor_wd",
        wake_deflection: str = "no_deflection",
        partial_wakes: dict[str, str] | list[str] | str | None = None,
        ground_models: dict[str, str] | list[str] | str | None = None,
        farm_controller: str = "basic_ctrl",
        mbook: ModelBook | None = None,
        max_wake_length_km: float | None = None,
        population_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        farm
            The wind farm
        states
            The ambient states
        wake_models
            The wake models, applied to all turbines.
            Will be looked up in the model book
        rotor_model
            The rotor model, for all turbines. Will be
            looked up in the model book
        wake_frame
            The wake frame. Will be looked up in the
            model book
        wake_deflection
            The wake deflection model
        partial_wakes
            The partial wakes mapping. Key: wake model name,
            value: partial wake model name
        ground_models
            The ground models mapping. Key: wake model name,
            value: ground model name
        farm_controller
            The farm controller. Will be
            looked up in the model book
        mbook
            The model book
        max_wake_length_km
            The maximum wake length in km. If None, no maximum length is applied.
        population_params
            The population parameters. If provided, this will be
            used to create the population model.
        kwargs
            Additional parameters for the base class
        """
        if mbook is None:
            mbook = fm.ModelBook()

        super().__init__(mbook, farm, **kwargs)

        self._SETPOP = None
        self._pop_model: PopulationModel | None = None
        if population_params is None:
            self.__states: States = states
        else:
            self._SETPOP = "set_pop_data"
            assert self._SETPOP not in mbook.turbine_models, (
                f"Algorithm '{self.name}': Model name '{self._SETPOP}' for population data model is reserved"
            )
            self._pop_model = self.get_model("PopulationModel")(**population_params)
            mbook.turbine_models[self._SETPOP] = self._pop_model
            self.__states = self.get_model("PopulationStates")(
                states, n_pop=self._pop_model.n_pop
            )
            for t in self.farm.turbines:
                if self._SETPOP not in t.models:
                    t.insert_model(0, self._SETPOP)
        self.n_states: int | None = None

        self.__rotor_model: RotorModel = self.mbook.rotor_models.get_item(rotor_model)
        self.rotor_model.name = rotor_model

        self.__wake_frame: WakeFrame = self.mbook.wake_frames.get_item(wake_frame)
        self.wake_frame.name = wake_frame

        self.__wake_deflection: WakeDeflection = self.mbook.wake_deflections.get_item(
            wake_deflection
        )
        self.wake_deflection.name = wake_deflection

        self.__wake_models: dict[str, WakeModel] = {}
        for w in wake_models:
            m = self.mbook.wake_models.get_item(w)
            m.name = w
            self.wake_models[w] = m

        def _set_wspecific(
            descr: str,
            target: dict[str, Any],
            values: dict[str, str] | list[str] | str | None,
            deffunc: Any,
            mbooks: Any,
            checkw: bool,
        ) -> None:
            if values is None:
                values = {}
            if isinstance(values, list) and len(values) == 1:
                values = values[0]
            if isinstance(values, str):
                for w in wake_models:
                    try:
                        pw = values
                        if checkw:
                            mbooks.get_item(pw).check_wmodel(
                                self.wake_models[w], error=True
                            )
                    except TypeError:
                        pw = deffunc(self.wake_models[w])
                    target[w] = mbooks.get_item(pw)
                    target[w].name = pw
            elif isinstance(values, list):
                for i, w in enumerate(wake_models):
                    if i >= len(values):
                        raise IndexError(
                            f"Not enough {descr} in list {values}, expecting {len(wake_models)}"
                        )
                    pw = values[i]
                    target[w] = mbooks.get_item(pw)
                    target[w].name = pw
            else:
                for w in wake_models:
                    if w in values:
                        pw = values[w]
                    else:
                        pw = deffunc(self.wake_models[w])
                    target[w] = mbooks.get_item(pw)
                    target[w].name = pw

        self.__partial_wakes: dict[str, PartialWakesModel] = {}
        _set_wspecific(
            descr="partial wakes",
            target=self.partial_wakes,
            values=partial_wakes,
            deffunc=mbook.default_partial_wakes,
            mbooks=self.mbook.partial_wakes,
            checkw=True,
        )

        self.__ground_models: dict[str, GroundModel] = {}
        _set_wspecific(
            descr="ground models",
            target=self.ground_models,
            values=ground_models,
            deffunc=lambda w: "no_ground",
            mbooks=self.mbook.ground_models,
            checkw=False,
        )

        self.__farm_controller: FarmController = self.mbook.farm_controllers.get_item(
            farm_controller
        )
        self.farm_controller.name = farm_controller
        self.farm_controller.find_turbine_types(self)

        self.__max_wlength_km = max_wake_length_km

    @property
    def states(self) -> States:
        """
        The states

        Returns
        -------
        states
            The states

        """
        return self.__states

    @states.setter
    def states(self, value: States) -> None:
        """Resets the states"""
        if self.__states is not value:
            if self.running:
                raise ValueError(f"{self.name}: Cannot set states while running")
            if self.states.initialized:
                self.states.finalize(self, verbosity=self.verbosity)
            self.__states = value
            self.init_states()

    @property
    def rotor_model(self) -> RotorModel:
        """
        The rotor model

        Returns
        -------
        rotor_model
            The rotor model

        """
        return self.__rotor_model

    @property
    def wake_models(self) -> dict[str, WakeModel]:
        """
        The wake models

        Returns
        -------
        wake_models
            The wake models. Key: name,
            value: the wake model

        """
        return self.__wake_models

    @property
    def wake_frame(self) -> WakeFrame:
        """
        The wake frame

        Returns
        -------
        wake_frame
            The wake frame

        """
        return self.__wake_frame

    @property
    def wake_deflection(self) -> WakeDeflection:
        """
        The wake deflection

        Returns
        -------
        wake_deflection
            The wake deflection model

        """
        return self.__wake_deflection

    @property
    def partial_wakes(self) -> dict[str, PartialWakesModel]:
        """
        The partial wakes models

        Returns
        -------
        partial_wakes
            The partial wakes models. Key: name,
            value: the partial wake model

        """
        return self.__partial_wakes

    @property
    def ground_models(self) -> dict[str, GroundModel]:
        """
        The ground models

        Returns
        -------
        ground_models
            The ground models, key: name,
            value: the ground model

        """
        return self.__ground_models

    @property
    def farm_controller(self) -> FarmController:
        """
        The farm controller

        Returns
        -------
        farm_controller
            The farm controller

        """
        return self.__farm_controller

    @property
    def population_model(self) -> PopulationModel | None:
        """
        The population model

        Returns
        -------
        population_model
            The population model, or None if not used

        """
        if self._SETPOP is None:
            return None
        return self._pop_model

    @property
    def max_wake_length_km(self) -> float:
        """
        The maximum wake length in km

        Returns
        -------
        max_wake_length_km
            The maximum wake length in km, or None if not set

        """
        if self.__max_wlength_km is None:
            raise KeyError(f"Algorithm '{self.name}': No maximum wake length set")
        return self.__max_wlength_km

    @property
    def has_max_wake_length(self) -> bool:
        """
        Whether a maximum wake length is set

        Returns
        -------
        has_max_wake_length
            True if a maximum wake length is set, False otherwise

        """
        return self.__max_wlength_km is not None

    def select_population_member(
        self,
        pop_farm_results: Dataset,
        pop_index: int | np.ndarray,
    ) -> Dataset:
        """
        Select a specific population member from the population model results.

        Parameters
        ----------
        pop_farm_results
            The farm results including population index dimension
        pop_index
            The population index to select. Either a single index
            for all states, or an array of shape (n_states,)

        Returns
        -------
        farm_results
            The farm results for the selected population member.

        """
        if self._SETPOP is None:
            raise ValueError(f"Algorithm '{self.name}': No population model defined")
        ini = self.initialized
        if ini:
            self.finalize()
        pop_states = self.states
        from .models.population import PopulationStates as _PopulationStates

        assert isinstance(pop_states, _PopulationStates)
        self.states = pop_states.states
        for t in self.farm.turbines:
            if self._SETPOP in t.models:
                del t.models[t.models.index(self._SETPOP)]
        if ini:
            self.initialize()

        POP = FC.POP
        assert POP in pop_farm_results.sizes, (
            f"Algorithm '{self.name}': Population index coordinate '{POP}' not found in provided farm results"
        )
        if isinstance(pop_index, np.ndarray):
            return Dataset(
                {
                    v: (
                        (FC.STATE, FC.TURBINE),
                        np.take_along_axis(d.values, pop_index[None, :, None], axis=0)[
                            0
                        ],
                    )
                    if d.dims == (POP, FC.STATE, FC.TURBINE)
                    else d
                    for v, d in pop_farm_results.data_vars.items()
                }
            )
        else:
            return pop_farm_results.sel({POP: pop_index})

    @classmethod
    def get_model(cls, name: str) -> Any:
        """
        Get the algorithm specific model

        Parameters
        ----------
        name
            The model name

        Returns
        -------
        model
            The model

        """
        return getattr(mdls, name)

    def update_n_turbines(self) -> None:
        """
        Reset the number of turbines,
        according to self.farm
        """
        if self.n_turbines != self.farm.n_turbines:
            super().update_n_turbines()
            self.farm_controller.find_turbine_types(self)
            self.farm_controller.collect_models(self)

    def print_deco(
        self,
        func_name: str | None = None,
        n_points: int | None = None,
        outputs: Output | None = None,
    ) -> None:
        """
        Helper function for printing model names

        Parameters
        ----------
        func_name
            Name of the calling function
        n_points
            The number of points
        outputs
            The outputs to be considered for printing

        """
        if self.verbosity > 0:
            deco = "-" * 60
            print(f"\n{deco}")
            print(f"  Algorithm: {type(self).__name__}")
            if func_name is not None:
                print(f"  Running {self.name}: {func_name}")
            print(deco)
            print(f"  n_states : {self.n_states}")
            print(f"  n_turbines: {self.n_turbines}")
            if n_points is not None:
                print(f"  n_points : {n_points}")
            print(deco)
            print(f"  states    : {self.states}")
            print(f"  rotor     : {self.rotor_model}")
            print(f"  controller: {self.farm_controller}")
            print(f"  deflection: {self.wake_deflection}")
            print(f"  wake frame: {self.wake_frame}")
            wl = (
                f"{self.__max_wlength_km} km"
                if self.__max_wlength_km is not None
                else None
            )
            print(f"  wake lngth: {wl}")
            print(deco)
            print("  wakes:")
            for i, wake_model in enumerate(self.wake_models.values()):
                print(f"    {i}) {wake_model.name}: {wake_model}")
            print(deco)
            print("  partial wakes:")
            for i, (wake_name, pwake) in enumerate(self.partial_wakes.items()):
                print(f"    {i}) {wake_name}: {pwake.name}, {pwake}")
            print(deco)
            print("  ground models:")
            for i, (wake_name, gmodel) in enumerate(self.ground_models.items()):
                print(f"    {i}) {wake_name}: {gmodel.name}, {gmodel}")
            print(deco)
            print("  turbine models:")
            assert self.farm_controller.pre_rotor_models is not None
            assert self.farm_controller.post_rotor_models is not None
            for i, m in enumerate(self.farm_controller.pre_rotor_models.models):
                print(f"    {i}) {m.name}: {m} [pre-rotor]")
            for i, m in enumerate(self.farm_controller.post_rotor_models.models):
                print(
                    f"    {i + len(self.farm_controller.pre_rotor_models.models)}) {m.name}: {m}"
                )
            print(deco)
            if outputs is not None:
                print("  outputs:")
                if isinstance(outputs, OutputList):
                    output_models = outputs.outputs
                else:
                    output_models = [outputs]
                for i, output in enumerate(output_models):
                    print(f"    {i}) {output}")
                print(deco)
            print()

    def _print_model_oder(
        self,
        mlist: FarmDataModelList | PointDataModelList,
        calc_pars: list[dict[str, Any]],
        outputs: Any | None = None,
    ) -> None:
        """Print the ordered calculation chain at the configured verbosity.

        Parameters
        ----------
        mlist
            Farm- or point-model chain in execution order.
        calc_pars
            Per-model calculation keyword mappings aligned with ``mlist``.
        outputs
            Selected output model or chain, included in the printed summary.
        """
        if self.verbosity > 0:
            deco = "-" * 50
            print(f"\n{deco}")
            print("  Model oder")
            print(f"{deco}")

            for i, m in enumerate(mlist.models):
                print(f"{i:02d}) {m.name}")
                if isinstance(m, FarmController):
                    if calc_pars[i]["pre_rotor"]:
                        assert m.pre_rotor_models is not None
                        for j, mm in enumerate(m.pre_rotor_models.models):
                            print(f"  {i:02d}.{j}) Pre-rotor: {mm.name}")
                    else:
                        assert m.post_rotor_models is not None
                        for j, mm in enumerate(m.post_rotor_models.models):
                            print(f"  {i:02d}.{j}) Post-rotor: {mm.name}")
            if outputs is not None:
                if isinstance(outputs, OutputList):
                    outputs = outputs.outputs
                else:
                    outputs = [outputs]
                for j, output in enumerate(outputs):
                    print(f"    {i + j}) {output.name}")

            print(deco)
            print()

    def init_states(self, force: bool = False) -> None:
        """
        Initialize states, if needed.

        Parameters
        ----------
        force
            Force initialization even if already initialized

        """
        if force or not self.states.initialized:
            self.states.initialize(
                self,
                loaded_data=self.loaded_data,
                force=force,
                verbosity=self.verbosity,
            )
        self.n_states = self.states.size()

    def sub_models(self) -> list[Any]:
        """
        List of all sub-models

        Returns
        -------
        smdls
            All sub models

        """
        mdls = [
            self.states,
            self.farm_controller,
            self.rotor_model,
            self.wake_deflection,
            self.wake_frame,
        ]
        mdls += list(self.wake_models.values())
        mdls += list(self.partial_wakes.values())
        mdls += list(self.ground_models.values())

        return mdls

    def initialize(self, force: bool = False) -> None:
        """
        Initializes the algorithm.

        Parameters
        ----------
        force
            Overwrite existing data

        """
        if force:
            self.clear_loaded_data()

        self.init_states(force=force)

        self.print(f"\nInitializing algorithm '{self.name}'")
        super().initialize(force=force)

    def _add_turbine_evaluation_models(
        self,
        mlist: FarmDataModelList,
        calc_pars: list[dict[str, Any]],
        calc_parameters: dict[str, dict[str, Any]],
        selection_in_farm_order: bool = False,
    ) -> None:
        """Append evaluation-only execution to ``mlist`` and ``calc_pars``."""
        # Reuse controller ordering and selections without rerunning power models.
        post_rotor = self.farm_controller.post_rotor_models
        assert post_rotor is not None
        evaluations = [
            m.name
            for m in post_rotor.models
            if isinstance(m, fm.turbine_models.TurbineEvaluation)
        ]
        if evaluations:
            mlist.models.append(self.farm_controller)
            calc_pars.append(
                dict(
                    calc_parameters.get(self.farm_controller.name, {}),
                    pre_rotor=False,
                    model_names=evaluations,
                    selection_in_farm_order=selection_in_farm_order,
                )
            )

    def _collect_farm_models(
        self,
        calc_parameters: dict[str, dict[str, Any]],
        ambient: bool,
        output: Output | None = None,
    ) -> tuple[FarmDataModelList, list[dict[str, Any]]]:
        """
        Helper function that creates model list
        """
        # prepare:
        calc_pars = []
        mlist = FarmDataModelList(models=[])
        mlist.name = f"{self.name}_calc"

        # 0) run pre-rotor turbine models via farm controller:
        mlist.models.append(self.farm_controller)
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        calc_pars[-1]["pre_rotor"] = True

        # 1) set initial data:
        mlist.models.append(self.get_model("InitFarmData")())
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        # 2) calculate ambient rotor results:
        mlist.models.append(self.rotor_model)
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        calc_pars[-1]["store"] = True

        # 3) run post-rotor turbine models via farm controller:
        mlist.models.append(self.farm_controller)
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        calc_pars[-1]["pre_rotor"] = False

        # 4) copy results to ambient, requires self.farm_vars:
        self.farm_vars = mlist.output_farm_vars(self)
        mlist.models.append(self.get_model("SetAmbFarmResults")())
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        # 5) calculate wake effects:
        if not ambient:
            mlist.models.append(self.get_model("FarmWakesCalculation")())
            calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        # Evaluate final powers, including unwaked turbines and ambient-only runs.
        self._add_turbine_evaluation_models(mlist, calc_pars, calc_parameters)

        # 6) prepare for output:
        mlist.models.append(self.get_model("PrepareForOutput")(output=output))
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        return mlist, calc_pars

    def _calc_farm_vars(self, mlist: FarmDataModelList) -> None:
        """Helper function that gathers the farm variables"""
        self.farm_vars = sorted(list(mlist.output_farm_vars(self)))

    def _calc_point_vars(self, mlist: PointDataModelList) -> None:
        """Helper function that gathers the point variables"""
        self.point_vars = sorted(list(mlist.output_point_vars(self)))

    def _resolve_outputs(
        self,
        outputs: str | Output | list[str | Output] | None,
    ) -> tuple[Output | None, bool]:
        """Resolve output selectors without copying or initializing models.

        Parameters
        ----------
        outputs
            An output instance, a ``ModelBook.outputs`` name or factory name,
            or a list of either. ``None`` disables output collection.

        Returns
        -------
        output
            The supplied or registered instance, an ordered ``OutputList`` for
            zero or multiple list entries, or ``None``. Instances are reused,
            not copied.
        single_output
            Whether a scalar selector or one-entry list selected one model.
            This permits unwrapping a mapping only when it has one result.

        Raises
        ------
        KeyError
            If an output model-book name is unavailable.
        TypeError
            If ``outputs`` is neither a supported selector nor ``None``.
        """
        # Preserve instance identity and declaration order across selector forms.
        if isinstance(outputs, Output):
            return outputs, True
        if isinstance(outputs, str):
            return self.mbook.outputs[outputs], True
        if isinstance(outputs, list):
            models = [
                self.mbook.outputs[o] if isinstance(o, str) else o for o in outputs
            ]
            oneout = len(models) == 1
            return models[0] if oneout else OutputList(outputs=models), oneout
        if outputs is None:
            return None, False
        raise TypeError(f"{self.name}: Expected output models or model-book names")

    def calc_farm(
        self,
        outputs: str | Output | list[str | Output] | None = "farm_results",
        calc_parameters: dict[str, dict[str, Any]] = {},
        ambient: bool = False,
        finalize: bool = True,
        clear_mem: bool = False,
        **kwargs: Any,
    ) -> Dataset | dict[str, Any]:
        """
        Calculate farm data.

        Parameters
        ----------
        outputs
            An ``Output`` instance, a name from ``ModelBook.outputs``, or a list
            of instances and names. A single output with one result returns that
            result directly. Multiple outputs return a result-name mapping.
            ``None`` or an empty list disables collection.
            Outputs requiring points evaluate point slices per state chunk,
            bounded by the engine's point chunk size. Choose farm variables through
            ``FarmResultsOutput(variables=...)``, not a list of variable names.
        calc_parameters
            Model-name-to-keyword-parameter mapping for scientific calculation.
        ambient
            Skip wake effects and calculate ambient results only.
        finalize
            Finalize initialized models and the algorithm after calculation.
        clear_mem
            Remove loaded model data when launching the run and finalize
            afterward, even if ``finalize`` is false.
        kwargs
            Additional engine calculation options passed through the base
            algorithm, including state selections.

        Returns
        -------
        results
            The sole result of a single selected output, otherwise a
            result-name-to-value mapping. ``None`` or an empty output list gives
            an empty mapping. The default dataset preserves input state order
            and original farm turbine order, with physical fields on
            ``(FC.STATE, FC.TURBINE)`` and explicit coordinates.

        Raises
        ------
        KeyError
            If an output model-book name is unavailable.
        TypeError
            If ``outputs`` is not a supported output selector.
        ValueError
            If composed outputs request different calculation points.

        Notes
        -----
        Output models are initialized before execution and remain read-only
        while running. Numerical buffers belong to chunk data or result mappings;
        rendering and result assembly occur during parent-side finalization.

        """
        # initialize algorithm:
        if not self.initialized:
            self.initialize()

        # Resolve output names and preserve the single-versus-multiple return contract.
        output, oneout = self._resolve_outputs(outputs)

        # Initialize the resolved output chain before preparing model inputs.
        if output is not None:
            self.init_model(output)

        # collect models:
        if isinstance(output, OutputList):
            output.get_calc_points(self)
        mlist, calc_pars = self._collect_farm_models(
            calc_parameters, ambient, output=output
        )

        # initialize models:
        if not mlist.initialized:
            self.init_model(mlist)
            self._calc_farm_vars(mlist)
            self.point_vars = []

        # welcome:
        self.print_deco("calc_farm", outputs=output)
        self._print_model_oder(mlist, calc_pars, output)

        # get input model data:
        model_data, extra_data = self.get_model_data(pop=clear_mem)
        self.print("\nInput model data:\n\n", model_data, "\n")
        if len(extra_data) > 0:
            self.print("Extra data:")
            for v, d in extra_data.items():
                self.print(f"  {v}: {type(d).__name__}")
        self.print("\nFarm variables:", ", ".join(self.farm_vars))

        # run main calculation:
        results = super().calc_farm(
            mlist,
            model_data,
            extra_data=extra_data,
            parameters=calc_pars,
            output=output,
            clear_mem=clear_mem,
            **kwargs,
        )
        if results is not None and oneout and len(results) == 1:
            results = next(iter(results.values()))
        del model_data

        # finalize models:
        if clear_mem or finalize:
            self.print("\n")
            mlist.finalize(self, self.verbosity - 1)
            self.finalize(clear_mem=clear_mem)

        return results

    def _collect_point_models(
        self,
        calc_parameters: dict[str, dict[str, Any]] = {},
        point_models: Any = None,
        ambient: bool = False,
    ) -> tuple[PointDataModelList, list[dict[str, Any]]]:
        """
        Helper function that creates model list
        """
        # prepare:
        calc_pars = []
        mlist = PointDataModelList(models=[])

        # prepare extra eval models:
        emodels: list[PointDataModel] = []
        emodels_cpars = []
        if point_models is not None:
            if not isinstance(point_models, list):
                point_models = [point_models]
            for m in point_models:
                if isinstance(m, str):
                    pname = m
                    pmodel = self.mbook.point_models[pname]
                    pmodel.name = pname
                    emodels.append(pmodel)
                elif isinstance(m, PointDataModel):
                    emodels.append(m)
                else:
                    raise TypeError(f"Model '{m}' is neither str nor PointDataModel")
                emodels_cpars.append(calc_parameters.get(emodels[-1].name, {}))
        emodel_list = PointDataModelList(models=emodels)

        # 0) calculate states results:
        mlist.models.append(self.states)
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        # 1) calculate ambient extra eval point results:
        mlist.models.append(emodel_list)
        calc_pars.append({"parameters": emodels_cpars})

        # 2) transfer ambient results:
        mlist.models.append(self.get_model("SetAmbPointResults")())
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        # 3) calc wake effects:
        if not ambient:
            mlist.models.append(
                self.get_model("PointWakesCalculation")(emodel_list, emodels_cpars)
            )
            calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        return mlist, calc_pars

    def _launch_parallel_points_calc(
        self,
        mlist: PointDataModelList,
        *data: Dataset,
        output: Output | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Calculate point chunks and expose collected public point dimensions.

        Parameters
        ----------
        mlist
            Initialized scientific point-model chain.
        data
            Model, farm and target input datasets passed to the base algorithm.
        output
            Initialized output model or chain; ``None`` disables collection.
        kwargs
            Additional base-algorithm engine options.

        Returns
        -------
        results
            Engine-owned output mapping, modified in place only at
            ``FC.POINT_RESULTS`` when present. Its dataset selects target point
            zero and drops ``FC.TPOINT``. Any retained ``FC.TARGET`` dimension
            becomes ``FC.POINT``, preserving supplied point order. Spatial/state
            aggregation may remove the corresponding dimensions.

        """
        # Collect target chunks through the same Output lifecycle as farm chunks.
        results = super()._launch_parallel_points_calc(
            mlist, *data, output=output, **kwargs
        )
        if FC.POINT_RESULTS in results:
            point_results = results[FC.POINT_RESULTS].sel({FC.TPOINT: 0}, drop=True)
            if FC.TARGET in point_results.dims:
                point_results = point_results.rename({FC.TARGET: FC.POINT})
            results[FC.POINT_RESULTS] = point_results
        return results

    def calc_points(
        self,
        farm_results: Dataset | None,
        points: np.ndarray | None = None,
        outputs: str | Output | list[str | Output] | None = "point_results",
        point_models: Any = None,
        calc_parameters: dict[str, dict[str, Any]] = {},
        persist_pdata: bool = False,
        finalize: bool = True,
        ambient: bool = False,
        states_sel: list[Any] | None = None,
        clear_mem: bool = False,
        point_results: Dataset | None = None,
        **kwargs: Any,
    ) -> Dataset | dict[str, Any]:
        """
        Calculate data at a given set of points.

        Parameters
        ----------
        farm_results
            Farm fields on ``(FC.STATE, FC.TURBINE)`` in original turbine
            order. Required for waked scientific calculations; ``None`` is
            accepted for ambient calculations or precomputed point results.
        points
            Static points with shape ``(n_points, 3)`` or state-dependent
            points with shape ``(n_states, n_points, 3)``. Static coordinates
            are broadcast to each engine state chunk. Components are ``x``,
            ``y`` and height in metres. Mutually exclusive with
            ``point_results``; one of them must be provided. Supplied point
            order is preserved.
        outputs
            An ``Output`` instance, a name from ``ModelBook.outputs``, or a list
            of instances and names. A single output with one result returns that
            result directly. Multiple outputs return a result-name mapping.
            ``None`` or an empty list disables collection.
            ``get_calc_points`` is ignored: supplied points or point results
            determine the calculation locations.
        point_models
            Additional point-model instances or ``ModelBook.point_models``
            names, supplied singly or in execution-order lists.
        calc_parameters
            Model-name-to-keyword-parameter mapping for scientific calculation.
        persist_pdata
            Persist the newly constructed point-input dataset before execution.
            Has no effect when ``point_results`` is supplied.
        finalize
            Finalize initialized models and the algorithm after calculation.
        ambient
            Skip wake effects and calculate ambient results only.
        states_sel
            Unique state-coordinate labels, in requested result order.
            ``None`` retains all states. Positional ``states_isel`` is rejected.
        clear_mem
            Remove loaded model data when launching the run and finalize
            afterward, even if ``finalize`` is false.
        point_results
            Precomputed fields on ``(FC.STATE, FC.POINT)``; scientific point
            calculation is skipped. Locations must be present as ``FC.TARGETS``
            or the metre-valued ``FV.X``, ``FV.Y`` and ``FV.H`` fields.
            Input arrays are read, not modified.
        kwargs
            Additional base-algorithm engine options.

        Returns
        -------
        results
            The sole result of a single selected output, otherwise a
            result-name-to-value mapping. Disabled collection gives an empty
            mapping. The default dataset has ``(FC.STATE, FC.POINT)`` physical
            fields and location coordinates in supplied point order.

        Raises
        ------
        KeyError
            If an output model-book name is unavailable.
        TypeError
            If an output selector or point model is unsupported, or
            ``states_isel`` is supplied.
        ValueError
            If neither or both point inputs are supplied, a waked calculation
            lacks farm results, precomputed locations are missing, selected
            state labels are duplicated, or unlabelled model data cannot be
            aligned with the farm state count.

        Notes
        -----
        Scientific models and outputs participate in the ordinary running and
        unsetting lifecycle. No output-requested replacement grid is generated.

        """
        # initialize:
        if not self.initialized:
            self.initialize()

        # Resolve output names and preserve the single-versus-multiple return contract.
        output, oneout = self._resolve_outputs(outputs)

        # Initialize the resolved output chain before preparing model inputs.
        if output is not None:
            self.init_model(output)

        if not ambient and farm_results is None and point_results is None:
            raise ValueError(
                f"Cannot calculate point results without farm results for ambient = {ambient}"
            )
        if farm_results is None:
            farm_results = Dataset(
                coords={
                    FC.STATE: self.states.index(),
                    FC.TURBINE: np.arange(self.n_turbines),
                }
            )

        # collect models and initialize:
        if point_results is None:
            mlist, calc_pars = self._collect_point_models(
                calc_parameters, point_models, ambient
            )
        else:
            mlist, calc_pars = PointDataModelList(), []

        # initialize models:
        if not mlist.initialized:
            self.init_model(mlist)
            self._calc_point_vars(mlist)
            if point_results is not None:
                self.point_vars = list(point_results.data_vars)
            if farm_results is None:
                self.farm_vars = []
            else:
                self.farm_vars = sorted(
                    [
                        v
                        for v in farm_results.data_vars.keys()
                        if farm_results[v].dims == (FC.STATE, FC.TURBINE)
                        or v == FV.WEIGHT
                    ]
                )

        # welcome:
        if (point_results is None) == (points is None):
            raise ValueError(
                f"{self.name}: Provide exactly one of points or point_results"
            )
        elif point_results is not None:
            n_points = point_results.sizes[FC.POINT]
        else:
            points = np.asarray(points)
            n_points = points.shape[-2]
        self.print_deco("calc_points", n_points=n_points, outputs=output)
        self._print_model_oder(mlist, calc_pars, output)

        if "states_isel" in kwargs:
            raise TypeError(
                "'states_isel' is not supported by calc_points; use state labels "
                "with 'states_sel'"
            )

        # subset selections:
        state_index = farm_results.get_index(FC.STATE)
        sel = {} if states_sel is None else {FC.STATE: states_sel}
        if states_sel is not None:
            if not state_index.is_unique:
                raise ValueError(
                    "Cannot select states by label because the state coordinate "
                    "contains duplicates"
                )
            farm_results = farm_results.sel(sel)
            if not farm_results.get_index(FC.STATE).is_unique:
                raise ValueError("State selection contains duplicate labels")
        n_states = farm_results.sizes[FC.STATE]

        # get input model data:
        model_data, extra_data = self.get_model_data(pop=clear_mem)
        if FC.STATE in model_data.dims and FC.STATE not in model_data.coords:
            if model_data.sizes[FC.STATE] != len(state_index):
                raise ValueError(
                    "Cannot attach state labels to model data with a different "
                    "state count"
                )
            model_data = model_data.assign_coords({FC.STATE: state_index})
        self.print("\nInput model data:\n\n", model_data, "\n")
        if len(extra_data) > 0:
            self.print("\nExtra data:")
            for v, d in extra_data.items():
                self.print(f"  {v}: {type(d).__name__}")

        # chunk farm results:
        self.print("\nInput farm data:\n\n", farm_results, "\n")

        # get point data:
        if point_results is None:
            if FC.STATE in farm_results.coords:
                sinds = farm_results.coords[FC.STATE]
            elif model_data is not None and FC.STATE in model_data.coords:
                sinds = model_data.coords[FC.STATE]
            else:
                sinds = None
            point_data = self.new_point_data(points, sinds, n_states=n_states)
            if persist_pdata:
                point_data = point_data.persist()
        else:
            point_data = self._prepare_point_results(point_results)
        self.print("\nInput point data:\n\n", point_data, "\n")
        self.print("\nFarm variables:", ", ".join(self.farm_vars))
        self.print("\nPoint variables:", ", ".join(self.point_vars))

        # calculate:
        results = super().calc_points(
            mlist,
            model_data,
            farm_results,
            point_data,
            extra_data=extra_data,
            output=output,
            parameters=calc_pars,
            sel=sel,
            **kwargs,
        )
        if results is not None and oneout and len(results) == 1:
            results = next(iter(results.values()))
        del model_data, farm_results, point_data

        # finalize models:
        if clear_mem or finalize:
            self.print("\n")
            mlist.finalize(self, self.verbosity - 1)
            self.finalize(clear_mem=clear_mem)

        return results

    def _prepare_point_results(self, point_results: Dataset) -> Dataset:
        """Adapt precomputed point fields without changing their point order.

        Parameters
        ----------
        point_results
            Public dataset with ``FC.STATE`` and ``FC.POINT`` axes, and either
            ``FC.TARGETS`` or ``FV.X``, ``FV.Y`` and ``FV.H`` in metres.
            Coordinate and physical fields may broadcast across states or points.

        Returns
        -------
        point_data
            Dataset using ``FC.TARGET`` instead of ``FC.POINT``. When locations
            are constructed from coordinate fields, physical variables have
            ``(FC.STATE, FC.TARGET, FC.TPOINT)`` dimensions with one target
            point. The caller's dataset is not mutated.

        Raises
        ------
        ValueError
            If ``FC.TARGETS`` is absent and any required coordinate field is
            missing.
        """
        # Require actual locations rather than generating replacement points.
        if FC.TARGETS in point_results:
            data = point_results.rename({FC.POINT: FC.TARGET})
        else:
            coordinates = (FV.X, FV.Y, FV.H)
            missing = [v for v in coordinates if v not in point_results]
            if missing:
                raise ValueError(
                    f"{self.name}: Precomputed point results require coordinates "
                    f"{missing}"
                )
            points = np.stack(
                [
                    point_results[v]
                    .broadcast_like(point_results)
                    .transpose(FC.STATE, FC.POINT)
                    .to_numpy()
                    for v in coordinates
                ],
                axis=-1,
            )
            data = self.new_point_data(points, point_results[FC.STATE])
            template = Dataset(
                coords={
                    FC.STATE: point_results[FC.STATE].to_numpy(),
                    FC.POINT: point_results[FC.POINT].to_numpy(),
                }
            )
            for variable in point_results.data_vars:
                values = (
                    point_results[variable]
                    .reset_coords(drop=True)
                    .broadcast_like(template)
                    .transpose(FC.STATE, FC.POINT)
                )
                data[variable] = values.rename({FC.POINT: FC.TARGET}).expand_dims(
                    {FC.TPOINT: [0]}, axis=-1
                )
        return data

    def calc_outputs(
        self,
        outputs: str | Output | list[str | Output] | None = "point_results",
        farm_results: Dataset | None = None,
        point_results: Dataset | None = None,
        calc_parameters: dict[str, dict[str, Any]] = {},
        finalize: bool = True,
        states_sel: list[Any] | None = None,
        clear_mem: bool = False,
        **kwargs: Any,
    ) -> Dataset | dict[str, Any]:
        """
        Run selected outputs on supplied results or their requested points.

        Parameters
        ----------
        outputs
            An ``Output`` instance, a name from ``ModelBook.outputs``, or a list
            of instances and names. Point-dependent children in a composed
            output must request identical calculation points.
        farm_results
            Precomputed farm fields in original turbine order on
            ``(FC.STATE, FC.TURBINE)``. These are collected without rerunning farm
            physics. Missing state coordinates are filled from ``self.states``.
            If absent and an output requests points, all required farm fields
            are calculated first.
        point_results
            Precomputed ``(FC.STATE, FC.POINT)`` fields with location coordinates
            as required by ``calc_points``. These take precedence over requested
            grids and are consumed without scientific point recalculation.
        calc_parameters
            Model-name-to-keyword-parameter mapping forwarded to point
            calculations when point evaluation is needed.
        finalize
            Finalize initialized models and the algorithm after execution.
        states_sel
            State-coordinate labels forwarded to collection or point execution.
            ``None`` retains all states; point execution requires unique labels.
        clear_mem
            Remove loaded model data for execution and finalize afterward,
            even if ``finalize`` is false.
        kwargs
            Additional engine or point-calculation options.

        Returns
        -------
        results
            The sole result of a single selected output, otherwise a
            result-name-to-value mapping. ``None`` disables collection and
            returns an empty mapping. Without supplied data or requested points,
            only the output's parent-side ``finalize_results`` hook is run.

        Raises
        ------
        KeyError
            If an output model-book name is unavailable.
        TypeError
            If ``outputs`` is not a supported selector.
        ValueError
            If composed outputs request different grids or supplied point
            results violate the ``calc_points`` contract.

        Notes
        -----
        Constructor configuration remains on output models; chunk buffers and
        assembled values belong to the run's explicit result mappings.

        """
        # initialize:
        if not self.initialized:
            self.initialize()

        # Resolve output names and preserve the single-versus-multiple return contract.
        output, oneout = self._resolve_outputs(outputs)
        if output is None:
            return {}

        # Initialize the resolved output chain before preparing model inputs.
        if output is not None:
            self.init_model(output)

        # welcome:
        points = output.get_calc_points(self) if point_results is None else None
        if farm_results is not None and FC.STATE not in farm_results.sizes:
            farm_results = farm_results.assign_coords({FC.STATE: self.states.index()})
        if farm_results is None and point_results is not None:
            kwargs.setdefault("ambient", True)
            return self.calc_points(
                farm_results=None,
                points=points,
                point_results=point_results,
                outputs=output,
                calc_parameters=calc_parameters,
                finalize=finalize,
                states_sel=states_sel,
                clear_mem=clear_mem,
                **kwargs,
            )
        if farm_results is None and points is not None:
            farm_results = self.calc_farm(outputs="farm_results_all", finalize=False)

        # run calculation:
        if farm_results is not None and points is None and point_results is None:
            # Run only output collection on already farm-ordered arrays.
            mlist = FarmDataModelList()
            self.init_model(mlist)
            model_data, extra_data = self.get_model_data(pop=clear_mem)
            results = super().calc_farm(
                mlist,
                model_data,
                farm_data=farm_results,
                extra_data=extra_data,
                output=output,
                sel=None if states_sel is None else {FC.STATE: states_sel},
                clear_mem=clear_mem,
                **kwargs,
            )
        elif farm_results is not None and (
            points is not None or point_results is not None
        ):
            results = self.calc_points(
                farm_results=farm_results,
                points=points,
                point_results=point_results,
                outputs=output,
                calc_parameters=calc_parameters,
                finalize=finalize,
                states_sel=states_sel,
                clear_mem=clear_mem,
                **kwargs,
            )
            return results
        else:
            self.print(f"{self.name}: Computing finalize_results for output {output}")
            results = Dict[str, Any](_name="output_results")
            output.finalize_results(
                algo=self,
                n_chunks_states=1,
                n_chunks_targets=1,
                output_results=results,
            )

        if clear_mem or finalize:
            self.finalize(clear_mem=clear_mem)
        if results is not None and oneout and len(results) == 1:
            results = next(iter(results.values()))

        return results
