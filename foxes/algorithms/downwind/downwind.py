from __future__ import annotations
# mypy: disable-error-code=override

import numpy as np
from xarray import Dataset
from typing import TYPE_CHECKING, Any

from foxes.core import Algorithm, FarmDataModelList
from foxes.core import PointDataModel, PointDataModelList, FarmController
from foxes.config import config
import foxes.models as fm
import foxes.variables as FV
import foxes.constants as FC

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

    DEFAULT_FARM_OUTPUTS = [
        FV.X,
        FV.Y,
        FV.H,
        FV.D,
        FV.AMB_WD,
        FV.AMB_REWS,
        FV.AMB_TI,
        FV.AMB_RHO,
        FV.AMB_CT,
        FV.AMB_P,
        FV.WD,
        FV.REWS,
        FV.YAW,
        FV.TI,
        FV.CT,
        FV.P,
    ]

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

        pop_model = self.population_model
        assert pop_model is not None
        POP = pop_model.index_coord
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
        self, func_name: str | None = None, n_points: int | None = None
    ) -> None:
        """
        Helper function for printing model names

        Parameters
        ----------
        func_name
            Name of the calling function
        n_points
            The number of points

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
            print()

    def _print_model_oder(
        self,
        mlist: FarmDataModelList,
        calc_pars: list[dict[str, Any]],
    ) -> None:
        """
        Helper function for printing model names
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

    def _collect_farm_models(
        self,
        outputs: list[str] | bool | None,
        calc_parameters: dict[str, dict[str, Any]],
        ambient: bool,
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

        # 6) reorder back to state-turbine dimensions:
        if not isinstance(outputs, bool) or outputs:
            mlist.models.append(self.get_model("ReorderFarmOutput")(outputs))
            calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        return mlist, calc_pars

    def _calc_farm_vars(self, mlist: FarmDataModelList) -> None:
        """Helper function that gathers the farm variables"""
        self.farm_vars = sorted(list(mlist.output_farm_vars(self)))

    def _launch_parallel_farm_calc(
        self,
        mlist: FarmDataModelList,
        model_data: Dataset,
        outputs: list[str] | None = None,
        normalize: bool = False,
        **kwargs: Any,
    ) -> Dataset:
        """
        Runs the main calculation, launching parallelization

        Parameters
        ----------
        mlist
            The model list
        model_data
            The initial model data
        outputs
            The output variables, or None for defaults
        normalize
            Normalize the weights to 1 wrt sum over states
        kwargs
            Additional parameters for running

        Returns
        -------
        farm_results
            The farm results. The calculated variables have
            dimensions (state, turbine)

        """
        out_vars = self.farm_vars if outputs is None else outputs
        farm_results = super()._launch_parallel_farm_calc(
            mlist, model_data, out_vars=out_vars, **kwargs
        )

        if normalize:
            farm_results[FV.WEIGHT] /= farm_results[FV.WEIGHT].sum(dim=FC.STATE)

        return farm_results

    def calc_farm(
        self,
        outputs: list[str] | str | None = None,
        calc_parameters: dict[str, dict[str, Any]] = {},
        ambient: bool = False,
        finalize: bool = True,
        clear_mem: bool = False,
        **kwargs: Any,
    ) -> Dataset:
        """
        Calculate farm data.

        Parameters
        ----------
        calc_parameters
            Parameters for model calculation.
            Key: model name str, value: parameter dict
        outputs
            The output variables, or None for defaults
        ambient
            Flag for ambient instead of waked calculation
        finalize
            Flag for finalization after calculation
        clear_mem
            Clear idata memory after starting the run
        kwargs
            Additional parameters for run_calculation

        Returns
        -------
        farm_results
            The farm results. The calculated variables have
            dimensions (state, turbine)

        """
        # initialize algorithm:
        if not self.initialized:
            self.initialize()

        # welcome:
        self.print_deco("calc_farm")

        # collect models:
        if outputs == "default":
            outputs = self.DEFAULT_FARM_OUTPUTS
        if isinstance(outputs, str):
            outputs = [outputs]
        mlist, calc_pars = self._collect_farm_models(outputs, calc_parameters, ambient)

        # initialize models:
        if not mlist.initialized:
            mlist.initialize(self, verbosity=self.verbosity - 1)
            self._calc_farm_vars(mlist)
        self._print_model_oder(mlist, calc_pars)

        # update outputs:
        if outputs is None:
            outputs = self.farm_vars
        else:
            outputs = sorted(list(set(outputs).intersection(self.farm_vars)))

        # get input model data:
        model_data, extra_data = self.get_model_data(pop=clear_mem)
        self.print("\nInput data:\n\n", model_data, "\n")
        if len(extra_data) > 0:
            self.print("Extra data:")
            for v, d in extra_data.items():
                self.print(f"  {v}: {type(d).__name__}")
        self.print("\nFarm variables:", ", ".join(self.farm_vars))
        self.print("\nOutput variables:", ", ".join(outputs))

        # run main calculation:
        farm_results = super().calc_farm(
            mlist,
            model_data,
            extra_data=extra_data,
            parameters=calc_pars,
            outputs=outputs,
            clear_mem=clear_mem,
            **kwargs,
        )
        if farm_results is not None:
            farm_results[FC.TNAME] = ((FC.TURBINE,), self.farm.turbine_names)
            for v in [FV.ORDER, FV.ORDER_SSEL, FV.ORDER_INV]:
                if v in farm_results:
                    farm_results[v] = farm_results[v].astype(config.dtype_int)
        del model_data

        # finalize models:
        if clear_mem or finalize:
            self.print("\n")
            mlist.finalize(self, self.verbosity - 1)
            self.finalize(clear_mem=clear_mem)

        if ambient and farm_results:
            dvars = [v for v in farm_results.data_vars.keys() if v in FV.var2amb]
            farm_results = farm_results.drop_vars(dvars)

        return farm_results

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
        outputs: list[str] | None = None,
        **kwargs: Any,
    ) -> Dataset:
        """
        Runs the main points calculation, launching parallelization

        Parameters
        ----------
        mlist
            The model list
        data
            The (mdata, fdata) inputs
        outputs
            The output variables, or None for defaults
        kwargs
            Additional parameters for running

        Returns
        -------
        point_results
            The point results. The calculated variables have
            dimensions (state, point)

        """
        return (
            super()
            ._launch_parallel_points_calc(mlist, *data, out_vars=outputs, **kwargs)
            .sel({FC.TPOINT: 0})
            .rename({FC.TARGET: FC.POINT})
        )

    def calc_points(
        self,
        farm_results: Dataset,
        points: np.ndarray,
        point_models: Any = None,
        outputs: list[str] | None = None,
        calc_parameters: dict[str, dict[str, Any]] = {},
        persist_pdata: bool = False,
        finalize: bool = True,
        ambient: bool = False,
        chunked_results: bool = False,
        states_sel: list[Any] | None = None,
        states_isel: list[int] | None = None,
        clear_mem: bool = False,
        **kwargs: Any,
    ) -> Dataset:
        """
        Calculate data at a given set of points.

        Parameters
        ----------
        farm_results
            The farm results. The calculated variables have
            dimensions (state, turbine)
        points
            The points of interest, shape: (n_states, n_points, 3)
        outputs
            The output variables, or None for defaults
        point_models
            Additional point models to be executed
        calc_parameters
            Parameters for model calculation.
            Key: model name str, value: parameter dict
        persist_pdata
            Switch for forcing dask to load all farm data
            into memory
        finalize
            Flag for finalization after calculation
        ambient
            Flag for ambient instead of waked calculation
        chunked_results
            Flag for chunked results
        states_sel
            Reduce to selected states
        states_isel
            Reduce to the selected states indices
        clear_mem
            Clear idata memory after starting the run
        kwargs
            Additional parameters for run_calculation

        Returns
        -------
        point_results
            The point results. The calculated variables have
            dimensions (state, point)

        """
        if not self.initialized:
            self.initialize()
        if not ambient and farm_results is None:
            raise ValueError(
                f"Cannot calculate point results without farm results for ambient = {ambient}"
            )

        # welcome:
        points = np.asarray(points)
        self.print_deco("calc_points", n_points=points.shape[1])

        # collect models and initialize:
        mlist, calc_pars = self._collect_point_models(
            calc_parameters, point_models, ambient
        )

        # initialize models:
        if not mlist.initialized:
            mlist.initialize(self, self.loaded_data, verbosity=self.verbosity - 1)

        # subset selections:
        sel = {} if states_sel is None else {FC.STATE: states_sel}
        isel = {} if states_isel is None else {FC.STATE: states_isel}
        if states_isel is not None:
            farm_results = farm_results.isel(isel)
        if states_sel is not None:
            farm_results = farm_results.sel(sel)
        n_states = farm_results.sizes[FC.STATE]

        # get input model data:
        model_data, extra_data = self.get_model_data(pop=clear_mem)
        self.print("\nInput data:\n\n", model_data, "\n")
        if len(extra_data) > 0:
            self.print("\nExtra data:")
            for v, d in extra_data.items():
                self.print(f"  {v}: {type(d).__name__}")
        self.print("\nOutput farm variables:", ", ".join(self.farm_vars))

        # chunk farm results:
        self.print("\nInput farm data:\n\n", farm_results, "\n")

        # get point data:
        if FC.STATE in farm_results.coords:
            sinds = farm_results.coords[FC.STATE]
        elif model_data is not None and FC.STATE in model_data.coords:
            sinds = model_data.coords[FC.STATE]
        else:
            sinds = None
        point_data = self.new_point_data(points, sinds, n_states=n_states)
        if persist_pdata:
            point_data = point_data.persist()
        self.print("\nInput point data:\n\n", point_data, "\n")

        # check vars:
        ovars = mlist.output_point_vars(self) if outputs is None else outputs
        self.print("\nOutput point variables:", ", ".join(ovars))

        # calculate:
        point_results = super().calc_points(
            mlist,
            model_data,
            farm_results,
            point_data,
            extra_data=extra_data,
            outputs=ovars,
            parameters=calc_pars,
            sel=sel,
            isel=isel,
            **kwargs,
        )
        del model_data, farm_results, point_data

        # finalize models:
        if finalize:
            self.print("\n")
            mlist.finalize(self, self.verbosity - 1)
            self.finalize()

        if ambient:
            dvars = [v for v in point_results.data_vars.keys() if v in FV.var2amb]
            point_results = point_results.drop_vars(dvars)

        if chunked_results:
            point_results = self.chunked(point_results)

        return point_results
