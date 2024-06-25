from foxes.core import Algorithm, FarmDataModelList
from foxes.core import PointDataModel, PointDataModelList, FarmController
import foxes.models as fm
import foxes.variables as FV
import foxes.constants as FC
from . import models as mdls


class Downwind(Algorithm):
    """
    The downwind algorithm.

    The turbines are evaluated once, in the order
    that is calculated by the provided `TurbineOrder`
    object.

    Attributes
    ----------
    states: foxes.core.States
        The ambient states
    wake_models: dict
        The wake models. Key: wake model name,
        value: foxes.core.WakeModel
    rotor_model: foxes.core.RotorModel
        The rotor model, for all turbines
    wake_frame: foxes.core.WakeFrame
        The wake frame
    partial_wakes: dict
        The partial wakes mapping. Key: wake model name,
        value: foxes.core.PartialWakesModel
    ground_models: dict
        The ground models mapping. Key: wake model name,
        value: foxes.core.GroundModel
    farm_controller: foxes.core.FarmController
        The farm controller
    n_states: int
        The number of states

    :group: algorithms.downwind

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
        FV.AMB_P,
        FV.WD,
        FV.REWS,
        FV.YAW,
        FV.TI,
        FV.CT,
        FV.P,
        FV.ORDER,
        FV.WEIGHT,
    ]

    def __init__(
        self,
        farm,
        states,
        wake_models,
        rotor_model="centre",
        wake_frame="rotor_wd",
        partial_wakes=None,
        ground_models=None,
        farm_controller="basic_ctrl",
        chunks={FC.STATE: 1000, FC.POINT: 4000},
        mbook=None,
        dbook=None,
        verbosity=1,
    ):
        """
        Constructor.

        Parameters
        ----------
        farm: foxes.WindFarm
            The wind farm
        states: foxes.core.States
            The ambient states
        wake_models: list of str
            The wake models, applied to all turbines.
            Will be looked up in the model book
        rotor_model: str
            The rotor model, for all turbines. Will be
            looked up in the model book
        wake_frame: str
            The wake frame. Will be looked up in the
            model book
        partial_wakes: dict, list or str, optional
            The partial wakes mapping. Key: wake model name,
            value: partial wake model name
        ground_models: dict, list or str, optional
            The ground models mapping. Key: wake model name,
            value: ground model name
        farm_controller: str
            The farm controller. Will be
            looked up in the model book
        chunks: dict
            The chunks choice for running in parallel with dask,
            e.g. `{"state": 1000}` for chunks of 1000 states
        mbook: foxes.ModelBook, optional
            The model book
        dbook: foxes.DataBook, optional
            The data book, or None for default
        verbosity: int
            The verbosity level, 0 means silent

        """
        if mbook is None:
            mbook = fm.ModelBook()

        super().__init__(mbook, farm, chunks, verbosity, dbook)

        self.states = states
        self.n_states = None
        self.states_data = None

        self.rotor_model = self.mbook.rotor_models[rotor_model]
        self.rotor_model.name = rotor_model

        self.wake_frame = self.mbook.wake_frames[wake_frame]
        self.wake_frame.name = wake_frame

        self.wake_models = {}
        for w in wake_models:
            m = self.mbook.wake_models[w]
            m.name = w
            self.wake_models[w] = m

        def _set_wspecific(descr, target, values, deffunc, mbooks, checkw):
            if values is None:
                values = {}
            if isinstance(values, list) and len(values) == 1:
                values = values[0]
            if isinstance(values, str):
                for w in wake_models:
                    try:
                        pw = values
                        if checkw:
                            mbooks[pw].check_wmodel(self.wake_models[w], error=True)
                    except TypeError:
                        pw = deffunc(self.wake_models[w])
                    target[w] = mbooks[pw]
                    target[w].name = pw
            elif isinstance(values, list):
                for i, w in enumerate(wake_models):
                    if i >= len(values):
                        raise IndexError(
                            f"Not enough {descr} in list {values}, expecting {len(wake_models)}"
                        )
                    pw = values[i]
                    target[w] = mbooks[pw]
                    target[w].name = pw
            else:
                for w in wake_models:
                    if w in values:
                        pw = values[w]
                    else:
                        pw = deffunc(self.wake_models[w])
                    target[w] = mbooks[pw]
                    target[w].name = pw

        self.partial_wakes = {}
        _set_wspecific(
            descr="partial wakes",
            target=self.partial_wakes,
            values=partial_wakes,
            deffunc=mbook.default_partial_wakes,
            mbooks=self.mbook.partial_wakes,
            checkw=True,
        )

        self.ground_models = {}
        _set_wspecific(
            descr="ground models",
            target=self.ground_models,
            values=ground_models,
            deffunc=lambda w: "no_ground",
            mbooks=self.mbook.ground_models,
            checkw=False,
        )

        self.farm_controller = self.mbook.farm_controllers[farm_controller]
        self.farm_controller.name = farm_controller

    @classmethod
    def get_model(cls, name):
        """
        Get the algorithm specific model

        Parameters
        ----------
        name: str
            The model name

        Returns
        -------
        model: foxes.core.model
            The model

        """
        return getattr(mdls, name)

    def _print_deco(self, func_name, n_points=None):
        """
        Helper function for printing model names
        """
        if self.verbosity > 0:
            deco = "-" * 50
            print(f"\n{deco}")
            print(f"  Running {self.name}: {func_name}")
            print(deco)
            print(f"  n_states : {self.n_states}")
            print(f"  n_turbines: {self.n_turbines}")
            if n_points is not None:
                print(f"  n_points : {n_points}")
            print(deco)
            print(f"  states   : {self.states}")
            print(f"  rotor    : {self.rotor_model}")
            print(f"  controller: {self.farm_controller}")
            print(f"  wake frame: {self.wake_frame}")
            print(deco)
            print(f"  wakes:")
            for i, w in enumerate(self.wake_models.values()):
                print(f"    {i}) {w.name}: {w}")
            print(deco)
            print(f"  partial wakes:")
            for i, (w, p) in enumerate(self.partial_wakes.items()):
                print(f"    {i}) {w}: {p.name}, {p}")
            print(deco)
            print(f"  turbine models:")
            for i, m in enumerate(self.farm_controller.pre_rotor_models.models):
                print(f"    {i}) {m.name}: {m} [pre-rotor]")
            for i, m in enumerate(self.farm_controller.post_rotor_models.models):
                print(
                    f"    {i+len(self.farm_controller.pre_rotor_models.models)}) {m.name}: {m}"
                )
            print(deco)
            print()

    def _print_model_oder(self, mlist, calc_pars):
        """
        Helper function for printing model names
        """
        if self.verbosity > 0:
            deco = "-" * 50
            print(f"\n{deco}")
            print(f"  Model oder")
            print(f"{deco}")

            for i, m in enumerate(mlist.models):
                print(f"{i:02d}) {m.name}")
                if isinstance(m, FarmController):
                    if calc_pars[i]["pre_rotor"]:
                        for j, mm in enumerate(m.pre_rotor_models.models):
                            print(f"  {i:02d}.{j}) Pre-rotor: {mm.name}")
                    else:
                        for j, mm in enumerate(m.post_rotor_models.models):
                            print(f"  {i:02d}.{j}) Post-rotor: {mm.name}")

            print(deco)
            print()

    def init_states(self):
        """
        Initialize states, if needed.
        """
        if not self.states.initialized:
            self.states.initialize(self, self.verbosity)
            self.n_states = self.states.size()

    def all_models(self, with_states=True):
        """
        Return all models

        Parameters
        ----------
        with_states: bool
            Flag for including states

        Returns
        -------
        mdls: list of foxes.core.Model
            The list of models

        """
        mdls = [self.states] if with_states else []
        mdls += [
            self.rotor_model,
            self.farm_controller,
            self.wake_frame,
        ]
        mdls += list(self.wake_models.values())
        mdls += list(self.partial_wakes.values())

        return mdls

    def initialize(self):
        """
        Initializes the algorithm.
        """
        self.print(f"\nInitializing algorithm '{self.name}'")
        super().initialize()

        self.init_states()

        for m in self.all_models(with_states=False):
            m.initialize(self, self.verbosity)

    def _collect_farm_models(
        self,
        outputs,
        calc_parameters,
        ambient,
    ):
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
        calc_pars[-1].update(
            {"store_rpoints": True, "store_rweights": True, "store_amb_res": True}
        )

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
        if outputs != False:
            mlist.models.append(self.get_model("ReorderFarmOutput")(outputs))
            calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        return mlist, calc_pars

    def _calc_farm_vars(self, mlist):
        """Helper function that gathers the farm variables"""
        self.farm_vars = sorted(list(set([FV.WEIGHT] + mlist.output_farm_vars(self))))

    def _run_farm_calc(self, mlist, *data, outputs=None, **kwargs):
        """Helper function for running the main farm calculation"""
        self.print(
            f"\nCalculating {self.n_states} states for {self.n_turbines} turbines"
        )
        out_vars = self.farm_vars if outputs is None else outputs
        farm_results = mlist.run_calculation(self, *data, out_vars=out_vars, **kwargs)
        farm_results[FC.TNAME] = ((FC.TURBINE,), self.farm.turbine_names)
        for v in [FV.ORDER, FV.ORDER_SSEL, FV.ORDER_INV]:
            if v in farm_results:
                farm_results[v] = farm_results[v].astype(FC.ITYPE)

        return farm_results

    def calc_farm(
        self,
        outputs=None,
        calc_parameters={},
        persist=True,
        finalize=True,
        ambient=False,
        chunked_results=False,
        **kwargs,
    ):
        """
        Calculate farm data.

        Parameters
        ----------
        calc_parameters: dict
            Parameters for model calculation.
            Key: model name str, value: parameter dict
        outputs: list of str, optional
            The output variables, or None for defaults
        persist: bool
            Switch for forcing dask to load all model data
            into memory
        finalize: bool
            Flag for finalization after calculation
        ambient: bool
            Flag for ambient instead of waked calculation
        chunked_results: bool
            Flag for chunked results
        kwargs: dict, optional
            Additional parameters for run_calculation

        Returns
        -------
        farm_results: xarray.Dataset
            The farm results. The calculated variables have
            dimensions (state, turbine)

        """
        # initialize algorithm:
        if not self.initialized:
            self.initialize()

        # welcome:
        self._print_deco("calc_farm")

        # collect models:
        if outputs == "default":
            outputs = self.DEFAULT_FARM_OUTPUTS
        mlist, calc_pars = self._collect_farm_models(outputs, calc_parameters, ambient)

        # initialize models:
        if not mlist.initialized:
            mlist.initialize(self, self.verbosity)
            self._calc_farm_vars(mlist)
        self._print_model_oder(mlist, calc_pars)

        # update outputs:
        if outputs is None:
            outputs = self.farm_vars
        else:
            outputs = sorted(list(set(outputs).intersection(self.farm_vars)))

        # get input model data:
        models_data = self.get_models_data()
        if persist:
            models_data = models_data.persist()
        self.print("\nInput data:\n\n", models_data, "\n")
        self.print(f"\nFarm variables:", ", ".join(self.farm_vars))
        self.print(f"\nOutput variables:", ", ".join(outputs))
        self.print(f"\nChunks: {self.chunks}\n")

        # run main calculation:
        farm_results = self._run_farm_calc(
            mlist,
            models_data,
            parameters=calc_pars,
            outputs=outputs,
            **kwargs,
        )
        del models_data

        # finalize models:
        if finalize:
            self.print("\n")
            mlist.finalize(self, self.verbosity)
            self.finalize()

        if ambient:
            dvars = [v for v in farm_results.data_vars.keys() if v in FV.var2amb]
            farm_results = farm_results.drop_vars(dvars)

        if chunked_results:
            farm_results = self.chunked(farm_results)

        return farm_results

    def _collect_point_models(
        self,
        calc_parameters={},
        point_models=None,
        ambient=False,
    ):
        """
        Helper function that creates model list
        """
        # prepare:
        calc_pars = []
        mlist = PointDataModelList(models=[])

        # prepare extra eval models:
        emodels = []
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
        emodels = PointDataModelList(models=emodels)

        # 0) calculate states results:
        mlist.models.append(self.states)
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        # 1) calculate ambient extra eval point results:
        mlist.models.append(emodels)
        calc_pars.append({"parameters": emodels_cpars})

        # 2) transfer ambient results:
        mlist.models.append(self.get_model("SetAmbPointResults")())
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        # 3) calc wake effects:
        if not ambient:
            mlist.models.append(
                self.get_model("PointWakesCalculation")(emodels, emodels_cpars)
            )
            calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        return mlist, calc_pars

    def calc_points(
        self,
        farm_results,
        points,
        point_models=None,
        calc_parameters={},
        persist_mdata=True,
        persist_pdata=False,
        finalize=True,
        ambient=False,
        chunked_results=False,
        **kwargs,
    ):
        """
        Calculate data at a given set of points.

        Parameters
        ----------
        farm_results: xarray.Dataset
            The farm results. The calculated variables have
            dimensions (state, turbine)
        points: numpy.ndarray
            The points of interest, shape: (n_states, n_points, 3)
        point_models: str or foxes.core.PointDataModel
            Additional point models to be executed
        calc_parameters: dict
            Parameters for model calculation.
            Key: model name str, value: parameter dict
        persist_mdata: bool
            Switch for forcing dask to load all model data
            into memory
        persist_fdata: bool
            Switch for forcing dask to load all farm data
            into memory
        finalize: bool
            Flag for finalization after calculation
        ambient: bool
            Flag for ambient instead of waked calculation
        chunked_results: bool
            Flag for chunked results
        kwargs: dict, optional
            Additional parameters for run_calculation

        Returns
        -------
        point_results: xarray.Dataset
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
        self._print_deco("calc_points", n_points=points.shape[1])

        # collect models and initialize:
        mlist, calc_pars = self._collect_point_models(
            calc_parameters, point_models, ambient
        )

        # initialize models:
        if not mlist.initialized:
            mlist.initialize(self, self.verbosity)

        # get input model data:
        models_data = self.get_models_data()
        if persist_mdata:
            models_data = models_data.persist()
        self.print("\nInput data:\n\n", models_data, "\n")
        self.print(f"\nOutput farm variables:", ", ".join(self.farm_vars))
        self.print(f"\nChunks: {self.chunks}\n")

        # chunk farm results:
        if self.chunks is not None:
            farm_results = self.chunked(farm_results)
        self.print("\nInput farm data:\n\n", farm_results, "\n")

        # get point data:
        if FC.STATE in farm_results.coords:
            sinds = farm_results.coords[FC.STATE]
        elif models_data is not None and FC.STATE in models_data.coords:
            sinds = models_data.coords[FC.STATE]
        else:
            sinds = None
        point_data = self.new_point_data(points, sinds)
        if persist_pdata:
            point_data = point_data.persist()
        self.print("\nInput point data:\n\n", point_data, "\n")

        # check vars:
        ovars = mlist.output_point_vars(self)
        self.print(f"\nOutput point variables:", ", ".join(ovars))
        self.print(f"\nChunks: {self.chunks}\n")

        # calculate:
        self.print(
            f"Calculating {len(ovars)} variables at {points.shape[1]} points in {self.n_states} states"
        )

        point_results = mlist.run_calculation(
            self,
            models_data,
            farm_results,
            point_data,
            out_vars=ovars,
            parameters=calc_pars,
            **kwargs,
        )

        del models_data, farm_results, point_data

        # finalize models:
        if finalize:
            self.print("\n")
            mlist.finalize(self, self.verbosity)
            self.finalize()

        if ambient:
            dvars = [v for v in point_results.data_vars.keys() if v in FV.var2amb]
            point_results = point_results.drop_vars(dvars)

        if chunked_results:
            point_results = self.chunked(point_results)

        return point_results

    def finalize(self, clear_mem=False):
        """
        Finalizes the algorithm.

        Parameters
        ----------
        clear_mem: bool
            Clear idata memory

        """
        for m in self.all_models():
            m.finalize(self, self.verbosity)

        super().finalize(clear_mem)
