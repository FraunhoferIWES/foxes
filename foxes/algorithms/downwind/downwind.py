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
     wake_models: list of foxes.core.WakeModel
         The wake models, applied to all turbines
     rotor_model: foxes.core.RotorModel
         The rotor model, for all turbines
     wake_frame: foxes.core.WakeFrame
         The wake frame
     partial_wakes_model: foxes.core.PartialWakesModel
         The partial wakes model
     farm_controller: foxes.core.FarmController
         The farm controller
     n_states: int
         The number of states

    :group: algorithms.downwind

    """

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

    def __init__(
        self,
        mbook,
        farm,
        states,
        wake_models,
        rotor_model="centre",
        wake_frame="rotor_wd",
        partial_wakes_model="auto",
        farm_controller="basic_ctrl",
        chunks={FC.STATE: 1000, FC.POINT: 10000},
        dbook=None,
        verbosity=1,
    ):
        """
        Constructor.

        Parameters
        ----------
        mbook: foxes.ModelBook
            The model book
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
        partial_wakes_model: str
            The partial wakes model. Will be
            looked up in the model book
        farm_controller: str
            The farm controller. Will be
            looked up in the model book
        chunks: dict
            The chunks choice for running in parallel with dask,
            e.g. `{"state": 1000}` for chunks of 1000 states
        dbook: foxes.DataBook, optional
            The data book, or None for default
        verbosity: int
            The verbosity level, 0 means silent

        """
        super().__init__(mbook, farm, chunks, verbosity, dbook)

        self.states = states
        self.n_states = None
        self.states_data = None

        self.rotor_model = self.mbook.rotor_models[rotor_model]
        self.rotor_model.name = rotor_model

        self.partial_wakes_model = self.mbook.partial_wakes[partial_wakes_model]
        self.partial_wakes_model.name = partial_wakes_model

        self.wake_frame = self.mbook.wake_frames[wake_frame]
        self.wake_frame.name = wake_frame

        self.wake_models = []
        for w in wake_models:
            self.wake_models.append(self.mbook.wake_models[w])
            self.wake_models[-1].name = w

        self.farm_controller = self.mbook.farm_controllers[farm_controller]
        self.farm_controller.name = farm_controller

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
            print(f"  partialwks: {self.partial_wakes_model}")
            print(f"  wake frame: {self.wake_frame}")
            print(deco)
            print(f"  wakes:")
            for i, w in enumerate(self.wake_models):
                print(f"    {i}) {w}")
            print(deco)
            print(f"  turbine models:")
            for i, m in enumerate(self.farm_controller.pre_rotor_models.models):
                print(f"    {i}) {m} [pre-rotor]")
            for i, m in enumerate(self.farm_controller.post_rotor_models.models):
                print(f"    {i+len(self.farm_controller.pre_rotor_models.models)}) {m}")
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
            self.partial_wakes_model,
        ] + self.wake_models

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
        calc_parameters,
        ambient,
    ):
        """
        Helper function that creates model list
        """
        # prepare:
        calc_pars = []
        t2f = fm.farm_models.Turbine2FarmModel
        mlist = FarmDataModelList(models=[])
        mlist.name = f"{self.name}_calc"

        # 0) set XHYD:
        m = fm.turbine_models.SetXYHD()
        mlist.models.append(t2f(m))
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        # 1) run pre-rotor turbine models via farm controller:
        mlist.models.append(self.farm_controller)
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        calc_pars[-1]["pre_rotor"] = True

        # 2) calculate yaw from wind direction at rotor centre:
        mlist.models.append(fm.rotor_models.CentreRotor(calc_vars=[FV.WD, FV.YAW]))
        mlist.models[-1].name = "calc_yaw_" + mlist.models[-1].name
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        # 3) calculate ambient rotor results:
        mlist.models.append(self.rotor_model)
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        calc_pars[-1].update(
            {"store_rpoints": True, "store_rweights": True, "store_amb_res": True}
        )

        # 4) calculate turbine order:
        mlist.models.append(self.get_model("CalcOrder")())
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        # 5) run post-rotor turbine models via farm controller:
        mlist.models.append(self.farm_controller)
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        calc_pars[-1]["pre_rotor"] = False

        # 6) copy results to ambient, requires self.farm_vars:
        self.farm_vars = mlist.output_farm_vars(self)
        mlist.models.append(self.get_model("SetAmbFarmResults")())
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        # 7) calculate wake effects:
        if not ambient:
            mlist.models.append(self.get_model("FarmWakesCalculation")())
            calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        return mlist, calc_pars

    def _calc_farm_vars(self, mlist):
        """Helper function that gathers the farm variables"""
        self.farm_vars = sorted(list(set([FV.WEIGHT] + mlist.output_farm_vars(self))))

    def _run_farm_calc(self, mlist, *data, **kwargs):
        """Helper function for running the main farm calculation"""
        self.print(
            f"\nCalculating {self.n_states} states for {self.n_turbines} turbines"
        )
        farm_results = mlist.run_calculation(
            self, *data, out_vars=self.farm_vars, **kwargs
        )
        farm_results[FC.TNAME] = ((FC.TURBINE,), self.farm.turbine_names)
        if FV.ORDER in farm_results:
            farm_results[FV.ORDER] = farm_results[FV.ORDER].astype(FC.ITYPE)

        return farm_results

    def calc_farm(
        self,
        calc_parameters={},
        persist=True,
        finalize=True,
        ambient=False,
        chunked_results=False,
    ):
        """
        Calculate farm data.

        Parameters
        ----------
        calc_parameters: dict
            Parameters for model calculation.
            Key: model name str, value: parameter dict
        persist: bool
            Switch for forcing dask to load all model data
            into memory
        finalize: bool
            Flag for finalization after calculation
        ambient: bool
            Flag for ambient instead of waked calculation
        chunked_results: bool
            Flag for chunked results

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
        mlist, calc_pars = self._collect_farm_models(calc_parameters, ambient)

        # initialize models:
        if not mlist.initialized:
            mlist.initialize(self, self.verbosity)
            self._calc_farm_vars(mlist)
        self._print_model_oder(mlist, calc_pars)

        # get input model data:
        models_data = self.get_models_data()
        if persist:
            models_data = models_data.persist()
        self.print("\nInput data:\n\n", models_data, "\n")
        self.print(f"\nOutput farm variables:", ", ".join(self.farm_vars))
        self.print(f"\nChunks: {self.chunks}\n")

        # run main calculation:
        farm_results = self._run_farm_calc(mlist, models_data, parameters=calc_pars)
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
        vars=None,
        vars_to_amb=None,
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
        mlist.models.append(
            self.get_model("SetAmbPointResults")(
                point_vars=vars, vars_to_amb=vars_to_amb
            )
        )
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        # 3) calc wake effects:
        if not ambient:
            mlist.models.append(
                self.get_model("PointWakesCalculation")(vars, emodels, emodels_cpars)
            )
            calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        return mlist, calc_pars

    def calc_points(
        self,
        farm_results,
        points,
        vars=None,
        vars_to_amb=None,
        point_models=None,
        calc_parameters={},
        persist_mdata=True,
        persist_pdata=False,
        finalize=True,
        ambient=False,
        chunked_results=False,
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
        vars: list of str, optional
            The variables that should be kept in the output,
            or `None` for all
        vars_to_amb: list of str, optional
            Variables for which ambient variables should
            be stored. None means all.
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
            vars, vars_to_amb, calc_parameters, point_models, ambient
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
        if vars is None:
            vars = ovars
        for v in vars:
            if v not in ovars:
                raise KeyError(f"Variable '{v}' not in output point vars: {ovars}")
        self.print(f"\nOutput point variables:", ", ".join(vars))
        self.print(f"\nChunks: {self.chunks}\n")

        # calculate:
        self.print(
            f"Calculating {len(vars)} variables at {points.shape[1]} points in {self.n_states} states"
        )
        point_results = mlist.run_calculation(
            self,
            models_data,
            farm_results,
            point_data,
            out_vars=vars,
            parameters=calc_pars,
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
