from foxes.core import Algorithm, FarmDataModelList
from foxes.core import PointDataModel, PointDataModelList
import foxes.algorithms.downwind.models as dm
import foxes.models as fm
import foxes.variables as FV
import foxes.constants as FC


class Downwind(Algorithm):
    """
    The downwind algorithm.

    The turbines are evaluated once, in the order
    that is calculated by the provided `TurbineOrder`
    object.

    Parameters
    ----------
    mbook : foxes.ModelBook
        The model book
    farm : foxes.WindFarm
        The wind farm
    states : foxes.core.States
        The ambient states
    wake_models : list of str
        The wake models, applied to all turbines.
        Will be looked up in the model book
    rotor_model : str
        The rotor model, for all turbines. Will be
        looked up in the model book
    wake_frame : str
        The wake frame. Will be looked up in the
        model book
    partial_wakes_model : str
        The partial wakes model. Will be
        looked up in the model book
    farm_controller : str
        The farm controller. Will be
        looked up in the model book
    chunks : dict
        The chunks choice for running in parallel with dask,
        e.g. `{"state": 1000}` for chunks of 1000 states
    dbook : foxes.DataBook, optional
        The data book, or None for default
    keep_models : list of str
        Keep these models data in memory and do not finalize them
    verbosity : int
        The verbosity level, 0 means silent

    Attributes
    ----------
    states : foxes.core.States
        The ambient states
    wake_models : list of foxes.core.WakeModel
        The wake models, applied to all turbines
    rotor_model : foxes.core.RotorModel
        The rotor model, for all turbines
    wake_frame : foxes.core.WakeFrame
        The wake frame
    partial_wakes_model : foxes.core.PartialWakesModel
        The partial wakes model
    farm_controller : foxes.core.FarmController
        The farm controller
    n_states : int
        The number of states

    """

    FarmWakesCalculation = dm.FarmWakesCalculation
    PointWakesCalculation = dm.point_wakes_calc.PointWakesCalculation
    SetAmbPointResults = dm.set_amb_point_results.SetAmbPointResults
    
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
        chunks={FV.STATE: 1000},
        dbook=None,
        keep_models=[],
        verbosity=1,
    ):
        super().__init__(mbook, farm, chunks, verbosity, dbook, keep_models)

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

        deco = "-" * 50
        self.print(f"\n{deco}")
        self.print(f"  Running {self.name}: {func_name}")
        self.print(deco)
        self.print(f"  n_states  : {self.n_states}")
        self.print(f"  n_turbines: {self.n_turbines}")
        if n_points is not None:
            self.print(f"  n_points  : {n_points}")
        self.print(deco)
        self.print(f"  states    : {self.states}")
        self.print(f"  rotor     : {self.rotor_model}")
        self.print(f"  controller: {self.farm_controller}")
        self.print(f"  partialwks: {self.partial_wakes_model}")
        self.print(f"  wake frame: {self.wake_frame}")
        self.print(deco)
        self.print(f"  wakes:")
        for i, w in enumerate(self.wake_models):
            self.print(f"    {i}) {w}")
        self.print(deco)
        self.print(f"  turbine models:")
        for i, m in enumerate(self.farm_controller.pre_rotor_models.models):
            self.print(f"    {i}) {m} [pre-rotor]")
        for i, m in enumerate(self.farm_controller.post_rotor_models.models):
            self.print(
                f"    {i+len(self.farm_controller.pre_rotor_models.models)}) {m}"
            )
        self.print(deco)
        self.print()

    def initialize(self):
        """
        Initializes the algorithm.          
        """
        self.print(f"\nInitializing algorithm '{self.name}'")
        super().initialize()           

        self.update_idata(self.states)
        self.n_states = self.states.size()

        mdls = [
            self.rotor_model,
            self.farm_controller,
            self.wake_frame,
            self.partial_wakes_model,
        ] + self.wake_models

        self.update_idata(mdls)

    def _collect_farm_models(
            self,
            vars_to_amb,
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

        # 0) set XHYD:
        mlist.models.append(t2f(fm.turbine_models.SetXYHD()))
        mlist.models[-1].name = "set_xyhd"
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        # 1) run pre-rotor turbine models via farm controller:
        mlist.models.append(self.farm_controller)
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        calc_pars[-1]["pre_rotor"] = True

        # 2) calculate yaw from wind direction at rotor centre:
        mlist.models.append(fm.rotor_models.CentreRotor(calc_vars=[FV.WD, FV.YAW]))
        mlist.models[-1].name = "calc_yaw"
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        # 3) calculate ambient rotor results:
        mlist.models.append(self.rotor_model)
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        calc_pars[-1].update(
            {"store_rpoints": True, "store_rweights": True, "store_amb_res": True}
        )

        # 4) calculate turbine order:
        mlist.models.append(dm.CalcOrder())
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        # 5) run post-rotor turbine models via farm controller:
        mlist.models.append(self.farm_controller)
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        calc_pars[-1]["pre_rotor"] = False

        # 6) copy results to ambient, requires self.farm_vars:
        self.farm_vars = mlist.output_farm_vars(self)
        mlist.models.append(dm.SetAmbFarmResults(vars_to_amb))
        mlist.models[-1].name = "set_amb_results"
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        # 7) calculate wake effects:
        if not ambient:
            mlist.models.append(self.FarmWakesCalculation())
            mlist.models[-1].name = "calc_wakes"
            calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        # update variables:
        self.farm_vars = [FV.WEIGHT] + mlist.output_farm_vars(self)

        return mlist, calc_pars

    def calc_farm(
        self,
        vars_to_amb=None,
        calc_parameters={},
        persist=True,
        finalize=True,
        ambient=False,
    ):
        """
        Calculate farm data.

        Parameters
        ----------
        vars_to_amb : list of str, optional
            Variables for which ambient variables should
            be stored. None means all.
        calc_parameters : dict
            Parameters for model calculation.
            Key: model name str, value: parameter dict
        persist : bool
            Switch for forcing dask to load all model data
            into memory
        finalize : bool
            Flag for finalization after calculation
        ambient : bool
            Flag for ambient instead of waked calculation

        Returns
        -------
        farm_results : xarray.Dataset
            The farm results. The calculated variables have
            dimensions (state, turbine)

        """
        # initialize algorithm:
        if not self.initialized:
            self.initialize()

        # welcome:
        self._print_deco("calc_farm")

        # collect models:
        mlist, calc_pars = self._collect_farm_models(
            vars_to_amb, calc_parameters, ambient)

        # initialize models and get input model data:
        self.update_idata(mlist)
        models_data = self.get_models_data()
        if persist:
            models_data = models_data.persist()
        self.print("\nInput data:\n\n", models_data, "\n")
        self.print(f"\nOutput farm variables:", ", ".join(self.farm_vars))
        self.print(f"\nChunks: {self.chunks}\n")

        # run main calculation:
        self.print(
            f"\nCalculating {self.n_states} states for {self.n_turbines} turbines"
        )
        farm_results = mlist.run_calculation(
            self, models_data, out_vars=self.farm_vars, parameters=calc_pars
        )
        farm_results[FV.TNAME] = ((FV.TURBINE,), self.farm.turbine_names)
        if FV.ORDER in farm_results:
            farm_results[FV.ORDER] = farm_results[FV.ORDER].astype(FC.ITYPE)
        del models_data

        # finalize models:
        if finalize:
            self.print("\n")
            mlist.finalize(self, self.verbosity)
            self.finalize()

        if ambient:
            dvars = [v for v in farm_results.data_vars.keys() if v in FV.var2amb]
            farm_results = farm_results.drop_vars(dvars)

        return farm_results

    def _collect_point_models(
            self,
            vars,
            vars_to_amb,
            calc_parameters,
            point_models,
            ambient,
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
            dm.SetAmbPointResults(point_vars=vars, vars_to_amb=vars_to_amb)
        )
        mlist.models[-1].name = "set_amb_results"
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))

        # 3) calc wake effects:
        if not ambient:
            mlist.models.append(dm.PointWakesCalculation(vars, emodels, emodels_cpars))
            mlist.models[-1].name = "calc_wakes"
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
    ):
        """
        Calculate data at a given set of points.

        Parameters
        ----------
        farm_results : xarray.Dataset
            The farm results. The calculated variables have
            dimensions (state, turbine)
        points : numpy.ndarray
            The points of interest, shape: (n_states, n_points, 3)
        vars : list of str, optional
            The variables that should be kept in the output,
            or `None` for all
        vars_to_amb : list of str, optional
            Variables for which ambient variables should
            be stored. None means all.
        point_models : str or foxes.core.PointDataModel
            Additional point models to be executed
        calc_parameters : dict
            Parameters for model calculation.
            Key: model name str, value: parameter dict
        persist_mdata : bool
            Switch for forcing dask to load all model data
            into memory
        persist_fdata : bool
            Switch for forcing dask to load all farm data
            into memory
        finalize : bool
            Flag for finalization after calculation
        ambient : bool
            Flag for ambient instead of waked calculation

        Returns
        -------
        point_results : xarray.Dataset
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

        # collect models:
        mlist, calc_pars = self._collect_point_models(
            vars, vars_to_amb, calc_parameters, point_models, ambient)

        # initialize models and get input model data:
        self.update_idata(mlist)
        models_data = self.get_models_data()
        if persist_mdata:
            models_data = models_data.persist()
        self.print("\nInput data:\n\n", models_data, "\n")
        self.print(f"\nOutput farm variables:", ", ".join(self.farm_vars))
        self.print(f"\nChunks: {self.chunks}\n")

        # chunk farm results:
        if self.chunks is not None:
            farm_results = farm_results.chunk(chunks={FV.STATE: self.chunks[FV.STATE]})
        self.print("\nInput farm data:\n\n", farm_results, "\n")

        # get point data:
        if FV.STATE in farm_results.coords:
            sinds = farm_results.coords[FV.STATE]
        elif models_data is not None and FV.STATE in models_data.coords:
            sinds = models_data.coords[FV.STATE]
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

        return point_results

    def finalize(self, clear_mem=False):
        """
        Finalizes the algorithm.

        Parameters
        ----------
        clear_mem : bool
            Clear idata memory, including keep_models entries

        """
        mdls = [
            self.states,
            self.rotor_model,
            self.farm_controller,
            self.wake_frame,
            self.partial_wakes_model,
        ] + self.wake_models
        
        for m in mdls:
            self.finalize_model(m)

        super().finalize(clear_mem)
