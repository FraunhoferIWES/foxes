from foxes.core import Algorithm, FarmDataModelList
from foxes.core import PointDataModel, PointDataModelList
import foxes.algorithms.downwind.models as dm
import foxes.models as fm
import foxes.variables as FV


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
    verbosity : int
        The verbosity level, 0 means silent

    """

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
        verbosity=1,
    ):
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
        self.farm_controller.collect_models(self)
        for i, m in enumerate(self.farm_controller.pre_rotor_models.models):
            self.print(f"    {i}) {m} [pre-rotor]")
        for i, m in enumerate(self.farm_controller.post_rotor_models.models):
            self.print(
                f"    {i+len(self.farm_controller.pre_rotor_models.models)}) {m}"
            )
        self.print(deco)

    def initialize(self, **states_init_pars):
        """
        Initializes the algorithm.

        Parameters
        ----------
        states_init_pars : dict, optional
            Parameters for states initialization

        """
        if not self.states.initialized:
            self.print(f"\nInitializing states '{self.states.name}'")
            self.states.initialize(self, verbosity=self.verbosity, **states_init_pars)

        self.n_states = self.states.size()
        self.states_data = self.get_models_data(self.states)
        self.print("States data:\n")
        self.print(self.states_data)
        super().initialize()

    def reset_states(self, states, **states_init_pars):
        """
        Reset the underlying states

        Parameters
        ----------
        states : foxes.core.States
            The new states

        """
        if states is not self.states:
            if self.initialized:
                self.finalize(clear_mem=True)
            self.states = states
            self.initialize(**states_init_pars)

    def calc_farm(
        self,
        vars_to_amb=None,
        init_parameters={},
        calc_parameters={},
        final_parameters={},
        persist=True,
        clear_mem_models=True,
        ambient=False,
        **states_init_pars,
    ):
        """
        Calculate farm data.

        Parameters
        ----------
        vars_to_amb : list of str, optional
            Variables for which ambient variables should
            be stored. None means all.
        init_parameters : dict
            Parameters for model initialization.
            Key: model name str, value: parameter dict
        calc_parameters : dict
            Parameters for model calculation.
            Key: model name str, value: parameter dict
        final_parameters : dict
            Parameters for model finalization.
            Key: model name str, value: parameter dict
        persist : bool
            Switch for forcing dask to load all model data
            into memory
        clear_mem_models : bool
            Switch for clearing model memory during model
            finalization
        ambient : bool
            Flag for ambient instead of waked calculation
        states_init_pars : dict, optional
            Parameters for states initialization

        Returns
        -------
        farm_results : xarray.Dataset
            The farm results. The calculated variables have
            dimensions (state, turbine)

        """

        if not self.initialized:
            self.initialize(**states_init_pars)

        # welcome:
        self._print_deco("calc_farm")

        # prepare:
        init_pars = []
        calc_pars = []
        final_pars = []
        t2f = fm.farm_models.Turbine2FarmModel
        mlist = FarmDataModelList(models=[])
        fdict = {"clear_mem": clear_mem_models}

        # 0) set XHYD:
        mlist.models.append(t2f(fm.turbine_models.SetXYHD()))
        mlist.models[-1].name = "set_xyhd"
        init_pars.append(init_parameters.get(mlist.models[-1].name, {}))
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        final_pars.append(final_parameters.get(mlist.models[-1].name, fdict))

        # 1) run pre-rotor turbine models via farm controller:
        mlist.models.append(self.farm_controller)
        init_pars.append(init_parameters.get(mlist.models[-1].name, {}))
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        final_pars.append(final_parameters.get(mlist.models[-1].name, fdict))
        calc_pars[-1]["pre_rotor"] = True

        # 2) calculate yaw from wind direction at rotor centre:
        mlist.models.append(fm.rotor_models.CentreRotor(calc_vars=[FV.WD, FV.YAW]))
        mlist.models[-1].name = "calc_yaw"
        init_pars.append(init_parameters.get(mlist.models[-1].name, {}))
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        final_pars.append(final_parameters.get(mlist.models[-1].name, fdict))

        # 3) calculate ambient rotor results:
        mlist.models.append(self.rotor_model)
        init_pars.append(init_parameters.get(mlist.models[-1].name, {}))
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        final_pars.append(final_parameters.get(mlist.models[-1].name, fdict))
        calc_pars[-1].update(
            {"store_rpoints": True, "store_rweights": True, "store_amb_res": True}
        )

        # 4) calculate turbine order:
        mlist.models.append(dm.CalcOrder())
        init_pars.append(init_parameters.get(mlist.models[-1].name, {}))
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        final_pars.append(final_parameters.get(mlist.models[-1].name, fdict))

        # 5) run post-rotor turbine models via farm controller:
        mlist.models.append(self.farm_controller)
        init_pars.append(init_parameters.get(mlist.models[-1].name, {}))
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        final_pars.append(final_parameters.get(mlist.models[-1].name, fdict))
        calc_pars[-1]["pre_rotor"] = False

        # 6) copy results to ambient, requires self.farm_vars:
        self.farm_vars = mlist.output_farm_vars(self)
        mlist.models.append(dm.SetAmbFarmResults(vars_to_amb))
        mlist.models[-1].name = "set_amb_results"
        init_pars.append(init_parameters.get(mlist.models[-1].name, {}))
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        final_pars.append(final_parameters.get(mlist.models[-1].name, fdict))

        # 7) calculate wake effects:
        if not ambient:
            mlist.models.append(dm.FarmWakesCalculation())
            mlist.models[-1].name = "calc_wakes"
            init_pars.append(init_parameters.get(mlist.models[-1].name, {}))
            calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
            final_pars.append(final_parameters.get(mlist.models[-1].name, fdict))

        # update variables:
        self.farm_vars = [FV.WEIGHT] + mlist.output_farm_vars(self)

        # initialize models:
        mlist.initialize(self, parameters=init_pars, verbosity=self.verbosity)

        # get input model data:
        models_data = self.get_models_data(mlist).merge(
            self.states_data, compat="identical"
        )
        if persist:
            models_data = models_data.persist()
        self.print("\nInput model data:\n\n", models_data, "\n")
        self.print(f"\nOutput farm variables:", ", ".join(self.farm_vars))
        self.print(f"\nChunks: {self.chunks}\n")

        # run main calculation:
        self.print(
            f"\nCalculating {self.n_states} states for {self.n_turbines} turbines"
        )
        farm_results = mlist.run_calculation(
            self, models_data, out_vars=self.farm_vars, parameters=calc_pars
        )
        del models_data

        # finalize models:
        self.print("\n")
        mlist.finalize(
            self, results=farm_results, parameters=final_pars, verbosity=self.verbosity
        )

        if ambient:
            dvars = [v for v in farm_results.data_vars.keys() if v in FV.var2amb]
            farm_results = farm_results.drop_vars(dvars)

        return farm_results

    def calc_points(
        self,
        farm_results,
        points,
        vars=None,
        vars_to_amb=None,
        point_models=None,
        init_parameters={},
        calc_parameters={},
        final_parameters={},
        persist_mdata=True,
        persist_pdata=False,
        clear_mem_models=True,
        ambient=False,
        **states_init_pars,
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
        init_parameters : dict
            Parameters for model initialization.
            Key: model name str, value: parameter dict
        calc_parameters : dict
            Parameters for model calculation.
            Key: model name str, value: parameter dict
        final_parameters : dict
            Parameters for model finalization.
            Key: model name str, value: parameter dict
        persist_mdata : bool
            Switch for forcing dask to load all model data
            into memory
        persist_fdata : bool
            Switch for forcing dask to load all farm data
            into memory
        clear_mem_models : bool
            Switch for clearing model memory during model
            finalization
        ambient : bool
            Flag for ambient instead of waked calculation
        states_init_pars : dict, optional
            Parameters for states initialization

        Returns
        -------
        point_results : xarray.Dataset
            The point results. The calculated variables have
            dimensions (state, point)

        """

        if not self.initialized:
            self.initialize(**states_init_pars)
        if not ambient and farm_results is None:
            raise ValueError(
                f"Cannot calculate point results without farm results for ambient = {ambient}"
            )

        self._print_deco("calc_points", n_points=points.shape[1])

        # prepare:
        init_pars = []
        calc_pars = []
        final_pars = []
        mlist = PointDataModelList(models=[])
        fdict = {"clear_mem": clear_mem_models}

        # prepare extra eval models:
        emodels = []
        emodels_ipars = []
        emodels_cpars = []
        emodels_fpars = []
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
                emodels_ipars.append(init_parameters.get(emodels[-1].name, {}))
                emodels_cpars.append(calc_parameters.get(emodels[-1].name, {}))
                emodels_fpars.append(final_parameters.get(emodels[-1].name, fdict))
        emodels = PointDataModelList(models=emodels)

        # 0) calculate states results:
        mlist.models.append(self.states)
        init_pars.append(init_parameters.get(mlist.models[-1].name, {}))
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        final_pars.append(final_parameters.get(mlist.models[-1].name, {}))

        # 1) calculate ambient extra eval point results:
        mlist.models.append(emodels)
        init_pars.append({"parameters": emodels_ipars})
        calc_pars.append({"parameters": emodels_cpars})
        final_pars.append({"parameters": emodels_fpars})

        # 2) transfer ambient results:
        mlist.models.append(
            dm.SetAmbPointResults(point_vars=vars, vars_to_amb=vars_to_amb)
        )
        mlist.models[-1].name = "set_amb_results"
        init_pars.append(init_parameters.get(mlist.models[-1].name, {}))
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        final_pars.append(final_parameters.get(mlist.models[-1].name, fdict))

        # 3) calc wake effects:
        if not ambient:
            mlist.models.append(dm.PointWakesCalculation(vars, emodels, emodels_cpars))
            mlist.models[-1].name = "calc_wakes"
            init_pars.append(init_parameters.get(mlist.models[-1].name, {}))
            calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
            final_pars.append(final_parameters.get(mlist.models[-1].name, fdict))

        # initialize models:
        mlist.initialize(self, parameters=init_pars, verbosity=self.verbosity)

        # get input model data:
        models_data = self.get_models_data(mlist).merge(
            self.states_data, compat="identical"
        )
        if persist_mdata:
            models_data = models_data.persist()
        self.print("\nInput model data:\n\n", models_data, "\n")

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
        self.print("\n")
        mlist.finalize(
            self, point_results, parameters=final_pars, verbosity=self.verbosity
        )

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
            Flag for deleting algorithm data and
            resetting initialization flag

        """
        if clear_mem:
            self.states_data = None
        super().finalize(clear_mem=clear_mem)
