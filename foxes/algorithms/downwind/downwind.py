
from foxes.core import Algorithm, FarmDataModelList
from foxes.core import PointDataModel, PointDataModelList
import foxes.algorithms.downwind.models as dm
import foxes.models as fm
import foxes.variables as FV

class Downwind(Algorithm):

    def __init__(
            self, 
            mbook, 
            farm,
            states,
            rotor_model,
            turbine_order,
            wake_models,
            wake_frame,
            partial_wakes_model,
            farm_controller="basic_ctrl",
            chunks={FV.STATE: 'auto', FV.TURBINE: -1},
            verbosity=1
        ):
        super().__init__(mbook, farm, chunks, verbosity)

        self.states   = states
        self.n_states = states.size()

        self.rotor_model = self.mbook.rotor_models[rotor_model]
        self.rotor_model.name = rotor_model

        self.turbine_order = self.mbook.turbine_orders[turbine_order]
        self.turbine_order.name = turbine_order

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

        self.models_data = None

    def _print_deco(self, func_name, n_points=None):

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
        self.print(f"  order     : {self.turbine_order}")
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
            self.print(f"    {i+len(self.farm_controller.pre_rotor_models.models)}) {m}")
        self.print(deco)

    def initialize(
            self,       
            vars_to_amb=None,  
            init_parameters={},
            calc_parameters={},
            final_parameters={},
            persist=True
        ):
        """
        Initializes the algorithm.
        """

        if not self.states.initialized:
            self.states.initialize(self)
            self.n_states = self.states.size()

        # prepare:
        self.init_pars_farm  = []
        self.calc_pars_farm  = []
        self.final_pars_farm = []
        t2f                  = fm.farm_models.Turbine2FarmModel
        self.mlist_farm      = FarmDataModelList(models=[])

        # 0) set XHYD:
        self.mlist_farm.models.append(t2f(fm.turbine_models.SetXYHD()))
        self.mlist_farm.models[-1].name = "set_xyhd"
        self.init_pars_farm.append(init_parameters.get(self.mlist_farm.models[-1].name, {}))
        self.calc_pars_farm.append(calc_parameters.get(self.mlist_farm.models[-1].name, {}))
        self.final_pars_farm.append(final_parameters.get(self.mlist_farm.models[-1].name, {}))

        # 1) run pre-rotor turbine models via farm controller:
        self.mlist_farm.models.append(self.farm_controller)
        self.init_pars_farm.append(init_parameters.get(self.mlist_farm.models[-1].name, {}))
        self.calc_pars_farm.append(calc_parameters.get(self.mlist_farm.models[-1].name, {}))
        self.final_pars_farm.append(final_parameters.get(self.mlist_farm.models[-1].name, {}))
        self.calc_pars_farm[-1]["pre_rotor"] = True

        # 2) calculate yaw from wind direction at rotor centre:
        self.mlist_farm.models.append(fm.rotor_models.CentreRotor(calc_vars=[FV.WD, FV.YAW]))
        self.mlist_farm.models[-1].name = "calc_yaw"
        self.init_pars_farm.append(init_parameters.get(self.mlist_farm.models[-1].name, {}))
        self.calc_pars_farm.append(calc_parameters.get(self.mlist_farm.models[-1].name, {}))
        self.final_pars_farm.append(final_parameters.get(self.mlist_farm.models[-1].name, {}))

        # 3) calculate ambient rotor results:
        self.mlist_farm.models.append(self.rotor_model)
        self.init_pars_farm.append(init_parameters.get(self.mlist_farm.models[-1].name, {}))
        self.calc_pars_farm.append(calc_parameters.get(self.mlist_farm.models[-1].name, {}))
        self.final_pars_farm.append(final_parameters.get(self.mlist_farm.models[-1].name, {}))
        self.calc_pars_farm[-1].update({
            "store_rpoints"  : True, 
            "store_rweights" : True, 
            "store_amb_res"  : True
        })
        
        # 4) calculate turbine order:
        self.mlist_farm.models.append(self.turbine_order)
        self.init_pars_farm.append(init_parameters.get(self.mlist_farm.models[-1].name, {}))
        self.calc_pars_farm.append(calc_parameters.get(self.mlist_farm.models[-1].name, {}))
        self.final_pars_farm.append(final_parameters.get(self.mlist_farm.models[-1].name, {}))
        
        # 5) run post-rotor turbine models via farm controller:
        self.mlist_farm.models.append(self.farm_controller)
        self.init_pars_farm.append(init_parameters.get(self.mlist_farm.models[-1].name, {}))
        self.calc_pars_farm.append(calc_parameters.get(self.mlist_farm.models[-1].name, {}))
        self.final_pars_farm.append(final_parameters.get(self.mlist_farm.models[-1].name, {}))
        self.calc_pars_farm[-1]["pre_rotor"] = False
        
        # 6) copy results to ambient, requires self.farm_vars:
        self.farm_vars = self.mlist_farm.output_farm_vars(self)
        self.mlist_farm.models.append(dm.SetAmbFarmResults(vars_to_amb))
        self.mlist_farm.models[-1].name = "set_amb_results"
        self.init_pars_farm.append(init_parameters.get(self.mlist_farm.models[-1].name, {}))
        self.calc_pars_farm.append(calc_parameters.get(self.mlist_farm.models[-1].name, {}))
        self.final_pars_farm.append(final_parameters.get(self.mlist_farm.models[-1].name, {}))

        # 7) calculate wake effects:
        self.mlist_farm.models.append(dm.FarmWakesCalculation())
        self.mlist_farm.models[-1].name = "calc_wakes"
        self.init_pars_farm.append(init_parameters.get(self.mlist_farm.models[-1].name, {}))
        self.calc_pars_farm.append(calc_parameters.get(self.mlist_farm.models[-1].name, {}))
        self.final_pars_farm.append(final_parameters.get(self.mlist_farm.models[-1].name, {}))

        # update variables:
        self.farm_vars = [FV.WEIGHT] + self.mlist_farm.output_farm_vars(self) 

        # initialize models:
        self.mlist_farm.initialize(self, parameters=self.init_pars_farm, verbosity=self.verbosity)

        # get input model data:
        self.models_data = self.get_models_data([self.states, self.mlist_farm])
        if persist:
            self.models_data = self.models_data.persist()

        super().initialize()

    def calc_farm(
            self, 
            vars_to_amb=None,
            init_parameters={},
            calc_parameters={},
            final_parameters={},
            persist=True
        ):

        if not self.initialized:
            self.initialize(vars_to_amb, init_parameters, calc_parameters, 
                                final_parameters, persist)

        # welcome:
        self._print_deco("calc_farm")
        self.print("\nInput model data:\n\n", self.models_data, "\n")
        self.print(f"\nOutput farm variables:", ", ".join(self.farm_vars), "\n")  

        # initialize models, if needed:
        if not self.mlist_farm.initialized:
            self.mlist_farm.initialize(self, parameters=self.init_pars_farm, 
                                            verbosity=self.verbosity)

        # run main calculation:
        self.print(f"\nCalculating {self.n_states} states for {self.n_turbines} turbines")
        farm_results = self.mlist_farm.run_calculation(self, self.models_data, out_vars=self.farm_vars, 
                                    loop_dims=[FV.STATE], out_core_vars=[FV.TURBINE, FV.VARS],
                                    parameters=self.calc_pars_farm)
        del self.models_data

        # finalize models:
        self.print("\n")
        self.mlist_farm.finalize(self, results=farm_results, parameters=self.final_pars_farm, 
                                    verbosity=self.verbosity)

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
            persist_pdata=False
        ):

        if not self.initialized:
            raise Exception(f"Algorithm '{self.name}': calc_points called before initialization")

        self._print_deco("calc_points", n_points=points.shape[1])

        # update eval models:
        self.emodels       = []
        self.emodels_ipars = []
        self.emodels_cpars = []
        self.emodels_fpars = []
        if point_models is not None:
            if not isinstance(point_models, list):
                point_models = [point_models]
            for m in point_models:
                if isinstance(m, str):
                    pname  = m
                    pmodel = self.mbook.point_models[pname]
                    pmodel.name = pname
                    self.emodels.append(pmodel)
                elif isinstance(m, PointDataModel):
                    self.emodels.append(m)
                else:
                    raise TypeError(f"Model '{m}' is neither str nor PointDataModel")
                self.emodels_ipars.append(init_parameters.get(self.emodels[-1].name, {}))
                self.emodels_cpars.append(calc_parameters.get(self.emodels[-1].name, {}))
                self.emodels_fpars.append(final_parameters.get(self.emodels[-1].name, {}))

        # prepare:
        init_pars_points  = []
        calc_pars_points  = []
        final_pars_points = []
        mlist_points      = PointDataModelList(models=[])

        # 0) calculate states results:
        mlist_points.models.append(self.states)
        init_pars_points.append(init_parameters.get(mlist_points.models[-1].name, {}))
        calc_pars_points.append(calc_parameters.get(mlist_points.models[-1].name, {}))
        final_pars_points.append(final_parameters.get(mlist_points.models[-1].name, {}))

        # 1) calculate ambient point results:
        mlist_points.models += self.emodels
        init_pars_points    += self.emodels_ipars
        calc_pars_points    += self.emodels_cpars
        final_pars_points   += self.emodels_fpars

        # 2) transfer ambient results:
        mlist_points.models.append(dm.SetAmbPointResults(point_vars=vars, vars_to_amb=vars_to_amb))
        mlist_points.models[-1].name = "set_amb_results"
        init_pars_points.append(init_parameters.get(mlist_points.models[-1].name, {}))
        calc_pars_points.append(calc_parameters.get(mlist_points.models[-1].name, {}))
        final_pars_points.append(final_parameters.get(mlist_points.models[-1].name, {}))

        # 3) calc wake effects:
        mlist_points.models.append(dm.PointWakesCalculation(point_vars=vars))
        mlist_points.models[-1].name = "calc_wakes"
        init_pars_points.append(init_parameters.get(mlist_points.models[-1].name, {}))
        calc_pars_points.append(calc_parameters.get(mlist_points.models[-1].name, {}))
        final_pars_points.append(final_parameters.get(mlist_points.models[-1].name, {}))

        # initialize models:
        mlist_points.initialize(self, parameters=init_pars_points, verbosity=self.verbosity)

        # get input model data:
        self.models_data = self.get_models_data([self.states, mlist_points])
        if persist_mdata:
            self.models_data = self.models_data.persist()
        self.print("\nInput model data:\n\n", self.models_data, "\n")

        # chunk farm results:
        if self.chunks is not None:
            farm_results = farm_results.chunk(chunks={FV.STATE: self.chunks[FV.STATE]})
        self.print("\nInput farm data:\n\n", farm_results, "\n")

        # get point data:
        sinds = farm_results.coords[FV.STATE] if len(farm_results.coords) else None
        point_data = self.new_point_data(points, sinds)
        if persist_pdata:
            point_data = point_data.persist()
        self.print("\nInput point data:\n\n", point_data, "\n")

        # check vars:
        ovars = mlist_points.output_point_vars(self)
        if vars is None:
            vars = ovars
        for v in vars:
            if v not in ovars:
                raise KeyError(f"Variable '{v}' not in output point vars: {ovars}")
        self.print(f"\nOutput point variables:", ", ".join(vars), "\n")  
        self.print(f"Calculating {len(vars)} variables at {points.shape[1]} points in {self.n_states} states")

        # calculate:
        point_results = mlist_points.run_calculation(self, self.models_data, farm_results, point_data, 
                                    out_vars=vars, loop_dims=[FV.STATE, FV.POINT], 
                                    parameters=calc_pars_points)
        del self.models_data, farm_results, point_data

        # finalize models:
        self.print("\n")
        mlist_points.finalize(self, point_results, parameters=final_pars_points, 
                                    verbosity=self.verbosity)

        return point_results
        