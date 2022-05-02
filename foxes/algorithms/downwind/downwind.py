
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
            vars_to_amb=None,
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
        for i, m in enumerate(self.farm_controller.turbine_models):
            self.print(f"    {i}) {m}")
        self.print(deco)

    def calc_farm(
            self, 
            vars_to_amb=None,
            init_parameters={},
            calc_parameters={},
            final_parameters={},
            persist=True
        ):

        self._print_deco("calc_farm")

        # prepare:
        init_pars  = []
        calc_pars  = []
        final_pars = []
        t2f        = fm.farm_models.Turbine2FarmModel
        mlist      = FarmDataModelList(models=[])

        # 0) set XHYD:
        mlist.models.append(t2f(fm.turbine_models.SetXYHD()))
        mlist.models[-1].name = "set_xyhd"
        init_pars.append(init_parameters.get(mlist.models[-1].name, {}))
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        final_pars.append(final_parameters.get(mlist.models[-1].name, {}))

        # 1) calculate yaw from wind direction at rotor centre:
        mlist.models.append(fm.rotor_models.CentreRotor(calc_vars=[FV.WD, FV.YAW]))
        mlist.models[-1].name = "calc_yaw"
        init_pars.append(init_parameters.get(mlist.models[-1].name, {}))
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        final_pars.append(final_parameters.get(mlist.models[-1].name, {}))
        
        # 2) calculate ambient rotor results:
        mlist.models.append(self.rotor_model)
        init_pars.append(init_parameters.get(mlist.models[-1].name, {}))
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        final_pars.append(final_parameters.get(mlist.models[-1].name, {}))
        calc_pars[-1].update({
            "store_rpoints": True, 
            "store_rweights": True, 
            "store_amb_res": True
        })

        # 3) calculate turbine order:
        mlist.models.append(self.turbine_order)
        init_pars.append(init_parameters.get(mlist.models[-1].name, {}))
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        final_pars.append(final_parameters.get(mlist.models[-1].name, {}))

        # 4) run turbine models via farm controller:
        mlist.models.append(self.farm_controller)
        init_pars.append(init_parameters.get(mlist.models[-1].name, {}))
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        final_pars.append(final_parameters.get(mlist.models[-1].name, {}))

        # 5) copy results to ambient, requires self.farm_vars:
        self.farm_vars = mlist.output_farm_vars(self)
        mlist.models.append(dm.SetAmbFarmResults(vars_to_amb))
        mlist.models[-1].name = "set_amb_results"
        init_pars.append(init_parameters.get(mlist.models[-1].name, {}))
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        final_pars.append(final_parameters.get(mlist.models[-1].name, {}))

        # 6) calculate wake effects:
        mlist.models.append(dm.FarmWakesCalculation())
        mlist.models[-1].name = "calc_wakes"
        init_pars.append(init_parameters.get(mlist.models[-1].name, {}))
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        final_pars.append(final_parameters.get(mlist.models[-1].name, {}))

        # update variables:
        self.farm_vars = [FV.WEIGHT] + mlist.output_farm_vars(self) 

        # initialize models:
        mlist.initialize(self, parameters=init_pars, verbosity=self.verbosity)

        # get input model data:
        models_data = self.get_models_data()
        if persist:
            models_data = models_data.persist()
        self.print("\nInput model data:\n\n", models_data, "\n")
        self.print(f"\nOutput farm variables:", ", ".join(self.farm_vars), "\n")  

        # run main calculation:
        self.print(f"\nCalculating {self.n_states} states for {self.n_turbines} turbines")
        farm_results = mlist.run_calculation(self, models_data, parameters=calc_pars)
        del models_data

        # finalize models:
        self.print("\n")
        mlist.finalize(self, parameters=final_pars, verbosity=self.verbosity)

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

        self._print_deco("calc_points", n_points=points.shape[1])

        # update eval models:
        emodels = []
        ipars   = []
        cpars   = []
        fpars   = []
        if point_models is not None:
            if not isinstance(point_models, list):
                point_models = [point_models]
            for m in point_models:
                if isinstance(m, str):
                    pname  = m
                    pmodel = self.mbook.point_models[pname]
                    pmodel.name = pname
                    emodels.append(pmodel)
                elif isinstance(m, PointDataModel):
                    emodels.append(m)
                else:
                    raise TypeError(f"Model '{m}' is neither str nor PointDataModel")
                ipars.append(init_parameters.get(emodels[-1].name, {}))
                cpars.append(calc_parameters.get(emodels[-1].name, {}))
                fpars.append(final_parameters.get(emodels[-1].name, {}))

        # prepare:
        init_pars  = []
        calc_pars  = []
        final_pars = []
        mlist      = PointDataModelList(models=[])

        # 0) calculate states results:
        mlist.models.append(self.states)
        init_pars.append(init_parameters.get(mlist.models[-1].name, {}))
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        final_pars.append(final_parameters.get(mlist.models[-1].name, {}))

        # 1) calculate ambient point results:
        mlist.models += emodels
        init_pars    += ipars
        calc_pars    += cpars
        final_pars   += fpars

        # 2) transfer ambient results:
        mlist.models.append(dm.SetAmbPointResults(point_vars=vars, vars_to_amb=vars_to_amb))
        mlist.models[-1].name = "set_amb_results"
        init_pars.append(init_parameters.get(mlist.models[-1].name, {}))
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        final_pars.append(final_parameters.get(mlist.models[-1].name, {}))

        # 3) calc wake effects:
        mlist.models.append(dm.PointWakesCalculation(point_vars=vars))
        mlist.models[-1].name = "calc_wakes"
        init_pars.append(init_parameters.get(mlist.models[-1].name, {}))
        calc_pars.append(calc_parameters.get(mlist.models[-1].name, {}))
        final_pars.append(final_parameters.get(mlist.models[-1].name, {}))

        # initialize models:
        mlist.initialize(self, parameters=init_pars, verbosity=self.verbosity)

        # get input model data:
        models_data = self.get_models_data()
        if persist_mdata:
            models_data = models_data.persist()
        self.print("\nInput model data:\n\n", models_data, "\n")

        self.print("\nInput farm data:\n\n", farm_results, "\n")

        # get point data:
        point_data = self.new_point_data(points)
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
        self.print(f"\nOutput farm variables:", ", ".join(vars), "\n")  
        self.print(f"Calculating {len(vars)} variables at {points.shape[1]} points in {self.n_states} states")

        # calculate:
        point_results = mlist.run_calculation(self, models_data, farm_results, point_data, 
                                            vars, parameters=calc_pars)
        del models_data, farm_results, point_data

        # finalize models:
        self.print("\n")
        mlist.finalize(self, parameters=final_pars, verbosity=self.verbosity)

        return point_results
        