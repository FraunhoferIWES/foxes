
from foxes.core import Algorithm, FarmDataModelList
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

        self.states      = states
        self.n_states    = states.size()
        self.vars_to_amb = vars_to_amb

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

    def calc_farm(self):

        deco = "-" * 50
        self.print(f"\n{deco}")
        self.print(f"  Running {self.name}: calc_farm")
        self.print(deco)
        self.print(f"  n_states  : {self.n_states}")
        self.print(f"  n_turbines: {self.n_turbines}")
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

        # prepare:
        init_pars  = []
        calc_pars  = []
        final_pars = []
        t2f        = fm.farm_models.Turbine2FarmModel
        mlist      = FarmDataModelList(models=[])

        # 0) set XHYD:
        mlist.models.append(t2f(fm.turbine_models.SetXYHD()))
        mlist.models[-1].name = "set_xyhd"
        init_pars.append({})
        calc_pars.append({})
        final_pars.append({})

        # 1) calculate yaw from wind direction at rotor centre:
        mlist.models.append(fm.rotor_models.CentreRotor(calc_vars=[FV.WD, FV.YAW]))
        mlist.models[-1].name = "calc_yaw"
        init_pars.append({})
        calc_pars.append({})
        final_pars.append({})
        
        # 2) calculate ambient rotor results:
        mlist.models.append(self.rotor_model)
        init_pars.append({})
        calc_pars.append({
            "store_rpoints": True, 
            "store_rweights": True, 
            "store_amb_res": True
        })
        final_pars.append({})

        # 3) calculate turbine order:
        mlist.models.append(self.turbine_order)
        init_pars.append({})
        calc_pars.append({})
        final_pars.append({})

        # 4) run turbine models via farm controller:
        mlist.models.append(self.farm_controller)
        init_pars.append({})
        calc_pars.append({})
        final_pars.append({})

        # 5) copy results to ambient, requires self.farm_vars:
        self.farm_vars = mlist.output_farm_vars(self)
        mlist.models.append(dm.SetAmbResults(self.vars_to_amb))
        mlist.models[-1].name = "set_amb_results"
        init_pars.append({})
        calc_pars.append({})
        final_pars.append({})

        # 6) calculate wake effects:
        mlist.models.append(dm.FarmWakesCalculation())
        mlist.models[-1].name = "calc_wakes"
        init_pars.append({})
        calc_pars.append({})
        final_pars.append({})

        # update variables:
        self.farm_vars = mlist.output_farm_vars(self)   

        # create data, filled with zeros:
        farm_data  = self.new_farm_data(mlist.input_farm_data(self), self.chunks).persist()
        self.print("\nInput data:\n\n", farm_data, "\n")

        # initialize models:
        mlist.initialize(self, farm_data, parameters=init_pars, verbosity=self.verbosity)

        # run main calculation:
        self.print(f"\nCalculating {self.n_states} states for {self.n_turbines} turbines")
        farm_data = mlist.run_calculation(self, farm_data, parameters=calc_pars)

        # finalize models:
        self.print("\n")
        mlist.finalize(self, farm_data, parameters=final_pars, verbosity=self.verbosity)

        return farm_data
        