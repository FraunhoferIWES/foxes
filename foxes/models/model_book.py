import foxes.models as fm
import foxes.variables as FV

class ModelBook:

    def __init__(
            self,
            rotor_model_plugins=[]
        ):

        self.point_models = {}

        rvars = [FV.REWS, FV.REWS2, FV.REWS3, FV.TI, FV.RHO]
        self.rotor_models = {
            "centre": fm.rotor_models.CentreRotor(calc_vars=rvars)
        } 
        nlist = list(range(2, 11)) + [20]
        for n in nlist:
            self.rotor_models[f"grid{n**2}"] = fm.rotor_models.GridRotor(calc_vars=rvars, n=n, reduce=True)

        self.turbine_types = {}

        self.turbine_models = {
            "kTI"   : fm.turbine_models.kTI(),
            "kTI_02": fm.turbine_models.kTI(kTI=0.2),
            "kTI_04": fm.turbine_models.kTI(kTI=0.4),

            "kTI_amb"   : fm.turbine_models.kTI(ti_var=FV.AMB_TI),
            "kTI_amb_02": fm.turbine_models.kTI(ti_var=FV.AMB_TI, kTI=0.2),
            "kTI_amb_04": fm.turbine_models.kTI(ti_var=FV.AMB_TI, kTI=0.4)
        }

        self.turbine_orders = {
            "order_wd": fm.turbine_orders.OrderWD(var_wd=FV.WD)
        }

        self.farm_models = {
            f"farm_{mname}": fm.farm_models.Turbine2FarmModel(m) \
                                for mname, m in self.turbine_models.items()
        }

        self.farm_controllers = {
            "basic_ctrl": fm.farm_controllers.BasicFarmController()
        }

        self.partial_wakes = {
            "rotor_points": fm.partial_wakes.RotorPoints(),
            "top_hat"     : fm.partial_wakes.PartialTopHat(),
            "distsliced"  : fm.partial_wakes.PartialDistSlicedWake(),
            "auto"        : fm.partial_wakes.Mapped()
        }
        nlst = list(range(2, 11)) + [20]
        for n in nlst:
            self.partial_wakes[f"axiwake{n}"] = fm.partial_wakes.PartialAxiwake(n)
        for n in nlist:
            self.partial_wakes[f"distsliced{n**2}"] = fm.partial_wakes.PartialDistSlicedWake(n)
        for n in nlist:
            self.partial_wakes[f"grid{n**2}"] = fm.partial_wakes.PartialGrid(n)

        self.wake_frames = {
            "mean_wd": fm.wake_frames.MeanFarmWind(var_wd=FV.WD)
        }

        self.wake_superpositions = {
            "linear"    : fm.wake_superpositions.LinearWakeSuperposition(scalings=f'source_turbine_{FV.REWS}'),
            "linear_amb": fm.wake_superpositions.LinearWakeSuperposition(scalings=f'source_turbine_{FV.AMB_REWS}'),

            "ti_linear"   : fm.wake_superpositions.TISuperposition(ti_superp="linear", superp_to_amb="quadratic"),
            "ti_quadratic": fm.wake_superpositions.TISuperposition(ti_superp="quadratic", superp_to_amb="quadratic")
        }

        self.wake_models = {}
        slist = ["linear", "linear_amb"]
        for s in slist:

            self.wake_models[f"Jensen_{s}"] = fm.wake_models.top_hat.JensenWake(superposition=s)
            self.wake_models[f"Jensen_{s}_k007"] = fm.wake_models.top_hat.JensenWake(k=0.07, superposition=s)

            self.wake_models[f"Bastankhah_{s}"] = fm.wake_models.gaussian.BastankhahWake(superposition=s)
            self.wake_models[f"Bastankhah_{s}_k002"] = fm.wake_models.gaussian.BastankhahWake(k=0.02, superposition=s)

        slist = ["ti_linear", "ti_quadratic"]
        for s in slist:
            self.wake_models[f"CrespoHernandez_{s[3:]}"] = fm.wake_models.top_hat.CrespoHernandezTIWake(superposition=s)
            self.wake_models[f"CrespoHernandez_{s[3:]}_k002"] = fm.wake_models.top_hat.CrespoHernandezTIWake(k=0.02, superposition=s)

        self.sources = {
            "point_models"       : self.point_models, 
            "rotor_models"       : self.rotor_models,
            "turbine_types"      : self.turbine_types,
            "turbine_models"     : self.turbine_models,
            "turbine_orders"     : self.turbine_orders,
            "farm_models"        : self.farm_models,
            "farm_controllers"   : self.farm_controllers,
            "partial_wakes"      : self.partial_wakes,
            "wake_frames"        : self.wake_frames,
            "wake_superpositions": self.wake_superpositions,
            "wake_models"        : self.wake_models
        }

        for s in self.sources.values():
            for k, m in s.items():
                m.name = k
 
        for rp in rotor_model_plugins:
            for r in self.rotor_models.values():
                r.add_plugin(mbook=self, **rp)