import foxes.models as fm
import foxes.variables as FV

class ModelBook:

    def __init__(self):

        self.states_models = {}

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
            "kTI": fm.turbine_models.kTI(),
            "kTI_02": fm.turbine_models.kTI(kTI=0.2),
            "kTI_04": fm.turbine_models.kTI(kTI=0.4)
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
            "rotor_points"   : fm.partial_wakes.RotorPoints(),
            "partial_top_hat": fm.partial_wakes.PartialTopHat(),
            "distsliced"     : fm.partial_wakes.PartialDistSlicedWake()
        }
        nlst = list(range(2, 11)) + [20, 50, 100]
        for n in nlst:
            self.partial_wakes[f"axiwake_{n}"] = fm.partial_wakes.PartialAxiwake(n)
        nlist = list(range(2, 11)) + [20]
        for n in nlist:
            self.partial_wakes[f"distsliced_{n**2}"] = fm.partial_wakes.PartialDistSlicedWake(n)

        self.wake_frames = {
            "mean_wd": fm.wake_frames.MeanFarmWind(var_wd=FV.WD)
        }

        self.wake_superpositions = {
            "linear"    : fm.wake_superpositions.LinearWakeSuperposition(scalings=f'source_turbine_{FV.REWS}'),
            "linear_amb": fm.wake_superpositions.LinearWakeSuperposition(scalings=f'source_turbine_{FV.AMB_REWS}')
        }

        self.wake_models = {}
        for s in self.wake_superpositions.keys():

            self.wake_models[f"Jensen_{s}"] = fm.wake_models.top_hat.JensenWake(superposition=s)
            self.wake_models[f"Jensen_{s}_k007"] = fm.wake_models.top_hat.JensenWake(k=0.07, superposition=s)

            self.wake_models[f"Bastankhah_{s}"] = fm.wake_models.gaussian.BastankhahWake(superposition=s)
            self.wake_models[f"Bastankhah_{s}_k002"] = fm.wake_models.gaussian.BastankhahWake(k=0.02, superposition=s)


 