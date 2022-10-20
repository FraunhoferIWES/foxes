import foxes.models as fm
import foxes.variables as FV
from foxes.utils import Dict


class ModelBook:
    """
    Container for all kinds of models.

    Parameters
    ----------
    Pct_file : str, optional
        Path to power/ct curve file, for creation
        of default turbine type model

    Attributes
    ----------
    point_models : foxes.utils.Dict
        The point models. Keys: model name str,
        values: foxes.core.PointDataModel
    rotor_models : foxes.utils.Dict
        The rotor models. Keys: model name str,
        values: foxes.core.RotorModel
    turbine_types : foxes.utils.Dict
        The turbine type models. Keys: model name str,
        values: foxes.core.TurbineType
    turbine_models : foxes.utils.Dict
        The turbine models. Keys: model name str,
        values: foxes.core.TurbineModel
    turbine_orders : foxes.utils.Dict
        The turbine orders. Keys: model name str,
        values: foxes.core.TurbineOrder
    farm_models : foxes.utils.Dict
        The farm models. Keys: model name str,
        values: foxes.core.FarmModel
    farm_controllers : foxes.utils.Dict
        The farm controllers. Keys: model name str,
        values: foxes.core.FarmController
    partial_wakes : foxes.utils.Dict
        The partial wakes. Keys: model name str,
        values: foxes.core.PartialWakeModel
    wake_frames : foxes.utils.Dict
        The wake frames. Keys: model name str,
        values: foxes.core.WakeFrame
    wake_superpositions : foxes.utils.Dict
        The wake superposition models. Keys: model name str,
        values: foxes.core.WakeSuperposition
    wake_models : foxes.utils.Dict
        The wake models. Keys: model name str,
        values: foxes.core.WakeModel
    sources : foxes.utils.Dict
        All sources dict

    """

    def __init__(self, Pct_file=None):

        self.point_models = Dict(name="point_models")

        self.rotor_models = Dict(name="rotor_models")
        rvars = [FV.REWS, FV.REWS2, FV.REWS3, FV.TI, FV.RHO]
        self.rotor_models["centre"] = fm.rotor_models.CentreRotor(calc_vars=rvars)
        nlist = list(range(2, 11)) + [20]
        for n in nlist:
            self.rotor_models[f"grid{n**2}"] = fm.rotor_models.GridRotor(
                calc_vars=rvars, n=n, reduce=True
            )

        self.turbine_types = Dict(name="turbine_types")
        self.turbine_types["null_type"] = fm.turbine_types.NullType()
        if Pct_file is not None:
            self.turbine_types["Pct"] = fm.turbine_types.PCtFile(Pct_file)

        self.turbine_models = Dict(
            name="turbine_models",
            kTI=fm.turbine_models.kTI(),
            kTI_02=fm.turbine_models.kTI(kTI=0.2),
            kTI_04=fm.turbine_models.kTI(kTI=0.4),
            kTI_amb=fm.turbine_models.kTI(ti_var=FV.AMB_TI),
            kTI_amb_02=fm.turbine_models.kTI(ti_var=FV.AMB_TI, kTI=0.2),
            kTI_amb_04=fm.turbine_models.kTI(ti_var=FV.AMB_TI, kTI=0.4),
            thrust2ct=fm.turbine_models.Thrust2Ct(),
        )

        self.farm_models = Dict(
            name="farm_models",
            **{
                f"farm_{mname}": fm.farm_models.Turbine2FarmModel(m)
                for mname, m in self.turbine_models.items()
            },
        )

        self.farm_controllers = Dict(
            name="farm_controllers",
            basic_ctrl=fm.farm_controllers.BasicFarmController(),
        )

        self.partial_wakes = Dict(
            name="partial_wakes",
            rotor_points=fm.partial_wakes.RotorPoints(),
            top_hat=fm.partial_wakes.PartialTopHat(),
            distsliced=fm.partial_wakes.PartialDistSlicedWake(),
            auto=fm.partial_wakes.Mapped(),
        )
        nlst = list(range(2, 11)) + [20]
        for n in nlst:
            self.partial_wakes[f"axiwake{n}"] = fm.partial_wakes.PartialAxiwake(n)
        for n in nlist:
            self.partial_wakes[
                f"distsliced{n**2}"
            ] = fm.partial_wakes.PartialDistSlicedWake(n)
        for n in nlist:
            self.partial_wakes[f"grid{n**2}"] = fm.partial_wakes.PartialGrid(n)

        self.wake_frames = Dict(
            name="wake_frames",
            rotor_wd=fm.wake_frames.RotorWD(var_wd=FV.WD),
            streamlines_100=fm.wake_frames.Streamlines(step=100),
        )

        self.wake_superpositions = Dict(
            name="wake_superpositions",
            linear=fm.wake_superpositions.LinearSuperposition(
                scalings=f"source_turbine_{FV.REWS}"
            ),
            linear_amb=fm.wake_superpositions.LinearSuperposition(
                scalings=f"source_turbine_{FV.AMB_REWS}"
            ),
            quadratic=fm.wake_superpositions.QuadraticSuperposition(
                scalings=f"source_turbine_{FV.REWS}"
            ),
            quadratic_amb=fm.wake_superpositions.QuadraticSuperposition(
                scalings=f"source_turbine_{FV.AMB_REWS}"
            ),
            max=fm.wake_superpositions.MaxSuperposition(
                scalings=f"source_turbine_{FV.REWS}"
            ),
            max_amb=fm.wake_superpositions.MaxSuperposition(
                scalings=f"source_turbine_{FV.AMB_REWS}"
            ),
            ti_linear=fm.wake_superpositions.TISuperposition(
                ti_superp="linear", superp_to_amb="quadratic"
            ),
            ti_quadratic=fm.wake_superpositions.TISuperposition(
                ti_superp="quadratic", superp_to_amb="quadratic"
            ),
            ti_max=fm.wake_superpositions.TISuperposition(
                ti_superp="max", superp_to_amb="quadratic"
            ),
        )

        self.wake_models = Dict(name="wake_models")
        slist = ["linear", "linear_amb", "quadratic", "quadratic_amb", "max", "max_amb"]
        for s in slist:

            self.wake_models[f"Jensen_{s}"] = fm.wake_models.wind.JensenWake(
                superposition=s
            )
            self.wake_models[f"Jensen_{s}_k007"] = fm.wake_models.wind.JensenWake(
                k=0.07, superposition=s
            )

            self.wake_models[f"Bastankhah_{s}"] = fm.wake_models.wind.BastankhahWake(
                superposition=s
            )
            self.wake_models[
                f"Bastankhah_{s}_k002"
            ] = fm.wake_models.wind.BastankhahWake(k=0.02, superposition=s)

            self.wake_models[f"TurbOPark_{s}"] = fm.wake_models.wind.TurbOParkWake(
                superposition=s
            )
            self.wake_models[f"TurbOPark_{s}_k004"] = fm.wake_models.wind.TurbOParkWake(
                k=0.04, superposition=s
            )

        slist = ["ti_linear", "ti_quadratic", "ti_max"]
        for s in slist:
            self.wake_models[
                f"CrespoHernandez_{s[3:]}"
            ] = fm.wake_models.ti.CrespoHernandezTIWake(superposition=s)
            self.wake_models[
                f"CrespoHernandez_{s[3:]}_k002"
            ] = fm.wake_models.ti.CrespoHernandezTIWake(k=0.02, superposition=s)

            self.wake_models[f"IECTI2005_{s[3:]}"] = fm.wake_models.ti.IECTIWake(
                superposition=s, iec_type="2005"
            )

            self.wake_models[f"IECTI2019_{s[3:]}"] = fm.wake_models.ti.IECTIWake(
                superposition=s, iec_type="2019"
            )

        self.sources = Dict(
            name="sources",
            point_models=self.point_models,
            rotor_models=self.rotor_models,
            turbine_types=self.turbine_types,
            turbine_models=self.turbine_models,
            farm_models=self.farm_models,
            farm_controllers=self.farm_controllers,
            partial_wakes=self.partial_wakes,
            wake_frames=self.wake_frames,
            wake_superpositions=self.wake_superpositions,
            wake_models=self.wake_models,
        )

        for s in self.sources.values():
            for k, m in s.items():
                m.name = k

    def __getitem__(self, key):
        return self.sources.__getitem__(key)

    def print_toc(self, subset=None, search=None):
        """
        Print the contents.

        Parameters
        ----------
        subset : list of str, optional
            Selection of model types
        search :  str, optional
            String that has to be part of the model name

        """
        for k in sorted(list(self.sources.keys())):
            ms = self.sources[k]
            if subset is None or k in subset:
                print(k)
                print("-" * len(k))
                if len(ms):
                    for mname in sorted(list(ms.keys())):
                        if search is None or search in mname:
                            print(f"{mname}: {ms[mname]}")
                else:
                    print("(none)")
                print()
