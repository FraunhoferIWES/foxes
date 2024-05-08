import foxes.models as fm
import foxes.variables as FV
from foxes.utils import Dict

from foxes.core import (
    PointDataModel,
    FarmDataModel,
    FarmController,
    RotorModel,
    TurbineType,
    TurbineModel,
    PartialWakesModel,
    WakeFrame,
    WakeSuperposition,
    WakeModel,
    AxialInductionModel,
    TurbineInductionModel,
)


class ModelBook:
    """
    Container for all kinds of models.

    Attributes
    ----------
    point_models: foxes.utils.Dict
        The point models. Keys: model name str,
        values: foxes.core.PointDataModel
    rotor_models: foxes.utils.Dict
        The rotor models. Keys: model name str,
        values: foxes.core.RotorModel
    turbine_types: foxes.utils.Dict
        The turbine type models. Keys: model name str,
        values: foxes.core.TurbineType
    turbine_models: foxes.utils.Dict
        The turbine models. Keys: model name str,
        values: foxes.core.TurbineModel
    farm_models: foxes.utils.Dict
        The farm models. Keys: model name str,
        values: foxes.core.FarmModel
    farm_controllers: foxes.utils.Dict
        The farm controllers. Keys: model name str,
        values: foxes.core.FarmController
    partial_wakes: foxes.utils.Dict
        The partial wakes. Keys: model name str,
        values: foxes.core.PartialWakeModel
    wake_frames: foxes.utils.Dict
        The wake frames. Keys: model name str,
        values: foxes.core.WakeFrame
    wake_superpositions: foxes.utils.Dict
        The wake superposition models. Keys: model name str,
        values: foxes.core.WakeSuperposition
    wake_models: foxes.utils.Dict
        The wake models. Keys: model name str,
        values: foxes.core.WakeModel
    induction_models: foxes.utils.Dict
        The induction models. Keys: model name str,
        values: foxes.core.AxialInductionModel
    sources: foxes.utils.Dict
        All sources dict
    base_classes: foxes.utils.Dict
        The base classes for all model types

    :group: models

    """

    def __init__(self, Pct_file=None):
        """
        Constructor.

        Parameters
        ----------
        Pct_file: str, optional
            Path to power/ct curve file, for creation
            of default turbine type model
        """
        self.point_models = Dict(name="point_models")
        self.point_models["tke2ti"] = fm.point_models.TKE2TI()

        self.rotor_models = Dict(name="rotor_models")
        rvars = [FV.REWS, FV.REWS2, FV.REWS3, FV.TI, FV.RHO]
        self.rotor_models["centre"] = fm.rotor_models.CentreRotor(calc_vars=rvars)
        nlist = list(range(2, 11)) + [20]
        for n in nlist:
            self.rotor_models[f"grid{n**2}"] = fm.rotor_models.GridRotor(
                calc_vars=rvars, n=n, reduce=True
            )
            self.rotor_models[f"level{n}"] = fm.rotor_models.LevelRotor(
                calc_vars=rvars, n=n, reduce=True
            )

        self.turbine_types = Dict(name="turbine_types")
        self.turbine_types["null_type"] = fm.turbine_types.NullType()
        self.turbine_types["NREL5MW"] = fm.turbine_types.PCtFile(
            "NREL-5MW-D126-H90.csv", rho=1.225
        )
        self.turbine_types["DTU10MW"] = fm.turbine_types.PCtFile(
            "DTU-10MW-D178d3-H119.csv", rho=1.225
        )
        self.turbine_types["IEA15MW"] = fm.turbine_types.PCtFile(
            "IEA-15MW-D240-H150.csv", rho=1.225
        )
        self.turbine_types["IWT7.5MW"] = fm.turbine_types.PCtFile(
            "IWT-7d5MW-D164-H100.csv", rho=1.225
        )
        if Pct_file is not None:
            self.turbine_types["Pct"] = fm.turbine_types.PCtFile(Pct_file)

        self.turbine_models = Dict(
            name="turbine_models",
            kTI=fm.turbine_models.kTI(),
            kTI_02=fm.turbine_models.kTI(kTI=0.2),
            kTI_04=fm.turbine_models.kTI(kTI=0.4),
            kTI_05=fm.turbine_models.kTI(kTI=0.5),
            kTI_amb=fm.turbine_models.kTI(ti_var=FV.AMB_TI),
            kTI_amb_02=fm.turbine_models.kTI(ti_var=FV.AMB_TI, kTI=0.2),
            kTI_amb_04=fm.turbine_models.kTI(ti_var=FV.AMB_TI, kTI=0.4),
            kTI_amb_05=fm.turbine_models.kTI(ti_var=FV.AMB_TI, kTI=0.5),
            thrust2ct=fm.turbine_models.Thrust2Ct(),
            PMask=fm.turbine_models.PowerMask(),
            yaw2yawm=fm.turbine_models.YAW2YAWM(),
            yawm2yaw=fm.turbine_models.YAWM2YAW(),
        )
        self.turbine_models["hubh_data"] = fm.turbine_models.RotorCentreCalc(
            {
                f"{FV.WD}_HH": FV.WD,
                f"{FV.WS}_HH": FV.WS,
                f"{FV.TI}_HH": FV.TI,
                f"{FV.RHO}_HH": FV.RHO,
            }
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
            centre=fm.partial_wakes.PartialCentre(),
        )
        nlst = list(range(2, 11)) + [20]
        for n in nlst:
            self.partial_wakes[f"axiwake{n}"] = fm.partial_wakes.PartialAxiwake(n)
        for n in nlist:
            self.partial_wakes[f"grid{n**2}"] = fm.partial_wakes.PartialGrid(n=n)

        self.wake_frames = Dict(
            name="wake_frames",
            rotor_wd=fm.wake_frames.RotorWD(var_wd=FV.WD),
            rotor_wd_farmo=fm.wake_frames.FarmOrder(),
            yawed=fm.wake_frames.YawedWakes(),
            yawed_k002=fm.wake_frames.YawedWakes(k=0.02),
            yawed_k004=fm.wake_frames.YawedWakes(k=0.04),
        )
        stps = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
        for s in stps:
            self.wake_frames[f"streamlines_{int(s)}"] = fm.wake_frames.Streamlines2D(
                step=s
            )
        for s in stps:
            self.wake_frames[f"streamlines_{int(s)}_yawed"] = fm.wake_frames.YawedWakes(
                base_frame=fm.wake_frames.Streamlines2D(step=s)
            )
        for s in stps:
            self.wake_frames[f"streamlines_{int(s)}_farmo"] = fm.wake_frames.FarmOrder(
                base_frame=fm.wake_frames.Streamlines2D(step=s)
            )
        dtlist = [
            ("1s", 1 / 60),
            ("10s", 1 / 6),
            ("30s", 0.5),
            ("1min", 1),
            ("10min", 10),
            ("30min", 30),
        ]
        self.wake_frames["timelines"] = fm.wake_frames.Timelines()
        for s, t in dtlist:
            self.wake_frames[f"timelines_{s}"] = fm.wake_frames.Timelines(dt_min=t)
        self.wake_frames["timelines_1km"] = fm.wake_frames.Timelines(
            max_wake_length=1000.0
        )
        self.wake_frames["seq_dyn_wakes"] = fm.wake_frames.SeqDynamicWakes()
        for s, t in dtlist:
            self.wake_frames[f"seq_dyn_wakes_{s}"] = fm.wake_frames.SeqDynamicWakes(
                dt_min=t
            )

        self.wake_superpositions = Dict(
            name="wake_superpositions",
            ws_linear=fm.wake_superpositions.WSLinear(scale_amb=False),
            ws_linear_lim=fm.wake_superpositions.WSLinear(
                scale_amb=False, lim_low=1e-4
            ),
            ws_linear_amb=fm.wake_superpositions.WSLinear(scale_amb=True),
            ws_linear_amb_lim=fm.wake_superpositions.WSLinear(
                scale_amb=True, lim_low=1e-4
            ),
            ws_quadratic=fm.wake_superpositions.WSQuadratic(scale_amb=False),
            ws_quadratic_lim=fm.wake_superpositions.WSQuadratic(
                scale_amb=False, lim_low=1e-4
            ),
            ws_quadratic_amb=fm.wake_superpositions.WSQuadratic(scale_amb=True),
            ws_quadratic_amb_lim=fm.wake_superpositions.WSQuadratic(
                scale_amb=True, lim_low=1e-4
            ),
            ws_cubic=fm.wake_superpositions.WSPow(pow=3, scale_amb=False),
            ws_cubic_amb=fm.wake_superpositions.WSPow(pow=3, scale_amb=True),
            ws_quartic=fm.wake_superpositions.WSPow(pow=4, scale_amb=False),
            ws_quartic_amb=fm.wake_superpositions.WSPow(pow=4, scale_amb=True),
            ws_max=fm.wake_superpositions.WSMax(scale_amb=False),
            ws_max_amb=fm.wake_superpositions.WSMax(scale_amb=True),
            ws_product=fm.wake_superpositions.WSProduct(),
            ws_product_lim=fm.wake_superpositions.WSProduct(lim_low=1e-4),
            ti_linear=fm.wake_superpositions.TILinear(superp_to_amb="quadratic"),
            ti_quadratic=fm.wake_superpositions.TIQuadratic(superp_to_amb="quadratic"),
            ti_cubic=fm.wake_superpositions.TIPow(pow=3, superp_to_amb="quadratic"),
            ti_quartic=fm.wake_superpositions.TIPow(pow=4, superp_to_amb="quadratic"),
            ti_max=fm.wake_superpositions.TIMax(superp_to_amb="quadratic"),
        )

        self.axial_induction = Dict(name="induction_models")
        self.axial_induction["Betz"] = fm.axial_induction_models.BetzAxialInduction()
        self.axial_induction["Madsen"] = (
            fm.axial_induction_models.MadsenAxialInduction()
        )

        self.wake_models = Dict(name="wake_models")
        slist = [
            "linear",
            "linear_lim",
            "linear_amb",
            "linear_amb_lim",
            "quadratic",
            "quadratic_lim",
            "quadratic_amb",
            "quadratic_amb_lim",
            "cubic",
            "cubic_amb",
            "quartic",
            "quartic_amb",
            "wmax",
            "max_amb",
            "product",
            "product_lim",
        ]
        for s in slist:
            self.wake_models[f"Jensen_{s}"] = fm.wake_models.wind.JensenWake(
                superposition=f"ws_{s}"
            )
            self.wake_models[f"Jensen_{s}_k002"] = fm.wake_models.wind.JensenWake(
                k=0.02, superposition=f"ws_{s}"
            )
            self.wake_models[f"Jensen_{s}_k004"] = fm.wake_models.wind.JensenWake(
                k=0.04, superposition=f"ws_{s}"
            )
            self.wake_models[f"Jensen_{s}_k007"] = fm.wake_models.wind.JensenWake(
                k=0.07, superposition=f"ws_{s}"
            )
            self.wake_models[f"Jensen_{s}_k0075"] = fm.wake_models.wind.JensenWake(
                k=0.075, superposition=f"ws_{s}"
            )

            self.wake_models[f"Bastankhah2014_{s}"] = (
                fm.wake_models.wind.Bastankhah2014(
                    superposition=f"ws_{s}", sbeta_factor=0.2
                )
            )
            self.wake_models[f"Bastankhah2014_{s}_k002"] = (
                fm.wake_models.wind.Bastankhah2014(
                    k=0.02, sbeta_factor=0.2, superposition=f"ws_{s}"
                )
            )
            self.wake_models[f"Bastankhah2014_{s}_k004"] = (
                fm.wake_models.wind.Bastankhah2014(
                    k=0.04, sbeta_factor=0.2, superposition=f"ws_{s}"
                )
            )

            self.wake_models[f"Bastankhah2014B_{s}"] = (
                fm.wake_models.wind.Bastankhah2014(
                    superposition=f"ws_{s}", sbeta_factor=0.2, induction="Betz"
                )
            )
            self.wake_models[f"Bastankhah2014B_{s}_k002"] = (
                fm.wake_models.wind.Bastankhah2014(
                    k=0.02, sbeta_factor=0.2, superposition=f"ws_{s}", induction="Betz"
                )
            )
            self.wake_models[f"Bastankhah2014B_{s}_k004"] = (
                fm.wake_models.wind.Bastankhah2014(
                    k=0.04, sbeta_factor=0.2, superposition=f"ws_{s}", induction="Betz"
                )
            )

            self.wake_models[f"Bastankhah025_{s}"] = fm.wake_models.wind.Bastankhah2014(
                superposition=f"ws_{s}", sbeta_factor=0.25
            )
            self.wake_models[f"Bastankhah025_{s}_k002"] = (
                fm.wake_models.wind.Bastankhah2014(
                    k=0.02, superposition=f"ws_{s}", sbeta_factor=0.25
                )
            )
            self.wake_models[f"Bastankhah025_{s}_k004"] = (
                fm.wake_models.wind.Bastankhah2014(
                    k=0.04, superposition=f"ws_{s}", sbeta_factor=0.25
                )
            )

            self.wake_models[f"Bastankhah025B_{s}"] = (
                fm.wake_models.wind.Bastankhah2014(
                    superposition=f"ws_{s}", sbeta_factor=0.25, induction="Betz"
                )
            )
            self.wake_models[f"Bastankhah025B_{s}_k002"] = (
                fm.wake_models.wind.Bastankhah2014(
                    k=0.02, superposition=f"ws_{s}", sbeta_factor=0.25, induction="Betz"
                )
            )
            self.wake_models[f"Bastankhah025B_{s}_k004"] = (
                fm.wake_models.wind.Bastankhah2014(
                    k=0.04, superposition=f"ws_{s}", sbeta_factor=0.25, induction="Betz"
                )
            )

            self.wake_models[f"Bastankhah2016_{s}"] = (
                fm.wake_models.wind.Bastankhah2016(superposition=f"ws_{s}")
            )
            self.wake_models[f"Bastankhah2016_{s}_k002"] = (
                fm.wake_models.wind.Bastankhah2016(superposition=f"ws_{s}", k=0.02)
            )
            self.wake_models[f"Bastankhah2016_{s}_k004"] = (
                fm.wake_models.wind.Bastankhah2016(superposition=f"ws_{s}", k=0.04)
            )

            self.wake_models[f"Bastankhah2016B_{s}"] = (
                fm.wake_models.wind.Bastankhah2016(
                    superposition=f"ws_{s}", induction="Betz"
                )
            )
            self.wake_models[f"Bastankhah2016B_{s}_k002"] = (
                fm.wake_models.wind.Bastankhah2016(
                    superposition=f"ws_{s}", k=0.02, induction="Betz"
                )
            )
            self.wake_models[f"Bastankhah2016B_{s}_k004"] = (
                fm.wake_models.wind.Bastankhah2016(
                    superposition=f"ws_{s}", k=0.04, induction="Betz"
                )
            )

            self.wake_models[f"TurbOPark_{s}_A002"] = fm.wake_models.wind.TurbOParkWake(
                A=0.02, superposition=f"ws_{s}"
            )
            self.wake_models[f"TurbOPark_{s}_A004"] = fm.wake_models.wind.TurbOParkWake(
                A=0.04, superposition=f"ws_{s}"
            )

            self.wake_models[f"TurbOParkB_{s}_A002"] = (
                fm.wake_models.wind.TurbOParkWake(
                    A=0.02, superposition=f"ws_{s}", induction="Betz"
                )
            )
            self.wake_models[f"TurbOParkB_{s}_A004"] = (
                fm.wake_models.wind.TurbOParkWake(
                    A=0.04, superposition=f"ws_{s}", induction="Betz"
                )
            )

            As = [0.02, 0.04]
            dxs = [0.01, 1.0, 5.0, 10.0, 50.0, 100.0]
            for A in As:
                for dx in dxs:
                    a = str(A).replace(".", "")
                    d = str(dx).replace(".", "") if dx < 1 else int(dx)
                    self.wake_models[f"TurbOParkIX_{s}_A{a}_dx{d}"] = (
                        fm.wake_models.wind.TurbOParkWakeIX(
                            A=A, superposition=f"ws_{s}", dx=dx
                        )
                    )

        slist = ["linear", "quadratic", "cubic", "quartic", "max"]
        for s in slist:
            self.wake_models[f"CrespoHernandez_{s}"] = (
                fm.wake_models.ti.CrespoHernandezTIWake(superposition=f"ti_{s}")
            )
            self.wake_models[f"CrespoHernandez_ambti_{s}"] = (
                fm.wake_models.ti.CrespoHernandezTIWake(
                    superposition=f"ti_{s}", use_ambti=True
                )
            )
            self.wake_models[f"CrespoHernandez_{s}_k002"] = (
                fm.wake_models.ti.CrespoHernandezTIWake(k=0.02, superposition=f"ti_{s}")
            )

            self.wake_models[f"IECTI2005_{s}"] = fm.wake_models.ti.IECTIWake(
                superposition=f"ti_{s}", iec_type="2005"
            )

            self.wake_models[f"IECTI2019_{s}"] = fm.wake_models.ti.IECTIWake(
                superposition=f"ti_{s}", iec_type="2019"
            )

        self.wake_models[f"RHB"] = fm.wake_models.induction.RankineHalfBody()
        self.wake_models[f"Rathmann"] = fm.wake_models.induction.Rathmann()
        self.wake_models[f"SelfSimilar"] = fm.wake_models.induction.SelfSimilar()
        self.wake_models[f"SelfSimilar2020"] = (
            fm.wake_models.induction.SelfSimilar2020()
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
            axial_induction=self.axial_induction,
        )
        self.base_classes = Dict(
            name="base_classes",
            point_models=PointDataModel,
            rotor_models=RotorModel,
            turbine_types=TurbineType,
            turbine_models=TurbineModel,
            farm_models=FarmDataModel,
            farm_controllers=FarmController,
            partial_wakes=PartialWakesModel,
            wake_frames=WakeFrame,
            wake_superpositions=WakeSuperposition,
            wake_models=WakeModel,
            axial_induction=AxialInductionModel,
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
        subset: list of str, optional
            Selection of model types
        search:  str, optional
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

    def get(self, model_type, name, class_name=None, *args, **kwargs):
        """
        Gets a model object.

        If not found, dynamically creates it (given the class name)

        Parameters
        ----------
        model_type: str
            The model type
        name: str
            The model name
        class_name: str, optinal
            Name of the model class
        args: tuple, optional
            Arguments for the model class
        kwargs: dict, optional
            Arguments for the model class

        Returns
        -------
        model: mclass
            The model object

        """
        if name not in self.sources[model_type]:
            if class_name is None:
                raise KeyError(
                    f"Model '{name}' of type '{model_type}' not found in model book. Available: {sorted(list(self.sources[model_type].keys()))}"
                )
            bclass = self.base_classes[model_type]
            self.sources[model_type][name] = bclass.new(class_name, *args, **kwargs)
        return self.sources[model_type][name]

    def default_partial_wakes(self, wake_model):
        """
        Gets a default partial wakes model name
        for a given wake model

        Parameters
        ----------
        wake_model: foxes.core.WakeModel
            The wake model

        Returns
        -------
        pwake: str
            The partial wake model name

        """
        if isinstance(wake_model, TurbineInductionModel):
            return "grid9"
        elif isinstance(wake_model, fm.wake_models.TopHatWakeModel):
            return "top_hat"
        elif isinstance(wake_model, fm.wake_models.AxisymmetricWakeModel):
            return "axiwake6"
        elif isinstance(wake_model, fm.wake_models.DistSlicedWakeModel):
            return "grid9"
        else:
            raise TypeError(
                f"No default partial wakes model defined for wake model type '{type(wake_model).__name__}'"
            )

    def finalize(self, algo, verbosity=0):
        """
        Finalizes the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        """
        for ms in self.sources.values():
            if isinstance(ms, Dict):
                for m in ms.values():
                    if m.initialized:
                        m.finalize(algo, verbosity)
