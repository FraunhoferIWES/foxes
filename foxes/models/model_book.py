from math import sqrt
import foxes.models as fm
import foxes.variables as FV
from foxes.utils import FDict

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
    GroundModel,
)


class ModelBook:
    """
    Container for all kinds of models.

    Attributes
    ----------
    point_models: foxes.utils.FDict
        The point models. Keys: model name str,
        values: foxes.core.PointDataModel
    rotor_models: foxes.utils.FDict
        The rotor models. Keys: model name str,
        values: foxes.core.RotorModel
    turbine_types: foxes.utils.FDict
        The turbine type models. Keys: model name str,
        values: foxes.core.TurbineType
    turbine_models: foxes.utils.FDict
        The turbine models. Keys: model name str,
        values: foxes.core.TurbineModel
    farm_models: foxes.utils.FDict
        The farm models. Keys: model name str,
        values: foxes.core.FarmModel
    farm_controllers: foxes.utils.FDict
        The farm controllers. Keys: model name str,
        values: foxes.core.FarmController
    partial_wakes: foxes.utils.FDict
        The partial wakes. Keys: model name str,
        values: foxes.core.PartialWakeModel
    wake_frames: foxes.utils.FDict
        The wake frames. Keys: model name str,
        values: foxes.core.WakeFrame
    wake_superpositions: foxes.utils.FDict
        The wake superposition models. Keys: model name str,
        values: foxes.core.WakeSuperposition
    wake_models: foxes.utils.FDict
        The wake models. Keys: model name str,
        values: foxes.core.WakeModel
    induction_models: foxes.utils.FDict
        The induction models. Keys: model name str,
        values: foxes.core.AxialInductionModel
    ground_models: foxes.utils.FDict
        The ground models. Keys: model name str,
        values: foxes.core.GroundModel
    sources: foxes.utils.FDict
        All sources dict
    base_classes: foxes.utils.FDict
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
        self.point_models = FDict(name="point_models")
        self.point_models["tke2ti"] = fm.point_models.TKE2TI()

        self.rotor_models = FDict(name="rotor_models")
        rvars = [FV.REWS, FV.REWS2, FV.REWS3, FV.TI, FV.RHO]
        self.rotor_models["centre"] = fm.rotor_models.CentreRotor(calc_vars=rvars)

        def _n2n(n2):
            n2 = float(n2)
            n = int(sqrt(n2))
            if n**2 != n2:
                raise Exception(
                    f"GridRotor factory: Value {n2} is not the square of an integer"
                )
            return n

        self.rotor_models.add_factory(
            fm.rotor_models.GridRotor,
            "grid<n2>",
            kwargs=dict(calc_vars=rvars, reduce=True),
            var2arg={"n2": "n"},
            n2=_n2n,
            hints={"n2": "(Number of points in square grid)"},
        )
        self.rotor_models.add_factory(
            fm.rotor_models.GridRotor,
            "grid<n2>_raw",
            kwargs=dict(calc_vars=rvars, reduce=False),
            var2arg={"n2": "n"},
            n2=_n2n,
            hints={"n2": "(Number of points in square grid)"},
        )
        self.rotor_models.add_factory(
            fm.rotor_models.LevelRotor,
            "level<n>",
            kwargs=dict(calc_vars=rvars, reduce=True),
            n=lambda x: int(x),
            hints={"n": "(Number of vertical levels)"},
        )
        self.rotor_models.add_factory(
            fm.rotor_models.LevelRotor,
            "level<n>_raw",
            kwargs=dict(calc_vars=rvars, reduce=False),
            n=lambda x: int(x),
            hints={"n": "(Number of vertical levels)"},
        )

        self.turbine_types = FDict(name="turbine_types")
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

        self.turbine_models = FDict(
            name="turbine_models",
            kTI=fm.turbine_models.kTI(),
            kTI_amb=fm.turbine_models.kTI(ti_var=FV.AMB_TI),
            thrust2ct=fm.turbine_models.Thrust2Ct(),
            PMask=fm.turbine_models.PowerMask(),
            yaw2yawm=fm.turbine_models.YAW2YAWM(),
            yawm2yaw=fm.turbine_models.YAWM2YAW(),
        )
        self.turbine_models.add_factory(
            fm.turbine_models.kTI,
            "kTI_<kTI>",
            kTI=lambda x: float(f"0.{x[1:]}" if x[0] == "0" else float(x)),
            hints={"kTI": "(Value, e.g. 004 for 0.04)"},
        )
        self.turbine_models.add_factory(
            fm.turbine_models.kTI,
            "kTI_amb_<kTI>",
            kwargs=dict(ti_var=FV.AMB_TI),
            kTI=lambda x: float(f"0.{x[1:]}" if x[0] == "0" else float(x)),
            hints={"kTI": "(Value, e.g. 004 for 0.04)"},
        )
        self.turbine_models.add_factory(
            fm.turbine_models.kTI,
            "kTI_<kTI>_<kb>",
            kTI=lambda x: float(f"0.{x[1:]}" if x[0] == "0" else float(x)),
            kb=lambda x: float(f"0.{x[1:]}" if x[0] == "0" else float(x)),
            hints={
                "kTI": "(Value, e.g. 004 for 0.04)",
                "kb": "(Value, e.g. 004 for 0.04)",
            },
        )
        self.turbine_models.add_factory(
            fm.turbine_models.kTI,
            "kTI_amb_<kTI>_<kb>",
            kwargs=dict(ti_var=FV.AMB_TI),
            kTI=lambda x: float(f"0.{x[1:]}" if x[0] == "0" else float(x)),
            kb=lambda x: float(f"0.{x[1:]}" if x[0] == "0" else float(x)),
            hints={
                "kTI": "(Value, e.g. 004 for 0.04)",
                "kb": "(Value, e.g. 004 for 0.04)",
            },
        )

        self.turbine_models["hubh_data"] = fm.turbine_models.RotorCentreCalc(
            {
                f"{FV.WD}_HH": FV.WD,
                f"{FV.WS}_HH": FV.WS,
                f"{FV.TI}_HH": FV.TI,
                f"{FV.RHO}_HH": FV.RHO,
            }
        )

        self.farm_models = FDict(
            name="farm_models",
            **{
                f"farm_{mname}": fm.farm_models.Turbine2FarmModel(m)
                for mname, m in self.turbine_models.items()
            },
        )

        self.farm_controllers = FDict(
            name="farm_controllers",
            basic_ctrl=fm.farm_controllers.BasicFarmController(),
        )

        self.partial_wakes = FDict(
            name="partial_wakes",
            rotor_points=fm.partial_wakes.RotorPoints(),
            top_hat=fm.partial_wakes.PartialTopHat(),
            centre=fm.partial_wakes.PartialCentre(),
        )
        self.partial_wakes.add_factory(
            fm.partial_wakes.PartialAxiwake,
            "axiwake<n>",
            n=lambda x: int(x),
            hints={"n": "(Number of evaluation points)"},
        )
        self.partial_wakes.add_factory(
            fm.partial_wakes.PartialGrid,
            "grid<n2>",
            var2arg={"n2": "n"},
            n2=_n2n,
            hints={"n2": "(Number of points in square grid)"},
        )

        self.wake_frames = FDict(
            name="wake_frames",
            rotor_wd=fm.wake_frames.RotorWD(var_wd=FV.WD),
            rotor_wd_farmo=fm.wake_frames.FarmOrder(),
        )
        self.wake_frames.add_k_factory(
            fm.wake_frames.YawedWakes,
            "yawed_[wake_k]",
        )
        self.wake_frames.add_factory(
            fm.wake_frames.Streamlines2D,
            "streamlines_<step>",
            step=lambda x: float(x),
            hints={"step": "(Step size in m)"},
        )
        self.wake_frames.add_factory(
            fm.wake_frames.Streamlines2D,
            "streamlines_<step>",
            step=lambda x: float(x),
            hints={"step": "(Step size in m)"},
        )

        self.wake_frames["timelines"] = fm.wake_frames.Timelines()

        def _todt(x):
            if x[-1] == "s":
                return float(x[:-1]) / 60
            elif x[-3:] == "min":
                return float(x[:-3])

        self.wake_frames.add_factory(
            fm.wake_frames.Timelines,
            "timelines_<dt>",
            dt=_todt,
            var2arg={"dt": "dt_min"},
            hints={"dt": "(Time step, e.g '10s', '1min' etc.)"},
        )
        self.wake_frames.add_factory(
            fm.wake_frames.SeqDynamicWakes,
            "seq_dyn_wakes_<dt>",
            dt=_todt,
            var2arg={"dt": "dt_min"},
            hints={"dt": "(Time step, e.g '10s', '1min' etc.)"},
        )

        self.wake_superpositions = FDict(
            name="wake_superpositions",
            ws_linear=fm.wake_superpositions.WSLinear(scale_amb=False),
            ws_linear_lim=fm.wake_superpositions.WSLinear(
                scale_amb=False, lim_low=1e-4
            ),
            ws_linear_amb=fm.wake_superpositions.WSLinear(scale_amb=True),
            ws_linear_amb_lim=fm.wake_superpositions.WSLinear(
                scale_amb=True, lim_low=1e-4
            ),
            ws_linear_loc=fm.wake_superpositions.WSLinearLocal(),
            ws_linear_loc_lim=fm.wake_superpositions.WSLinearLocal(lim_low=1e-4),
            ws_quadratic=fm.wake_superpositions.WSQuadratic(scale_amb=False),
            ws_quadratic_lim=fm.wake_superpositions.WSQuadratic(
                scale_amb=False, lim_low=1e-4
            ),
            ws_quadratic_amb=fm.wake_superpositions.WSQuadratic(scale_amb=True),
            ws_quadratic_amb_lim=fm.wake_superpositions.WSQuadratic(
                scale_amb=True, lim_low=1e-4
            ),
            ws_quadratic_loc=fm.wake_superpositions.WSQuadraticLocal(),
            ws_quadratic_loc_lim=fm.wake_superpositions.WSQuadraticLocal(lim_low=1e-4),
            ws_cubic=fm.wake_superpositions.WSPow(pow=3, scale_amb=False),
            ws_cubic_amb=fm.wake_superpositions.WSPow(pow=3, scale_amb=True),
            ws_cubic_loc=fm.wake_superpositions.WSPowLocal(pow=3),
            ws_cubic_loc_lim=fm.wake_superpositions.WSPowLocal(pow=3, lim_low=1e-4),
            ws_quartic=fm.wake_superpositions.WSPow(pow=4, scale_amb=False),
            ws_quartic_amb=fm.wake_superpositions.WSPow(pow=4, scale_amb=True),
            ws_quartic_loc=fm.wake_superpositions.WSPowLocal(pow=4),
            ws_quartic_loc_lim=fm.wake_superpositions.WSPowLocal(pow=4, lim_low=1e-4),
            ws_max=fm.wake_superpositions.WSMax(scale_amb=False),
            ws_max_amb=fm.wake_superpositions.WSMax(scale_amb=True),
            ws_max_loc=fm.wake_superpositions.WSMaxLocal(),
            ws_max_loc_lim=fm.wake_superpositions.WSMaxLocal(lim_low=1e-4),
            ws_product=fm.wake_superpositions.WSProduct(),
            ws_product_lim=fm.wake_superpositions.WSProduct(lim_low=1e-4),
            ti_linear=fm.wake_superpositions.TILinear(superp_to_amb="quadratic"),
            ti_quadratic=fm.wake_superpositions.TIQuadratic(superp_to_amb="quadratic"),
            ti_cubic=fm.wake_superpositions.TIPow(pow=3, superp_to_amb="quadratic"),
            ti_quartic=fm.wake_superpositions.TIPow(pow=4, superp_to_amb="quadratic"),
            ti_max=fm.wake_superpositions.TIMax(superp_to_amb="quadratic"),
        )

        self.axial_induction = FDict(name="induction_models")
        self.axial_induction["Betz"] = fm.axial_induction.BetzAxialInduction()
        self.axial_induction["Madsen"] = fm.axial_induction.MadsenAxialInduction()

        self.wake_models = FDict(name="wake_models")

        self.wake_models.add_k_factory(
            fm.wake_models.wind.JensenWake,
            "Jensen_<superposition>_[wake_k]",
            kwargs=dict(induction="Betz"),
            superposition=lambda s: f"ws_{s}",
            hints={"superposition": "(Superposition, e.g. linear for ws_linear)"},
        )

        self.wake_models.add_k_factory(
            fm.wake_models.wind.Bastankhah2014,
            "Bastankhah2014_<superposition>_[wake_k]",
            kwargs=dict(sbeta_factor=0.2, induction="Madsen"),
            superposition=lambda s: f"ws_{s}",
            hints={"superposition": "(Superposition, e.g. linear for ws_linear)"},
        )
        self.wake_models.add_k_factory(
            fm.wake_models.wind.Bastankhah2014,
            "Bastankhah2014B_<superposition>_[wake_k]",
            kwargs=dict(sbeta_factor=0.2, induction="Betz"),
            superposition=lambda s: f"ws_{s}",
            hints={"superposition": "(Superposition, e.g. linear for ws_linear)"},
        )
        self.wake_models.add_k_factory(
            fm.wake_models.wind.Bastankhah2014,
            "Bastankhah025_<superposition>_[wake_k]",
            kwargs=dict(sbeta_factor=0.25, induction="Madsen"),
            superposition=lambda s: f"ws_{s}",
            hints={"superposition": "(Superposition, e.g. linear for ws_linear)"},
        )
        self.wake_models.add_k_factory(
            fm.wake_models.wind.Bastankhah2014,
            "Bastankhah025B_<superposition>_[wake_k]",
            kwargs=dict(sbeta_factor=0.25, induction="Betz"),
            superposition=lambda s: f"ws_{s}",
            hints={"superposition": "(Superposition, e.g. linear for ws_linear)"},
        )

        self.wake_models.add_k_factory(
            fm.wake_models.wind.Bastankhah2016,
            "Bastankhah2016_<superposition>_[wake_k]",
            kwargs=dict(induction="Madsen"),
            superposition=lambda s: f"ws_{s}",
            hints={"superposition": "(Superposition, e.g. linear for ws_linear)"},
        )
        self.wake_models.add_k_factory(
            fm.wake_models.wind.Bastankhah2016,
            "Bastankhah2016B_<superposition>_[wake_k]",
            kwargs=dict(induction="Betz"),
            superposition=lambda s: f"ws_{s}",
            hints={"superposition": "(Superposition, e.g. linear for ws_linear)"},
        )

        self.wake_models.add_k_factory(
            fm.wake_models.wind.TurbOParkWake,
            "TurbOPark_<superposition>_[wake_k]",
            kwargs=dict(induction="Madsen"),
            superposition=lambda s: f"ws_{s}",
            hints={"superposition": "(Superposition, e.g. linear for ws_linear)"},
        )
        self.wake_models.add_k_factory(
            fm.wake_models.wind.TurbOParkWake,
            "TurbOParkB_<superposition>_[wake_k]",
            kwargs=dict(induction="Betz"),
            superposition=lambda s: f"ws_{s}",
            hints={"superposition": "(Superposition, e.g. linear for ws_linear)"},
        )

        self.wake_models.add_k_factory(
            fm.wake_models.wind.TurbOParkWakeIX,
            "TurbOParkIX_<superposition>_[wake_k]_dx<dx>",
            superposition=lambda s: f"ws_{s}",
            dx=lambda x: float(x),
            hints={
                "superposition": "(Superposition, e.g. linear for ws_linear)",
                "dx": "(Integration step in m)",
            },
        )

        self.wake_models.add_k_factory(
            fm.wake_models.ti.CrespoHernandezTIWake,
            "CrespoHernandez_<superposition>_[wake_k]",
            kwargs=dict(use_ambti=False),
            superposition=lambda s: f"ti_{s}",
            hints={"superposition": "(Superposition, e.g. linear for ti_linear)"},
        )

        self.wake_models.add_factory(
            fm.wake_models.ti.IECTIWake,
            "IECTI2005_<superposition>",
            kwargs=dict(iec_type="2005"),
            superposition=lambda s: f"ti_{s}",
            hints={"superposition": "(Superposition, e.g. linear for ti_linear)"},
        )

        self.wake_models.add_factory(
            fm.wake_models.ti.IECTIWake,
            "IECTI2019_<superposition>",
            kwargs=dict(iec_type="2019"),
            superposition=lambda s: f"ti_{s}",
            hints={"superposition": "(Superposition, e.g. linear for ti_linear)"},
        )
        self.wake_models.add_k_factory(
            fm.wake_models.ti.IECTIWake,
            "IECTI2019k_<superposition>_[wake_k]",
            kwargs=dict(iec_type="2019", opening_angle=None),
            superposition=lambda s: f"ti_{s}",
            hints={"superposition": "(Superposition, e.g. linear for ti_linear)"},
        )

        self.wake_models.add_factory(
            fm.wake_models.ti.IECTIWake,
            "IECTI2005_<superposition>",
            kwargs=dict(iec_type="2005"),
            superposition=lambda s: f"ti_{s}",
            hints={"superposition": "(Superposition, e.g. linear for ti_linear)"},
        )
        self.wake_models.add_k_factory(
            fm.wake_models.ti.IECTIWake,
            "IECTI2005k_<superposition>_[wake_k]",
            kwargs=dict(iec_type="2005", opening_angle=None),
            superposition=lambda s: f"ti_{s}",
            hints={"superposition": "(Superposition, e.g. linear for ti_linear)"},
        )

        self.wake_models[f"RHB"] = fm.wake_models.induction.RankineHalfBody()
        self.wake_models[f"Rathmann"] = fm.wake_models.induction.Rathmann()
        self.wake_models[f"SelfSimilar"] = fm.wake_models.induction.SelfSimilar()
        self.wake_models[f"SelfSimilar2020"] = (
            fm.wake_models.induction.SelfSimilar2020()
        )
        self.wake_models[f"VortexSheet"] = fm.wake_models.induction.VortexSheet()

        self.ground_models = FDict(name="ground_models")
        self.ground_models["no_ground"] = fm.ground_models.NoGround()
        self.ground_models["ground_mirror"] = fm.ground_models.GroundMirror()
        self.ground_models.add_factory(
            fm.ground_models.WakeMirror,
            "blh_mirror_h<height>",
            var2arg={"height": "heights"},
            height=lambda h: [0.0, float(h)],
            hints={"height": "(Boundary layer wake reflection height)"},
        )

        self.sources = FDict(
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
            ground_models=self.ground_models,
        )
        self.base_classes = FDict(
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
            ground_models=GroundModel,
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
                    if isinstance(ms, FDict):
                        for f in ms.factories:
                            if search is None or search in f.name_template:
                                print()
                                print(f)
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
            if isinstance(ms, FDict):
                for m in ms.values():
                    if m.initialized:
                        m.finalize(algo, verbosity)
