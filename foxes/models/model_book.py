"""Model registries, parameterized factories and model-book lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import foxes.models as fm
from foxes.core import (
    AxialInductionModel,
    FarmController,
    FarmDataModel,
    GroundModel,
    Output,
    PartialWakesModel,
    PointDataModel,
    RotorModel,
    TurbineModel,
    TurbineType,
    WakeDeflection,
    WakeFrame,
    WakeModel,
    WakeSuperposition,
)
from foxes.models.wake_models import TurbineInductionModel
from foxes.utils import FDict

from ._model_book_basics import (
    _populate_farm_models,
    _populate_point_models,
    _populate_rotor_models,
    _populate_turbine_models,
    _populate_turbine_types,
)
from ._model_book_outputs import _populate_outputs
from ._model_book_wakes import (
    _populate_axial_induction,
    _populate_gaussian_wakes,
    _populate_ground_models,
    _populate_induction_wakes,
    _populate_park_wakes,
    _populate_partial_wakes,
    _populate_reference_wakes,
    _populate_superpositions,
    _populate_turbulence_wakes,
    _populate_wake_deflections,
    _populate_wake_frames,
)

if TYPE_CHECKING:
    from foxes.core import Algorithm


class ModelBook:
    """Independent model registries and parameterized name factories.

    The ``outputs`` registry contains lightweight presets and factories for
    collectors, writers, animations, evaluation, layout, roses and slices.
    Configure selected outputs before algorithm initialization. Registry entries
    are model instances owned by this book, not shared between books.

    Attributes
    ----------
    point_models
        Point-field conversions and parameterized point models.
    rotor_models
        Rotor quadrature presets and point-count factories.
    turbine_types
        Turbine performance models, including any custom ``Pct`` preset.
    turbine_models
        Turbine-local scientific operations.
    farm_models
        Farm-wide wrappers around the turbine operations.
    farm_controllers
        Controllers selecting and ordering turbine operations.
    partial_wakes
        Partial-wake integration models and quadrature factories.
    wake_deflections
        Wake-deflection models and parameterized path integrations.
    wake_frames
        Spatial and time-dependent wake coordinate systems.
    wake_superpositions
        Wind-speed and turbulence combination rules.
    axial_induction
        Axial-induction models.
    wake_models
        Wake-deficit, turbulence and turbine-induction models.
    ground_models
        Ground and boundary-layer reflection models.
    outputs
        Constructor-defined consumers, collectors and file writers.
    sources
        Collection-name index referring to the registries above.
    base_classes
        Collection-name index of the base classes used for dynamic creation.
    """

    point_models: FDict
    rotor_models: FDict
    turbine_types: FDict
    turbine_models: FDict
    farm_models: FDict
    farm_controllers: FDict
    partial_wakes: FDict
    wake_deflections: FDict
    wake_frames: FDict
    wake_superpositions: FDict
    axial_induction: FDict
    wake_models: FDict
    ground_models: FDict
    outputs: FDict
    sources: FDict
    base_classes: FDict

    def __init__(self, Pct_file: str | None = None) -> None:
        """Create independent model registries and their parameterized factories.

        Parameters
        ----------
        Pct_file
            Power/ct curve path for an additional turbine-type preset named
            ``"Pct"``; ``None`` registers only the built-in turbine types.
        """
        _populate_point_models(self)
        _populate_rotor_models(self)
        _populate_turbine_types(self, Pct_file)
        _populate_turbine_models(self)
        _populate_farm_models(self)
        _populate_partial_wakes(self)
        _populate_wake_deflections(self)
        _populate_wake_frames(self)
        _populate_superpositions(self)
        _populate_axial_induction(self)
        _populate_reference_wakes(self)
        _populate_gaussian_wakes(self)
        _populate_park_wakes(self)
        _populate_turbulence_wakes(self)
        _populate_induction_wakes(self)
        _populate_ground_models(self)
        _populate_outputs(self)
        self._connect_registries()

    def _connect_registries(self) -> None:
        """Index all registries and assign preset names to their model instances."""
        self.sources = FDict(
            _name="sources",
            point_models=self.point_models,
            rotor_models=self.rotor_models,
            turbine_types=self.turbine_types,
            turbine_models=self.turbine_models,
            farm_models=self.farm_models,
            farm_controllers=self.farm_controllers,
            partial_wakes=self.partial_wakes,
            wake_frames=self.wake_frames,
            wake_deflections=self.wake_deflections,
            wake_superpositions=self.wake_superpositions,
            wake_models=self.wake_models,
            axial_induction=self.axial_induction,
            ground_models=self.ground_models,
            outputs=self.outputs,
        )
        self.base_classes = FDict(
            _name="base_classes",
            point_models=PointDataModel,
            rotor_models=RotorModel,
            turbine_types=TurbineType,
            turbine_models=TurbineModel,
            farm_models=FarmDataModel,
            farm_controllers=FarmController,
            partial_wakes=PartialWakesModel,
            wake_frames=WakeFrame,
            wake_deflections=WakeDeflection,
            wake_superpositions=WakeSuperposition,
            wake_models=WakeModel,
            axial_induction=AxialInductionModel,
            ground_models=GroundModel,
            outputs=Output,
        )

        for s in self.sources.values():
            for k, m in s.items():
                m.name = k

    def __getitem__(self, key: str) -> Any:
        """Return a model registry by its collection name.

        Parameters
        ----------
        key
            Collection name, such as ``"wake_models"`` or ``"outputs"``.

        Returns
        -------
        registry
            Mutable registry of model instances and parameterized factories.

        Raises
        ------
        KeyError
            If the collection does not exist.
        """
        return self.sources.get_item(key)

    def print_toc(
        self, subset: list[str] | None = None, search: str | None = None
    ) -> None:
        """Print registered models and factory templates.

        Parameters
        ----------
        subset
            Collection names to include; ``None`` includes every collection.
        search
            Substring required in a model name or factory template; ``None``
            disables filtering.
        """

        for collection in sorted(self.sources):
            if subset is not None and collection not in subset:
                continue
            registry = self.sources[collection]
            print(collection)
            print("-" * len(collection))
            found = False
            for name in sorted(registry):
                if search is None or search in name:
                    print(f"{name}: {registry[name]}")
                    found = True
            if isinstance(registry, FDict):
                for factory in registry.factories:
                    if search is None or search in factory.name_template:
                        if found:
                            print()
                        print(factory)
                        found = True
            if not found:
                print("(none)")
            print()

    def get(
        self,
        model_type: str,
        name: str,
        class_name: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Resolve a registered model or construct and register a new one.

        Parameterized factory names are resolved by the selected registry.
        Existing instances are returned without copying.

        Parameters
        ----------
        model_type
            Collection name, such as ``"outputs"``.
        name
            Registered model name or parameterized factory name.
        class_name
            Class name used to construct an unregistered model; ``None``
            requires an existing entry or matching factory.
        args
            Positional constructor arguments for a new model.
        kwargs
            Keyword constructor arguments for a new model.

        Returns
        -------
        model
            Model instance owned by this book.

        Raises
        ------
        KeyError
            If the collection is unknown or no model, factory or class resolves
            the requested name.
        """
        if name not in self.sources[model_type]:
            if class_name is None:
                raise KeyError(
                    f"Model '{name}' of type '{model_type}' not found in model book. Available: {sorted(list(self.sources[model_type].keys()))}"
                )
            bclass = self.base_classes[model_type]
            self.sources[model_type][name] = bclass.new(class_name, *args, **kwargs)
        return self.sources[model_type][name]

    def default_partial_wakes(self, wake_model: WakeModel) -> str:
        """Choose a partial-wake integration suitable for the wake geometry.

        Parameters
        ----------
        wake_model
            Wake model whose geometric family determines the integration.

        Returns
        -------
        partial_wakes
            Registered partial-wake model or factory name.

        Raises
        ------
        TypeError
            If the wake-model family has no default partial-wake integration.
        """
        if isinstance(wake_model, TurbineInductionModel):
            return "grid9"
        elif isinstance(wake_model, fm.wake_models.TopHatWakeModel):
            return "top_hat"
        elif isinstance(wake_model, fm.wake_models.GaussianWakeModel):
            return "gaussian_lookup"
        elif isinstance(wake_model, fm.wake_models.AxisymmetricWakeModel):
            return "axiwake6"
        elif isinstance(wake_model, fm.wake_models.DistSlicedWakeModel):
            return "grid9"
        else:
            raise TypeError(
                f"No default partial wakes model defined for wake model type '{type(wake_model).__name__}'"
            )

    def finalize(self, algo: Algorithm, verbosity: int = 0) -> None:
        """Finalize every initialized model in this book.

        Parameters
        ----------
        algo
            Algorithm supplying finalization context.
        verbosity
            Diagnostic verbosity; zero is silent.
        """
        for ms in self.sources.values():
            if isinstance(ms, FDict):
                for m in ms.values():
                    if m.initialized:
                        m.finalize(algo, verbosity)
