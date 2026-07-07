import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def import_module(
    name: str,
    package: str | None = None,
    pip_hint: str | None = None,
    conda_hint: str | None = None,
) -> ModuleType:
    """
    Imports a module dynamically.

    Parameters
    ----------
    name: str
        The module name
    package: str, optional
        The explicit package name, deduced from name
        if not given
    pip_hint: str, optional
        Installation advice, in case the import fails
    conda_hint: str, optional
        Installation advice, in case the import fails

    Returns
    -------
    mdl: module
        The imported package

    :group: utils

    """
    try:
        return importlib.import_module(name, package)
    except ModuleNotFoundError:
        mdl = name if package is None else f"{package}.{name}"
        piph = pip_hint if pip_hint is not None else f"pip install {name}"
        cndh = (
            conda_hint
            if conda_hint is not None
            else f"conda install {name} -c conda-forge"
        )
        hts = " or ".join([f"'{h}'" for h in [piph, cndh] if len(h)])
        raise ModuleNotFoundError(f"Module '{mdl}' not found, maybe try {hts}")


def load_module(name: str, path: str | Path) -> ModuleType:
    """
    Imports a module from a file path

    Parameters
    ----------
    name: str
        The name of the module
    path: str
        The path to the python file

    Returns
    -------
    module:
        The module object

    :group: utils

    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Failed to create import spec for module '{name}' at '{path}'"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module
