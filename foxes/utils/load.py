import importlib


def import_module(name, package=None, hint=None):
    """
    Imports a module dynamically.

    Parameters
    ----------
    name: str
        The module name
    package: str, optional
        The explicit package name, deduced from name
        if not given
    hint: str, optional
        Installation advice, in case the import fails

    Returns
    -------
    mdl: module
        The imnported package

    :group: utils

    """
    try:
        return importlib.import_module(name, package)
    except ModuleNotFoundError:
        mdl = name if package is None else f"{package}.{name}"
        hint = hint if hint is not None else f"pip install {name}"
        raise ModuleNotFoundError(f"Module '{mdl}' not found, maybe try '{hint}'")
