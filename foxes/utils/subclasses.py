from typing import Any, TypeVar, cast

T = TypeVar("T")


def all_subclasses(cls: type[Any]) -> set[type[Any]]:
    """
    Searches all classes derived from some
    base class.

    Parameters
    ----------
    cls
        The base class

    Returns
    -------
    The derived classes
        The derived classes

    :group: utils

    """
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]
    )


def new_cls(base_cls: type[Any], cls_name: str | None) -> type[Any] | None:
    """
    Run-time class selector.

    Parameters
    ----------
    base_cls
        The base class
    cls_name
        Name of the class

    Returns
    -------
    cls
        The derived class

    :group: utils

    """

    if cls_name is None:
        return None

    allc = all_subclasses(base_cls)
    found = cls_name in [scls.__name__ for scls in allc]

    if found:
        for scls in allc:
            if scls.__name__ == cls_name:
                return scls
        raise RuntimeError(
            f"Class lookup inconsistency for '{cls_name}' in '{base_cls.__name__}'"
        )

    else:
        estr = "Class '{}' not found, available classes derived from '{}' are \n {}".format(
            cls_name, base_cls.__name__, sorted([i.__name__ for i in allc])
        )
        raise KeyError(estr)


def new_instance(
    base_cls: type[T], cls_name: str | None, *args: Any, **kwargs: Any
) -> T | None:
    """
    Run-time factory.

    Parameters
    ----------
    base_cls
        The base class
    cls_name
        Name of the class
    args
        Additional parameters for the constructor
    kwargs
        Additional parameters for the constructor

    Returns
    -------
    obj
        The instance of the derived class

    :group: utils

    """

    cls = new_cls(base_cls, cls_name)
    if cls is None:
        return None
    else:
        return cast(T, cls(*args, **kwargs))
