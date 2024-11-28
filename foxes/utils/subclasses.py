def all_subclasses(cls):
    """
    Searches all classes derived from some
    base class.

    Parameters
    ----------
    cls: class
        The base class

    Returns
    -------
    list of class:
        The derived classes

    :group: utils

    """
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]
    )


def new_cls(base_cls, cls_name):
    """
    Run-time class selector.

    Parameters
    ----------
    base_cls: object
        The base class
    cls_name: string
        Name of the class

    Returns
    -------
    cls: object
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

    else:
        estr = "Class '{}' not found, available classes derived from '{}' are \n {}".format(
            cls_name, base_cls.__name__, sorted([i.__name__ for i in allc])
        )
        raise KeyError(estr)


def new_instance(base_cls, cls_name, *args, **kwargs):
    """
    Run-time factory.

    Parameters
    ----------
    base_cls: object
        The base class
    cls_name: string
        Name of the class
    args: tuple, optional
        Additional parameters for the constructor
    kwargs: dict, optional
        Additional parameters for the constructor

    Returns
    -------
    obj: object
        The instance of the derived class

    :group: utils

    """

    cls = new_cls(base_cls, cls_name)
    if cls is None:
        return None
    else:
        return cls(*args, **kwargs)
