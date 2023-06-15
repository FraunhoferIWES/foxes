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
