from .load import import_module


def print_mem(obj, min_csize=0, max_csize=None, pre_str="OBJECT SIZE"):
    """
    Prints the memory consumption of a model and its components

    Parmeters
    ---------
    obj: object
        The object to be analyzed
    min_csize: int
        The minimal size of a component for being shown
    max_csize: int, optional
        The maximal allowed size of a component
    pre_str: str
        String to be printed before

    :group: utils

    """
    objsize = import_module("objsize")
    n = obj.name if hasattr(obj, "name") else ""
    print(pre_str, type(obj).__name__, n, objsize.get_deep_size(obj))
    for k in dir(obj):
        o = None
        try:
            if (
                hasattr(obj, k)
                and not callable(getattr(obj, k))
                and (len(k) < 3 or k[:2] != "__")
            ):
                o = getattr(obj, k)
        except ValueError:
            pass

        if o is not None:
            s = objsize.get_deep_size(getattr(obj, k))
            if s >= min_csize:
                print("   ", k, s)
                if max_csize is not None and s > max_csize:
                    raise ValueError(f"Component {k} exceeds maximal size {max_csize}")
