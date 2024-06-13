import numpy as np

from .dict import Dict


class Factory:
    """
    Constructs objects from a choice of allowed
    constructor parameters

    Attributes
    ----------
    base: class
        The class of which objects are to be created
    name_template: str
        The name template, e.g. 'name_<A>_<B>_<C>' for
        variables A, B, C
    args: tuple
        Fixed arguments for the base class
    kwargs: dict
        Fixed arguments for the base class
    var2arg: dict
        Mapping from variable to constructor argument
    hints: dict
        Hints for print_toc, only for variables for which the
        options are functions or missing
    options: dict
        For each variable, e.g. A, B or C, the list or dict
        or function that maps a str to the actual value

    :group: utils

    """

    def __init__(
        self,
        base,
        name_template,
        args=(),
        kwargs={},
        var2arg={},
        hints={},
        **options,
    ):
        """
        Constructor.

        Parameters
        ----------
        base: class
            The class of which objects are to be created
        name_template: str
            The name template, e.g. 'name_<A>_<B>_<C>' for
            variables A, B, C
        args: tuple
            Fixed arguments for the base class
        kwargs: dict
            Fixed arguments for the base class
        var2arg: dict
            Mapping from variable to constructor argument
        hints: dict
            Hints for print_toc, only for variables for which the
            options are functions or missing
        options: dict
            For each variable, e.g. A, B or C, the list or dict
            or function that maps a str to the actual value

        """
        self.base = base
        self.name_template = name_template
        self.args = args
        self.kwargs = kwargs
        self.var2arg = var2arg
        self.hints = hints

        parts = name_template.split(">")
        if len(parts) < 2:
            raise ValueError(
                f"Factory '{name_template}': Expecting at least one variable in template, pattern '<..>'"
            )

        self._vars = []
        self._pre = []
        for i, p in enumerate(parts):
            if i < len(parts) - 1:
                parts2 = p.split("<")
                if len(parts2) != 2:
                    raise ValueError(
                        f"Factory '{name_template}': incomplete pattern brackets '<..>' between variables, e.g. '_'"
                    )
                if i > 0 and len(parts2[0]) == 0:
                    raise ValueError(
                        f"Factory '{name_template}': Missing seperator like '_' in template between variables '{self._vars[-1]}' and '{parts[1]}'"
                    )
                self._pre.append(parts2[0])
                self._vars.append(parts2[1])
            else:
                self._pre.append(p)

        if len(self.variables) > 1:
            for vi, v in enumerate(self.variables):
                p = self._pre[vi]
                if vi < len(self.variables) - 1 and p == "":
                    raise ValueError(
                        f"Factory '{name_template}': Require indicator before variable '{v}' in template, e.g. '{v}<{v}>'"
                    )

        self.options = Dict(name=f"{self._pre[0]}_options")
        for v, o in options.items():
            if v not in self.variables:
                raise KeyError(
                    f"Factory '{name_template}': Variable '{v}' found in options, but not in template"
                )
            if isinstance(o, list) or isinstance(o, tuple):
                o = {str(k): k for k in o}
            if isinstance(o, dict):
                for k in o.keys():
                    if not isinstance(k, str):
                        raise TypeError(
                            f"Factory '{name_template}': Found option for variable '{v}' that is not a str, {k}"
                        )
                self.options[v] = Dict(name=f"{self._pre[0]}_options_{v}", **o)
            elif hasattr(o, "__call__"):
                self.options[v] = o
            else:
                raise ValueError(
                    f"Factory '{name_template}': Variable '{v}' has option of type '{type(v).__name__}'. Only list, tuple, dict or function are supported"
                )

    @property
    def name_prefix(self):
        """
        The beginning of the name template

        Returns
        -------
        nbase: str
            The beginning of the name template

        """
        return self._pre[0]

    @property
    def name_suffix(self):
        """
        The ending of the name template

        Returns
        -------
        nbase: str
            The ending of the name template

        """
        return self._pre[-1]

    @property
    def variables(self):
        """
        The list of variables

        Returns
        -------
        vrs: list of str
            The variables

        """
        return self._vars

    def __str__(self):
        """String representation"""
        s = f"{self.name_template}: {self.base.__name__} with"
        for k, d in self.kwargs.items():
            s += f"\n  {k}={d}"
        for v in self.variables:
            if v in self.options and isinstance(self.options[v], dict):
                s += f"\n  {v} from {list(self.options[v])}"
            else:
                s += f"\n  {v}={self.hints.get(v, '(value)')}"
        return s

    def check_match(self, name):
        """
        Tests if a name matches the template

        Parameters
        ----------
        name: str
            The name to be checked

        Returns
        -------
        success: bool
            True if the template is matched

        """
        data_str = name
        for vi in range(len(self.variables)):
            p = self._pre[vi]
            i = data_str.find(p)
            j = i + len(p)
            if i < 0 or len(data_str) <= j:
                return False
            data_str = data_str[j:]

            q = self._pre[vi + 1]
            if q != "":
                i = data_str.find(q)
                j = i + len(q)
                if i < 0 or len(data_str) <= j:
                    return False
            else:
                data_str = ""

        return True

    def construct(self, name):
        """
        Create an object of the base class.

        Parameters
        ----------
        name: str
            The name, matching the template

        Returns
        -------
        obj: object
            The instance of the base class

        """
        j = 0
        wlist = []
        for pi, p in enumerate(self._pre):
            if len(p) > 0:
                i = name[j:].find(p)
                if i < 0 or (pi == 0 and i > 0):
                    raise ValueError(
                        f"Factory '{self.name_template}': Name '{name}' not matching template"
                    )
                w = name[j : j + i]
                j += i + len(p)
            else:
                w = name[j:]
            if pi > 0:
                wlist.append(w)

        kwargs = {}
        for vi, v in enumerate(self.variables):
            w = self.var2arg.get(v, v)
            data = wlist[vi]
            if v in self.options:
                o = self.options[v]
                if hasattr(o, "__call__"):
                    kwargs[w] = o(data)
                else:
                    kwargs[w] = self.options[v][data]
            else:
                kwargs[w] = data

        kwargs.update(self.kwargs)

        return self.base(*self.args, **kwargs)


class FDict(Dict):
    """
    A dictionary with factory support

    Attributes
    ----------
    store_created: bool
        Flag for storing created objects
    factories: list of foxes.utils.Factory
        The factories

    :group: utils

    """

    def __init__(self, *args, store_created=True, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Parameters for the base class
        store_created: bool
            Flag for storing created objects
        kwargs: dict, optional
            Parameters for the base class

        """
        super().__init__(*args, **kwargs)
        self.store_created = store_created
        self.factories = []

    def add_factory(self, *args, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Parameters for the Factory constructor
        kwargs: dict, optional
            Parameters for the Factory constructor

        """
        f = Factory(*args, **kwargs)
        i = len(self.factories)
        for gi in range(len(self.factories) - 1, -1, -1):
            g = self.factories[gi]
            if (
                g.name_prefix == f.name_prefix
                and g.name_suffix == f.name_suffix
                and len(f.variables) > len(g.variables)
            ):
                i = gi

        if i == len(self.factories):
            self.factories.append(f)
        else:
            self.factories.insert(i, f)

    def __contains__(self, key):
        found = super().__contains__(key)
        if not found:
            for f in self.factories:
                if f.check_match(key):
                    return True
        return found

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            for f in self.factories:
                try:
                    obj = f.construct(key)
                    if self.store_created:
                        self[key] = obj
                    return obj
                except ValueError:
                    pass

        k = ", ".join(sorted(list(self.keys())))
        e = f"{self.name}: Cannot find key '{key}', also no factory matches. Known keys: {k}. Known factories: {[f.name_template for f in self.factories]}"
        raise KeyError(e)
