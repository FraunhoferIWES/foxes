import numpy as np

from .dict import Dict
import foxes.variables as FV


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
    example: str
        An example name
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
        example_vars=None,
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
        example_vars: dict, optional
            Variable values for creating an example
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

        self._vars = []
        self._pre = []
        parts = name_template.split(">")

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

        exvars = dict(
            n=5,
            n2=9,
            superposition="linear",
            kTI=0.2,
            kb=0.001,
            step=100,
            dx=100,
            dt="10s",
        )
        if example_vars is not None:
            exvars.update(example_vars)
        try:
            self.example = ""
            for i, v in enumerate(self._vars):
                self.example += f"{self._pre[i]}{exvars[v]}"
            self.example += self._pre[-1]
            if not self.check_match(self.example, error=False):
                raise ValueError(
                    f"Example '{self.example}' does not match template '{self.name_template}'"
                )
        except KeyError:
            self.example = None

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
        if self.example is not None:
            s += f"\nExample: {self.example}"
        return s

    def get_examples(self, **var_values):
        """
        Create example names from given values

        Parameters
        ----------
        var_values: dict
            Variables values. Key: Variable,
            value: list or value

        Returns
        -------
        examples: list of str
            The examples

        """

        def gete(i, vals, vars, values, examples):
            if i >= len(vars):
                e = ""
                for i, v in enumerate(self._vars):
                    e += f"{self._pre[i]}{vals[v]}"
                e += self._pre[-1]
                self.check_match(e, error=True)
                examples.append(e)
            else:
                v = vars[i]
                if v in self._vars:
                    vls = np.atleast_1d(values[i])
                    for x in vls:
                        vals[v] = x
                        gete(i + 1, vals, vars, values, examples)
                else:
                    gete(i + 1, vals, vars, values, examples)

        examples = []
        gete(
            0,
            {},
            list(var_values.keys()),
            list(var_values.values()),
            examples,
        )

        return examples

    def check_match(self, name, error=False, ret_pars=False):
        """
        Tests if a name matches the template and constructs
        parameters

        Parameters
        ----------
        name: str
            The name to be checked
        error: bool
            Flag for raising a Value error in case of
            mismatch
        ret_pars: bool
            Flag for returning the parameters

        Returns
        -------
        success: bool
            True if the template is matched
        pars: dict, optional
            The constructed parameters

        """
        j = 0
        wlist = []
        for pi, p in enumerate(self._pre):
            if len(p) > 0:
                i = name[j:].find(p)
                if i < 0 or (pi == 0 and i > 0):
                    if error:
                        raise ValueError(
                            f"Factory '{self.name_template}': Name '{name}' not matching template"
                        )
                    elif ret_pars:
                        return False, {}
                    else:
                        return False
                w = name[j : j + i]
                j += i + len(p)
            else:
                w = name[j:]
            if pi > 0:
                wlist.append(w)

        if ret_pars:
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
            return True, kwargs

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
        __, kwargs = self.check_match(name, error=True, ret_pars=True)
        return self.base(*self.args, **kwargs)


class WakeKFactory:
    """
    A factory that automatically handles
    wake_k parameters

    Attributes
    ----------
    factories: list of Factory
        The individual factories

    :group: utils

    """

    def __init__(self, base, name_template, *args, hints={}, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        base: class
            The class of which objects are to be created
        name_template: str
            The name template, e.g. 'name_<A>_<B>_<C>' for
            variables A, B, C. Indicate wake_k part by '_[wake_k]'
        args: tuple, optional
            Additional arguments for Factory
        hints: dict
            Hints for print_toc, only for variables for which the
            options are functions or missing
        kwargs: dict
            Additional arguments for Factory

        """
        self._base = base
        self._kwargs = kwargs
        self._template0 = name_template
        self.factories = []

        i0 = name_template.find("_[wake_k]")
        i1 = i0 + len("_[wake_k]")
        kw = kwargs.pop("kwargs", {})
        v2a = kwargs.pop("var2arg", {})

        if i0 < 0:
            raise ValueError(
                f"String '_[wake_k]' not found in name template '{name_template}'"
            )

        exvars = dict(k=0.04, ka=0.2, ambka=0.4, kb=0.001)
        if "example_vars" in kwargs:
            exvars.update(kwargs.pop("example_vars"))

        # add case ka, kb:
        t0 = name_template[:i0]
        t = t0 + "_ka<ka>_kb<kb>"
        if len(name_template) > i1:
            t += name_template[i1:]
        h = hints.copy()
        h["ka"] = "(Value, e.g. 04 for 0.4)"
        h["kb"] = "(Value, e.g. 001 for 0.01)"
        kw["ti_var"] = FV.TI
        self.factories.append(
            Factory(
                base,
                t,
                *args,
                hints=h,
                kwargs=kw.copy(),
                **kwargs,
                ka=lambda x: float(f"0.{x[1:]}" if x[0] == "0" else float(x)),
                kb=lambda x: float(f"0.{x[1:]}" if x[0] == "0" else float(x)),
                example_vars=exvars,
            )
        )

        # add case ambient ambka, kb:
        t = name_template[:i0] + "_ambka<ambka>_kb<kb>"
        if len(name_template) > i1:
            t += name_template[i1:]
        h = hints.copy()
        h["ambka"] = "(Value, e.g. 04 for 0.4)"
        h["kb"] = "(Value, e.g. 001 for 0.01)"
        kw["ti_var"] = FV.AMB_TI
        hv2a = v2a.copy()
        hv2a["ambka"] = "ka"
        self.factories.append(
            Factory(
                base,
                t,
                *args,
                hints=h,
                kwargs=kw.copy(),
                var2arg=hv2a,
                **kwargs,
                ambka=lambda x: float(f"0.{x[1:]}" if x[0] == "0" else float(x)),
                kb=lambda x: float(f"0.{x[1:]}" if x[0] == "0" else float(x)),
                example_vars=exvars,
            )
        )

        # add case ka:
        t = name_template[:i0] + "_ka<ka>"
        if len(name_template) > i1:
            t += name_template[i1:]
        h = hints.copy()
        h["ka"] = "(Value, e.g. 04 for 0.4)"
        kw["ti_var"] = FV.TI
        self.factories.append(
            Factory(
                base,
                t,
                *args,
                hints=h,
                kwargs=kw.copy(),
                **kwargs,
                ka=lambda x: float(f"0.{x[1:]}" if x[0] == "0" else float(x)),
                example_vars=exvars,
            )
        )

        # add case ambka:
        t = name_template[:i0] + "_ambka<ambka>"
        if len(name_template) > i1:
            t += name_template[i1:]
        h = hints.copy()
        h["ambka"] = "(Value, e.g. 04 for 0.4)"
        kw["ti_var"] = FV.AMB_TI
        hv2a = v2a.copy()
        hv2a["ambka"] = "ka"
        self.factories.append(
            Factory(
                base,
                t,
                *args,
                hints=h,
                kwargs=kw.copy(),
                var2arg=hv2a,
                **kwargs,
                ambka=lambda x: float(f"0.{x[1:]}" if x[0] == "0" else float(x)),
                example_vars=exvars,
            )
        )

        # add case k:
        t = name_template[:i0] + "_k<k>"
        if len(name_template) > i1:
            t += name_template[i1:]
        h = hints.copy()
        h["k"] = "(Value, e.g. 004 for 0.04)"
        kw["ti_var"] = FV.TI
        self.factories.append(
            Factory(
                base,
                t,
                *args,
                hints=h,
                kwargs=kw.copy(),
                **kwargs,
                k=lambda x: float(f"0.{x[1:]}" if x[0] == "0" else float(x)),
                example_vars=exvars,
            )
        )

        # add case without k:
        t = name_template[:i0]
        if len(name_template) > i1:
            t += name_template[i1:]
        kw["ti_var"] = FV.TI
        self.factories.append(
            Factory(
                base,
                t,
                *args,
                hints=hints,
                kwargs=kw.copy(),
                **kwargs,
                example_vars=exvars,
            )
        )

    def __str__(self):
        """String representation"""
        s = f"{self._template0}: {self._base.__name__} with"
        for k, d in self._kwargs.items():
            s += f"\n  {k}={d}"
        f0 = self.factories[-1]
        for v in f0.variables:
            if v in f0.options and isinstance(f0.options[v], dict):
                s += f"\n  {v} from {list(f0.options[v])}"
            else:
                s += f"\n  {v}={f0.hints.get(v, '(value)')}"
        s += (
            f"\n  [wake_k]=(None or k<k> or ka<ka> or ka<ka>_kb<kb>, e.g. 004 for 0.04)"
        )
        return s


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

    def add_factory(self, *args, factory=None, **kwargs):
        """
        Adds a Factory object.

        Parameters
        ----------
        args: tuple, optional
            Parameters for the Factory constructor
        factory: Factory, optional
            The factory object
        kwargs: dict, optional
            Parameters for the Factory constructor

        """
        f = Factory(*args, **kwargs) if factory is None else factory
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

    def add_k_factory(self, *args, **kwargs):
        """
        Adds a WakeKFactory.

        Parameters
        ----------
        args: tuple, optional
            Parameters for the Factory constructor
        kwargs: dict, optional
            Parameters for the Factory constructor

        """
        for f in WakeKFactory(*args, **kwargs).factories:
            self.add_factory(factory=f)

    def __contains__(self, key):
        found = super().__contains__(key)
        if not found:
            for f in self.factories:
                if f.check_match(key):
                    return True
        return found

    def get_item(self, key, prnt=True):
        """
        Gets an item, prints readable error if not found

        Parameters
        ----------
        key: immutable object
            The key
        prnt: bool
            Flag for message printing

        """
        try:
            return self[key]
        except KeyError as e:
            if prnt:
                print(
                    f"\n{self.name}: Cannot find key '{key}', also no factory matches.\n"
                )
                print("Known keys:")
                for k in self.keys():
                    print("   ", k)
                if len(self.factories):
                    print("\nKnown factories:")
                    for f in self.factories:
                        print("   ", f.name_template)
                print()
            raise e

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
