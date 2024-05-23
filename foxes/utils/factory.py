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

        i = name_template.find("<")
        if i < 0:
            raise ValueError(f"Factory '{name_template}': Expecting at least one variable in template, pattern '<..>'")

        self._bname = name_template[:i]
        pre = [""]
        wlist = [self._bname]
        while wlist[0][-1] == "_":
             wlist[0] =  wlist[0][:-1]

        def _find_var(tmpl):
            nonlocal wlist, pre
            i = tmpl.find(">")
            if i < 0:
                raise ValueError(f"Factory '{name_template}': Missing closing bracket '>' in '{tmpl}'")
            wlist.append(tmpl[:i])
            if i < len(tmpl) - 1:
                tmpl = tmpl[i+1:]
                j = tmpl.find("<")
                if j >= 0:
                    pre.append(tmpl[:j].replace("_", ""))
                    _find_var(tmpl[j+1:])
        
        _find_var(name_template[i+1:])
        self._wlist = wlist
        self._pre = pre
        
        self.options = Dict(name=f"{self.base_name}_options")
        for v, o  in options.items():
            if v not in self.variables:
                raise KeyError(f"Factory '{name_template}': Variable '{v}' found in options, but not in template")
            if isinstance(o, list) or isinstance(o, tuple):
                o = {str(k): k for k in o}
            if isinstance(o, dict):
                for k in o.keys():
                    if not isinstance(k, str):
                        raise TypeError(f"Factory '{name_template}': Found option for variable '{v}' that is not a str, {k}")
                self.options[v] = Dict(name=f"{self.base_name}_options_{v}", **o)
            elif hasattr(o, "__call__"):
                self.options[v] = o
            else:
                raise ValueError(f"Factory '{name_template}': Variable '{v}' has option of type '{type(v).__name__}'. Only list, tuple, dict or function are supported")

    @property
    def base_name(self):
        """ 
        The base name of the name template
        
        Returns
        -------
        name: str
            The base name
        
        """
        return self._wlist[0]
    
    @property
    def variables(self):
        """
        The list of variables
        
        Returns
        -------
        vrs: list of str
            The variables
        
        """
        return self._wlist[1:]
    
    def __str__(self):
        """ String representation """
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
        if len(name) < len(self._bname) or name[:len(self._bname)] != self._bname:
            return False
        
        wlist = name[len(self._bname):].split("_")
        if len(wlist) != len(self.variables):
            return False
        
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
        if len(name) < len(self._bname) or name[:len(self._bname)] != self._bname:
            raise ValueError(f"Factory '{self.name_template}': Name '{name}' not matching template")
        
        wlist = name[len(self._bname):].split("_")
        if len(wlist) != len(self.variables):
            raise ValueError(f"Factory '{self.name_template}': Name '{name}' not matching template")
        for i in range(len(wlist)):
            l = len(self._pre[i])
            if l > 0 and wlist[i][:l] != self._pre[i]:
                raise ValueError(f"Factory '{self.name_template}': Name '{name}' not matching template")
            wlist[i] = wlist[i][l:]

        kwargs = {}
        for i, v in enumerate(self.variables):
            w = self.var2arg.get(v, v)
            if v in self.options:
                o = self.options[v]
                if hasattr(o, "__call__"):
                    kwargs[w] = o(wlist[i])
                else:
                    kwargs[w] = self.options[v][wlist[i]]
            else:
                kwargs[w] = wlist[i]
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
        self.factories.append(Factory(*args, **kwargs))
    
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
                