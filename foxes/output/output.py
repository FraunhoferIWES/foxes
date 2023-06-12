from foxes.utils import PandasFileHelper, all_subclasses


class Output:
    """
    Base class for foxes output.

    The job of this class is to provide handy
    helper functions.

    :group: output

    """

    @classmethod
    def write(cls, file_path, data, format_col2var={}, format_dict={}, **kwargs):
        """
        Writes data to file via pandas.

        The kwargs are forwarded to the underlying pandas writing function.

        Parameters
        ----------
        file_path: string
            The path to the output file
        data: pandas.DataFrame
            The data
        format_col2var: dict
            Mapping from column names to flappy variables,
            for formatting
        format_dict: dict
            Dictionary with format entries for columns, e.g.
            {FV.P: '{:.4f}'}. Note that the keys are flappy variables

        """
        fdict = {}
        for c in data.columns:
            v = format_col2var.get(c, c)
            if v in format_dict:
                fdict[c] = format_dict[v]
            elif v in PandasFileHelper.DEFAULT_FORMAT_DICT:
                fdict[c] = PandasFileHelper.DEFAULT_FORMAT_DICT[v]

        PandasFileHelper.write_file(data, file_path, fdict, **kwargs)

    @classmethod
    def print_models(cls):
        """
        Prints all model names.
        """
        names = sorted([scls.__name__ for scls in all_subclasses(cls)])
        for n in names:
            print(n)

    @classmethod
    def new(cls, model_type, **kwargs):
        """
        Run-time output model factory.

        Parameters
        ----------
        model_type: string
            The selected derived class name

        """

        if model_type is None:
            return None

        allc = all_subclasses(cls)
        found = model_type in [scls.__name__ for scls in allc]

        if found:
            for scls in allc:
                if scls.__name__ == model_type:
                    return scls(**kwargs)

        else:
            estr = "Output type '{}' is not defined, available types are \n {}".format(
                model_type, sorted([i.__name__ for i in allc])
            )
            raise KeyError(estr)
