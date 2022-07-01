from foxes.input.states.states_table import StatesTable


class Timeseries(StatesTable):
    """
    Timeseries states data.
    """

    RDICT = {"index_col": 0, "parse_dates": [0]}
