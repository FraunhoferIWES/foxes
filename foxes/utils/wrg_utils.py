import pandas as pd

class ReaderWRG():

    def __init__(self, file_name):
        self._file_name = file_name
        self._prepare()

    def _prepare(self):
        # read the first two lines
        with open(self._file_name) as fstream:
            self._nx, self._ny, self._utmx0, self._utmy0, self._res = fstream.readline().split()
            second_line = fstream.readline().split()
            n_cols = len(second_line)
            self._n_sectors = int(second_line[8])
            self._nx, self._ny, self._utmx0, self._utmy0, self._res \
                = int(self._nx), int(self._ny), float(self._utmx0), float(self._utmy0), int(self._res)

        cols_sel = lambda name, secs : [f"{name}_{i}" for i in range(secs)]

        cols = [0] * (8+3*self._n_sectors)
        cols[0] = "utmx"
        cols[1] = "utmy"
        cols[2] = "z"
        cols[3] = "h"
        cols[4] = "A"
        cols[5] = "K"
        cols[6] = "pw"
        cols[7] = "n_sectors"
        cols[8::3] = cols_sel("fs", self._n_sectors)
        cols[9::3] = cols_sel("As", self._n_sectors)
        cols[10::3] = cols_sel("Ks", self._n_sectors)

        self._data = pd.read_csv(self._file_name, names=cols, skiprows=1,
                                 sep='\s+', usecols=range(1, n_cols))

        self._data[cols_sel("fs", self._n_sectors)] /= 10
        self._data[cols_sel("As", self._n_sectors)] /= 10
        self._data[cols_sel("Ks", self._n_sectors)] /= 100

    def data(self):
        return self._data
