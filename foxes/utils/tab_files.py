import numpy as np
from xarray import Dataset
from io import StringIO

def read_tab_file(fname, normalize=True):
    """
    Reads a tab file into a Dataset
    
    Parameters
    ----------
    fname: str
        The path to the tab file
    normalize: bool
        Normalize the frequencies such that
        they add to 1000 in each sector
    
    Returns
    -------
    out: xarray.Dataset
        The data

    :group: utils

    """

    header = []
    with open(fname, "r") as f:
        header.append(f.readline().replace("\t", " ").strip())
        header.append(f.readline().replace("\t", " ").strip())
        header.append(f.readline().replace("\t", " ").strip())
        s = "0 " + f.read().replace("\t", " ").strip()
    data = np.genfromtxt(StringIO(s))

    descr = header[0]
    lat, lon, height = np.fromstring(header[1], sep=' ')
    nsec, a, b = np.fromstring(header[2], sep=' ')

    delta_wd = 360/nsec
    out = Dataset(
        coords={
            "ws": data[:, 0],
            "wd": np.arange(0, 360., delta_wd)
        },
        data_vars={
            "frequency": (("ws", "wd"), data[:, 1:])
        },
        attrs={
            "description": descr,
            "latitude": lat,
            "longitude": lon,
            "height": height, 
            "delta_ws": a,
            "delta_wd": delta_wd,
            "shift_wd": b,
        }
    )

    if normalize:
        out["frequency"] *= 1000/np.sum(data[:, 1:], axis=0)[None, :]

    return out


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", help="Path to the tab file")
    args = parser.parse_args()

    read_tab_file(args.fname)