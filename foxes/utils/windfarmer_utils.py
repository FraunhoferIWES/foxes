import numpy as np
from io import StringIO

def read_tab_file(fname):

    header = []
    with open(fname, "r") as f:
        header.append(f.readline().replace("\t", " ").strip())
        header.append(f.readline().replace("\t", " ").strip())
        header.append(f.readline().replace("\t", " ").strip())
        s = "0 " + f.read().replace("\t", " ").strip()
    data = np.genfromtxt(StringIO(s))

    print(header)
    descr = header[0]
    print(header[1])
    print(np.fromstring(header[1]), sep=' ')
    quit()
    lat, lon, height = np.fromstring(header[1])
    print(lat, lon, height)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", help="Path to the tab file")
    args = parser.parse_args()

    read_tab_file(args.fname)