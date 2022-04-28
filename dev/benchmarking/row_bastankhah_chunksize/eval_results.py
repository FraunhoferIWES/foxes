import pandas as pd
import argparse
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rfile", help="The results csv", default="results.csv")
    parser.add_argument("-o", "--ofile", help="The output graphics file", default="results.png")
    parser.add_argument("-s", "--show", help="Show the plot", action="store_true")
    parser.add_argument("-t", "--title", help="The plot title", default=None)
    args = parser.parse_args()

    print("Reading file", args.rfile)
    data = pd.read_csv(args.rfile)
    print(data)

    fig, ax = plt.subplots(figsize=(8,4))

    t = args.title
    for s, g in data.groupby("scheduler"):

        g.sort_values('chunksize', axis=0, inplace=True)

        ax.plot(g['chunksize'], g['time'], label=s)
    
        if t is None:
            t = f"{g['n_states'].iloc[0]} states, 1 row, {g['n_turbines'].iloc[0]} turbines"

    ax.set_title(t)
    ax.set_xlabel('Chunk size')
    ax.set_ylabel('Time [s]')
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.ofile)

    if args.show:
        plt.show()

    