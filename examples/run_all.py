"""

This is a script for automatically running all foxes
examples, which have to be located within the same
folder as this file, or a subset of them.

Please have a look at the list of options, by running

    python run_all.py --help

Example run:

    python run_all.py

The main purpose of this script is to provide an easy
tool for checking if all tutorials have been adopted
to the latest version of the code.

"""

import os
import argparse
from contextlib import contextmanager
import subprocess


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def clean():
    if os.path.isfile("clean.sh"):
        res = subprocess.run(
            ["bash", "clean.sh"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        ret = res.returncode
        if ret:
            print("\nERROR DURING CLEANING\n")
            print(res.stderr.decode("utf-8"))
            return 1
    return 0


def run_example(path, nofig):
    try:
        with cd(path):
            if os.path.isfile("README.md") or os.path.isfile("run.py"):
                print("Path:", os.getcwd())

                if not os.path.isfile("README.md"):
                    print("\nFILE 'README.md' NOT FOUND\n")
                    return 0

                commands = []
                with open("README.md") as f:
                    for line in f:
                        if "-h" not in line:
                            line = line.replace('"', "")
                            s = line.strip().split(" ")
                            if "run.py" in s:
                                i = s.index("run.py")
                                commands.append(["python"] + s[i:])
                                if nofig:
                                    commands[-1] += ["--nofig"]
                            elif "inputs.yaml" in s:
                                commands.append(s)
                                if nofig:
                                    commands[-1] += ["--nofig"]

                if len(commands) == 0:
                    print("\nNO COMMAND FOUND IN README.md\n")
                    return 0

                for cmd in commands:
                    print("  ", " ".join(cmd))

                    res = subprocess.run(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
                    ret = res.returncode
                    if ret:
                        print("\nERROR DURING RUN\n")
                        print(res.stderr.decode("utf-8"))
                        clean()
                        return 1

            return clean()

    except FileNotFoundError:
        print(f"\nFAILED TO ENTER DIRECTORY {path}\n")
        return 1


def run(args):
    incld = args.include
    excld = args.exclude

    counter = 0
    dryres = []
    for w in sorted(os.walk(".")):
        for d in sorted(w[1]):
            tdir = os.path.join(w[0], d)

            ok = True
            for k in excld:
                if k in tdir:
                    ok = False
                    break
            if ok and incld is not None:
                for k in incld:
                    if k not in tdir:
                        ok = False
                        break

            if ok:
                if counter >= args.step:
                    if not (args.dry or args.Dry):
                        print("\nEXAMPLE", counter)

                    if args.dry:
                        if os.path.isfile(os.path.join(tdir, "README.md")):
                            dryres.append(tdir)

                    elif args.Dry:
                        if os.path.isfile(os.path.join(tdir, "README.md")):
                            dryres.append(d)

                    elif run_example(tdir, args.nofig):
                        raise Exception(f"\nEXAMPLE {tdir} FAILED.")

                counter += 1

    if args.dry or args.Dry:
        for r in sorted(dryres):
            print(r)
    else:
        print("\nTutorials OK.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--include",
        help="Include keywords for directory names, or None for all",
        nargs="+",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-e",
        "--exclude",
        help="Exclude keywords for directory names",
        nargs="+",
        type=str,
        default=[],
    )
    parser.add_argument(
        "-o",
        "--incopt",
        help="Flag for inclusion of optimization cases",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-d",
        "--dry",
        help="Dry run, only print directories",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-D",
        "--Dry",
        help="Dry run, only print names",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-s", "--step", help="Set the start step, default=0", type=int, default=0
    )
    parser.add_argument("-nf", "--nofig", help="Skip all figures", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
