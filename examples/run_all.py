"""Run each documented example once, directly in Python and in isolation."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import shlex
import sys
from shutil import copytree
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import yaml

from foxes.config import config
from foxes.config.conf import Config
from foxes.utils import Dict
import foxes.constants as FC


ROOT = Path(__file__).resolve().parent


def example_dirs(root=ROOT):
    """Return each immediate documented example directory once."""
    return sorted(path.parent for path in Path(root).glob("*/README.md"))


def command(path, quick=False):
    """Read exactly one explicit run-all command from an example's README."""
    marker = "<!-- run-all-quick:" if quick else "<!-- run-all:"
    entries = [
        line.strip()[len(marker) : -3].strip()
        for line in (Path(path) / "README.md").read_text().splitlines()
        if line.strip().startswith(marker) and line.strip().endswith("-->")
    ]
    if len(entries) != 1:
        raise ValueError(
            f"{path}: Expected exactly one {marker} command, got {len(entries)}"
        )
    return shlex.split(entries[0])


def _load(path):
    spec = importlib.util.spec_from_file_location(
        f"foxes_example_{path.parent.name}", path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"{path}: Not an importable Python script")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


@contextmanager
def isolated_example(path, nofig):
    """Restore process state and remove all example-owned inputs and outputs."""
    previous_dir = Path.cwd()
    previous_config = dict(config)
    previous_attributes = deepcopy(vars(config))
    previous_backend = plt.get_backend()
    previous_figures = set(plt.get_fignums())
    switch_backend = nofig and not previous_figures
    interactive = plt.isinteractive()
    if nofig:
        plt.ioff()
    if switch_backend:
        plt.switch_backend("Agg")
    try:
        with TemporaryDirectory(prefix="foxes-example-") as temporary:
            work = Path(temporary) / path.name
            copytree(
                path,
                work,
                ignore=lambda directory, names: [
                    name
                    for name in names
                    if name in {"results", "__pycache__", ".ipynb_checkpoints"}
                ],
            )
            os.chdir(work)
            try:
                # UTM ownership includes attributes as well as protected mapping keys.
                fresh = Config()
                dict.clear(config)
                dict.update(config, fresh)
                vars(config).clear()
                vars(config).update(vars(fresh))
                config[FC.WORK_DIR] = work
                config[FC.INPUT_DIR] = work
                config[FC.OUTPUT_DIR] = work / "results"
                config.output_dir.mkdir()
                yield work
            finally:
                os.chdir(previous_dir)
    finally:
        for number in set(plt.get_fignums()) - previous_figures:
            plt.close(number)
        if switch_backend:
            plt.switch_backend(previous_backend)
        plt.interactive(interactive)
        dict.clear(config)
        dict.update(config, previous_config)
        vars(config).clear()
        vars(config).update(previous_attributes)


def run_example(path, nofig=False, quick=False, verbosity=0):
    """Execute one README-selected command without subprocesses or source artifacts.

    Quick mode uses bounded local inputs and local engines, including synthetic
    ICON/EWW data and an inlined WindIO document. Normal mode retains tutorial
    input and engine defaults. Exceptions propagate with the example name.
    Scripts receive ``-v 0`` by default; an explicit ``-v`` or ``--verbosity``
    option takes precedence. The aggregate runner still prints one line per example.
    """
    path = Path(path).resolve()
    args = command(path, quick)
    print(f"Example {path.name}: {' '.join(args)}")
    with isolated_example(path, nofig) as work:
        if quick:
            _load(ROOT / "_quick_inputs.py").prepare_inputs(work, path.name)
        executable, *arguments = args
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        parser.add_argument("-v", "--verbosity", type=int)
        command_verbosity = parser.parse_known_args(arguments)[0].verbosity
        example_verbosity = (
            verbosity if command_verbosity is None else command_verbosity
        )
        if executable in {"python", "python3", "uv"}:
            if executable == "uv":
                if not arguments or arguments.pop(0) != "run":
                    raise ValueError(f"{path}: Expected uv run")
                if arguments and arguments[0] == "python":
                    arguments.pop(0)
            script, *arguments = arguments
            if nofig and "--nofig" not in arguments and "-nf" not in arguments:
                arguments.append("--nofig")
            if command_verbosity is None:
                arguments.extend(["-v", str(verbosity)])
            _load(work / script).main(arguments)
        elif executable == "foxes_yaml":
            from foxes.input.yaml.yaml import foxes_yaml

            foxes_yaml(arguments[0], nofig=nofig, verbosity=example_verbosity)
        elif executable == "foxes_windio":
            from foxes.input.yaml.dict import run_dict
            from foxes.input.yaml.windio.windio import foxes_windio, read_windio_dict

            if quick:
                data = Dict(yaml.safe_load((work / "quick_windio.yaml").read_text()))
                definition, algo, _ = read_windio_dict(
                    data, verbosity=example_verbosity
                )
                run_dict(
                    definition,
                    algo=algo,
                    nofig=nofig,
                    verbosity=example_verbosity,
                    engine_pars={"engine_type": "numpy", "chunk_size_states": 2},
                )
            else:
                foxes_windio(arguments[0], nofig=nofig, verbosity=example_verbosity)
        else:
            raise ValueError(f"{path}: Unsupported example command {executable!r}")
    return 0


def run(args):
    """Run each selected example exactly once, or list the selected names."""
    selected = [
        path
        for path in example_dirs()
        if not any(key in path.name for key in args.exclude)
        and (args.include is None or any(key in path.name for key in args.include))
    ][args.step :]
    for path in selected:
        if args.dry or args.Dry:
            print(path.name if args.Dry else path)
        else:
            run_example(path, args.nofig, args.quick, args.verbosity)
    if not (args.dry or args.Dry):
        print(f"Examples OK: {len(selected)} run once each.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--include", nargs="+")
    parser.add_argument("-e", "--exclude", nargs="+", default=[])
    parser.add_argument("-d", "--dry", action="store_true")
    parser.add_argument("-D", "--Dry", action="store_true")
    parser.add_argument("-s", "--step", type=int, default=0)
    parser.add_argument(
        "-nf", "--nofig", action="store_true", help="Do not show figures"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Use bounded local fixtures"
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=0,
        help="Example output: 0 quiet, 1 summary, 2 reports, 3 library progress",
    )
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
