# foxes example: _timelines_

Shows dynamic wake development for timeseries states input.

## Check options
Check options by
```
python3 run.py -h
```

## Run command
```
python3 run.py
```

Select the original two-panel flow/REWS animation through model-book outputs:

```console
uv run run.py -nf -o create_turbine_gif
uv run run.py -nf -o farm_results create_turbine_mp4
```

These write `ani_WS.gif` or `ani_WS.mp4`; MP4 requires ffmpeg.
`-v 1` announces creation. The default `-v 2` adds `Writing <path>` when encoding
begins, successful completion, frame counts and encoder settings; `-v 3`
adds per-frame composition and encoding progress. `-v 0` is quiet.
`-F/--fps` controls playback. `-nf` does not suppress requested animation files.
The old `-a` and `-A` options are removed.

## Output selection and automated runs

The default output selection is `farm_results create_turbine_gif`. Use `-o NAME [NAME ...]` to choose model-book outputs; `-o farm_results_write_n2` is a writer-only example. Reporting that needs unselected fields is skipped with a message. `--nofig` suppresses display without disabling explicitly selected file outputs.

```console
python run.py --nofig
```

The runner executes this example once, in a temporary directory. Its quick mode uses bounded local inputs and a local engine; generated files are removed.

<!-- run-all: python run.py -->
<!-- run-all-quick: python run.py -e numpy -nt 4 -S 3 -o farm_results -->

Verbosity (`-v`/`--verbosity`) defaults to `-v 2` for reports. Levels `2` and `3` print each collected `farm_results` dataset once, even when fields needed by later reports are absent. Levels `0` and `1` do not print datasets; writer-only selections do not invent farm results. Use `0` for quiet execution, `1` for a concise summary, or `3` for library progress. The aggregate runner passes `-v 0` by default, still announces each example, and accepts `-v N` (or `--verbosity N`); explicit example-command choices take precedence.
