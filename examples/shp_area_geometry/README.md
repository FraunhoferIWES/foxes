# foxes example: shp_area_geometry

This example reads a shapefile into an AreaGeometry object and visualizes the loaded area.

## Check options
Check options by
```
python3 run.py -h
```

## Run command
Run with defaults (`data/area.shp` as input, no output file written):
```
python3 run.py --nofig
```

Run with an explicit shapefile path:
```
python3 run.py --shp_path /path/to/example.shp --output my_area.png
```

Convert to UTM with `-u` / `--to_utm` (the figure title includes the UTM zone):
```
python3 run.py -u -o my_area_utm.png
```

Set a custom title string:
```
python3 run.py -t "Buildable Areas" -o my_area.png
```

Combine multiple shapefiles by intersection (instead of default union):
```
python3 run.py -s "/path/to/areas/*.shp" -i -o my_area_intersection.png
```

If your shapefile uses another name column (e.g. `TYPE`):
```
python3 run.py --shp_path /path/to/example.shp --name_col TYPE
```

For the example buildable area data in this workspace, `TYPE` is the default `--name_col`.
