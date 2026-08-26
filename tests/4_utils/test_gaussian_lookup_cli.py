from foxes.utils.create_gaussian_lookup import create_gaussian_lookup_artifact
from foxes.utils.create_gaussian_lookup import main
from foxes.utils.gaussian_lookup import AXIS_R_OVER_SIGMA
from foxes.utils.gaussian_lookup import AXIS_SIGMA_OVER_D
from foxes.utils.gaussian_lookup import load_lookup_dataset


def test_create_gaussian_lookup_artifact_writes_expected_dataset(tmp_path):
    out_file = tmp_path / "lookup_artifact.nc"

    out_path = create_gaussian_lookup_artifact(
        out_file=out_file,
        radial_resolution=0.1,
        sigma_over_d_min=0.02,
        sigma_over_d_max=1.0,
        sigma_resolution=0.05,
        sigma_spacing="linear",
        n_rho=96,
        version_tag="cli-test-v1",
        verbosity=0,
    )

    assert out_path == out_file
    assert out_file.is_file()

    ds = load_lookup_dataset(out_file)
    assert ds.attrs["version_tag"] == "cli-test-v1"
    assert ds.attrs["radial_resolution"] == 0.1
    assert ds.attrs["sigma_resolution"] == 0.05


def test_main_parses_args_and_writes_artifact(tmp_path):
    out_file = tmp_path / "from_main.nc"

    main(
        [
            str(out_file),
            "--radial-resolution",
            "0.1",
            "--r-over-sigma-max",
            "30.0",
            "--sigma-over-d-min",
            "0.02",
            "--sigma-over-d-max",
            "1.0",
            "--sigma-resolution",
            "0.05",
            "--sigma-spacing",
            "log",
            "--n-rho",
            "80",
            "--version-tag",
            "main-test-v1",
            "--complevel",
            "1",
            "-v",
            "0",
        ]
    )

    assert out_file.is_file()

    ds = load_lookup_dataset(out_file)
    assert ds.attrs["version_tag"] == "main-test-v1"
    assert ds.attrs["radial_resolution"] == 0.1
    assert ds.attrs["sigma_resolution"] == 0.05
    assert ds.attrs["axis_r_over_sigma_max"] == 30.0
    assert AXIS_R_OVER_SIGMA in ds.coords
    assert AXIS_SIGMA_OVER_D in ds.coords
