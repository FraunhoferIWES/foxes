import foxes


def test_downwind_print_deco_includes_ground_models(capsys):
    farm = foxes.WindFarm()
    farm.add_turbine(foxes.Turbine([0.0, 0.0], turbine_models=["NREL5MW"]), verbosity=0)

    algo = foxes.algorithms.Downwind(
        farm,
        foxes.input.states.SingleStateStates(ws=8.0, wd=270.0, ti=0.08, rho=1.225),
        wake_models=["Jensen_linear_k007"],
        ground_models={"Jensen_linear_k007": "ground_mirror"},
        verbosity=1,
    )

    algo.initialize()
    algo.print_deco()

    output = capsys.readouterr().out
    assert "  ground models:" in output
    assert "Jensen_linear_k007: ground_mirror" in output
