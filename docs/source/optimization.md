# Optimization

Starting with *foxes* version 0.8, the optimization functionalities have been moved to the separate package [foxes-opt](https://github.com/FraunhoferIWES/foxes-opt). This has to be installed in addition to the main package:

```console
pip install foxes[opt]
```

or

```console
pip install foxes-opt
```

or

```console
conda install foxes-opt -c conda-forge
```

Compared to older versions, replace `foxes.opt` by `foxes_opt` in all your scripts - then everything should just run as before.

Optimization examples can be found [here](https://fraunhoferiwes.github.io/foxes-opt/examples.html) in the `foxes-opt` [documentation](https://fraunhoferiwes.github.io/foxes-opt).
