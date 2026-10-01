# FOXES Docstring Conventions

## Purpose And Authority

This document is the canonical docstring standard for FOXES production Python.
It expands the repository policy in [AGENTS.md](../AGENTS.md) and uses the terms
and types from [naming conventions](naming-conventions.md). Architecture and
runtime ownership remain defined in [architecture](architecture.md).

FOXES uses NumPy-style docstrings rendered by numpydoc and Sphinx AutoAPI. Type
information belongs in annotations and is not repeated in docstring section
labels. Docstrings describe the current contract only. Under the forward-only
policy, do not document removed behavior, deprecated aliases, or a legacy path
unless an explicitly authorized exception still exists.

Existing source contains older variations. They are not precedents for new
work. When a public API is touched, migrate its affected docstrings directly to
this standard without preserving obsolete wording or formatting.

## Required Coverage

Document every public:

- module that owns a distinct user-facing or extension concern;
- class and constructor;
- function and method;
- property;
- class method and static method; and
- abstract or protected extension hook that downstream subclasses are expected
	to implement or call.

A symbol is public when it is exported by a package, documented as an extension
contract, used by examples or configuration, or intentionally available to
downstream callers without a private underscore prefix. Public model-book names,
YAML-facing constructors, and console-script implementation functions require
the same care even when most callers reach them indirectly.

Private helpers need a docstring when their contract, dimensions, mutation,
algorithm, or failure behavior is not obvious from the signature and code. Do
not add empty narration to a trivial private helper.

### Overrides And Inheritance

- An override that changes behavior, accepted values, side effects, errors,
	return semantics, lifecycle, or dimensions has its own complete docstring.
- A pure pass-through override with exactly the inherited contract may inherit
	the base docstring. Do not copy the same prose into both places.
- An abstract method documents the obligations of implementations, including
	required shapes, variables, mutation, and return values. A concrete override
	documents only additional or changed behavior, while remaining understandable
	through the generated inheritance view.
- An overload set has one authoritative docstring on the implementation. Explain
	the behavior selected by each overload condition without duplicating type
	annotations.

## Structure And Style

Use triple double quotes. Start with a concise one-line summary:

- Use an imperative verb for an operation: "Calculate farm results."
- Use a noun phrase for a class, property, or exposed value: "The active engine."
- State the domain result, not the implementation technique.
- End the summary with a period.

Add a blank line and extended prose only when it clarifies behavior, scientific
meaning, lifecycle, mutation, ordering, or constraints. Keep paragraphs focused
and use the exact FOXES vocabulary from `docs/naming-conventions.md`.

Use these sections in this order, omitting sections that do not apply:

1. `Parameters`
2. `Attributes`
3. `Returns` or `Yields`
4. `Raises`
5. `Warns`
6. `See Also`
7. `Notes`
8. `References`
9. `Examples`

Section headings use the NumPy style exactly:

```text
Parameters
----------
```

Use reStructuredText inside Python docstrings:

- Write literals, parameter names, FOXES constants, and short code expressions
	with double backticks, for example ``FC.STATE`` and ``calc_farm()``.
- Use resolvable Sphinx roles such as `:class:` or `:meth:` only when a real
	cross-reference helps the reader.
- Use `:math:` for inline mathematics and a `.. math::` block for displayed
	equations.
- Do not use Markdown links or headings inside a Python docstring.

## Parameters

List parameters in signature order. Use the exact parameter name without a type
suffix; omit `self` and `cls`. Write variadic parameters as `args` and `kwargs`,
matching the name without the leading asterisks.

```text
Parameters
----------
algo
    The calculation algorithm.
fdata
    Farm data with leading dimensions ``(FC.STATE, FC.TURBINE)``.
verbosity
    The verbosity level, where ``0`` is silent.
```

Parameter prose describes:

- domain meaning and role;
- accepted semantics that annotations cannot express;
- physical units;
- dimensions, shapes, coordinates, ordering, and required `FC`/`FV` fields;
- whether the value is read, mutated, retained, or consumed;
- relationships or exclusivity with other parameters; and
- behavior of meaningful sentinel values such as ``None``.

Do not repeat the annotation, write `optional`, or restate a literal default
that is already clear in the signature. Explain a default only when its behavior
or scientific meaning is not obvious. For a boolean, describe what enabling it
does instead of writing "Boolean flag."

Distinguish a model instance from a model-book name. If a parameter resolves a
registered name, identify the collection and whether parameterized factory names
are accepted. Do not imply that an instance-only API accepts `str`.

## Attributes

Use `Attributes` for public state that callers inspect or set directly and that
is not already fully represented by documented properties. Describe mutation,
units, and lifecycle validity. Do not list private caches or implementation
state.

Class prose explains the concept and invariants. Constructor arguments belong
in the `__init__` docstring, following the established AutoAPI configuration
that combines class and constructor content. Do not duplicate constructor
parameters in the class docstring.

## Returns And Yields

Omit `Returns` for a function that always returns `None`. Otherwise, use a
semantic result name, not a type name:

```text
Returns
-------
farm_results
    Farm results with dimensions ``(FC.STATE, FC.TURBINE)`` and the requested
    output variables.
```

- Never use `int`, `dict`, `Dataset`, or another annotation as the return entry
	name. Types are rendered from the signature.
- For multiple positional returns, document one semantic entry per returned
	component in order.
- For a mapping, describe key meaning, value meaning, array dimensions, and
	ownership or copying behavior.
- For an xarray result, name significant dimensions, coordinates, variables,
	ordering guarantees, and relevant attributes.
- For an iterator or generator, use `Yields` instead of `Returns` and describe
	one yielded element.
- If a method mutates an input or object and also returns it, state both facts.
- If a return can be `None`, explain the condition that produces `None`.

A property uses a noun-phrase summary and a semantic `Returns` entry matching
the exposed concept. A context manager documents the value returned by
`__enter__` and cleanup guaranteed by `__exit__` in the class or factory
docstring.

## Raises And Warnings

Document exceptions that are intentional parts of the public contract and that
a caller can reasonably prevent or handle. Use the exception class as the entry
and describe the exact condition:

```text
Raises
------
ValueError
    If the supplied dimensions do not start with
    ``(FC.STATE, FC.TURBINE)``.
KeyError
    If the requested model-book name is not registered.
```

Do not list incidental implementation exceptions, assertions, or impossible
states. If the implementation deliberately translates a lower-level exception,
document the public exception and preserve the original as its cause.

Use `Warns` only for emitted Python warnings, naming the warning class and its
condition. Do not use it for log messages, verbosity output, or general caveats;
put those in `Notes`.

## See Also, Notes, And References

Use `See Also` for a small set of directly related public APIs, with one short
reason for each relationship. Do not turn it into a package index.

Use `Notes` for details that matter after the basic contract is understood:

- scientific assumptions and equations;
- lifecycle or initialization requirements;
- chunk-local versus full-calculation behavior;
- numerical stability, NaN, clipping, or interpolation policy;
- state, turbine, target, or point ordering;
- mutation, caching, ownership, or thread/process considerations; and
- optional backend or dependency behavior.

Use `References` for scientific publications or external specifications that
define the implemented method. Give enough bibliographic information to locate
the source and use a stable DOI or URL where available. Do not cite a source
that merely mentions the method.

## Examples

Add `Examples` when correct use is not obvious from the signature, especially
for public factories, model selection, data shapes, context-managed engines, and
multi-step APIs. A trivial getter does not need an example.

Examples must:

- use the public import surface;
- be minimal, deterministic, and runnable;
- use synthetic or explicitly public packaged data;
- avoid network access, interactive input, and optional dependencies unless the
	requirement is the subject of the example;
- show the result or assertion that proves the behavior; and
- remain compatible with Python 3.10.

Use `>>>` prompts for short doctest-style examples. Longer workflows belong in
`examples/`, notebooks, or `docs/source/`; link them from `See Also` or prose
instead of embedding an unmaintainable script.

## FOXES Scientific Contracts

A FOXES docstring makes hidden scientific and array contracts explicit.

### Dimensions And Shapes

- Name dimensions with `FC` constants, not only informal axis numbers.
- State the leading dimensions and any fixed trailing component dimension.
- Distinguish internal target data
	``(FC.STATE, FC.TARGET, FC.TPOINT)`` from public point results using
	``FC.POINT``.
- State broadcasting, scalar expansion, subset selection, and ordering behavior
	when callers can observe it.
- For chunk-local APIs, explain which state or point subset is present and
	whether global indices are retained.

### Variables And Units

- Name physical fields with `FV` constants.
- State units for inputs, outputs, coordinates, and thresholds.
- Distinguish ambient `FV.AMB_*` values from waked values.
- Explain weighting, normalization, direction conventions, and NaN handling
	when relevant.
- Do not invent a new friendly name for an established FOXES variable.

### Lifecycle And Mutation

- State whether the object must be initialized or running.
- Describe data loaded during `initialize`, moved during `set_running`, restored
	during `unset_running`, or released during `finalize` when publicly relevant.
- Identify mutated containers and whether arrays are copied, viewed, cached, or
	stored for later chunks.
- State whether repeated calls are idempotent and whether callers may reuse the
	object after failure or finalization.

### Engines And Optional Integrations

- Keep scientific model docstrings backend-independent unless behavior truly
	depends on an engine.
- For engine APIs, document chunking, ordering, cleanup, serialization, and
	failure propagation that callers can observe.
- Name the required optional extra when an API depends on one and describe the
	error when it is unavailable.
- Do not imply that importing base `foxes` requires an optional integration.

## Special API Forms

### Abstract Methods And Callbacks

An abstract method defines what an implementation receives, may mutate, and
must return. Describe required variables and dimensions, not a specific current
subclass algorithm.

A callback parameter states its call signature in prose using the actual
semantic argument names, when it runs, what it may mutate, and whether its
return value is used. The annotation remains the type authority.

### Factories And Registries

Factory and registry docstrings distinguish:

- Python class names used by family-specific `new()` methods;
- fixed model-book keys;
- parameterized factory names; and
- instances supplied directly.

Describe caching, registration side effects, collision behavior, validation,
and the exception raised for an unknown name. Never suggest a universal
`Model.new()` API.

### File And Configuration Boundaries

Document accepted path kinds, formats, required fields, coordinate conventions,
and whether paths are resolved through FOXES configuration or `DataBook`.
Describe validation failures at the input boundary. For YAML/WindIO-facing
parameters, use the same name and meaning as the Python contract or identify the
adapter mapping explicitly.

## Templates

### Function Or Method

```python
def calculate_output(
    algo: Algorithm,
    fdata: FData,
    variable: str,
) -> np.ndarray:
    """Calculate one farm output variable.

    The calculation preserves state and turbine ordering and does not mutate
    ``fdata``.

    Parameters
    ----------
    algo
        The calculation algorithm.
    fdata
        Farm data with leading dimensions ``(FC.STATE, FC.TURBINE)``.
    variable
        The ``FV`` variable to calculate.

    Returns
    -------
    values
        Calculated values in the variable's documented units, with dimensions
        ``(FC.STATE, FC.TURBINE)``.

    Raises
    ------
    KeyError
        If ``variable`` is unavailable in ``fdata``.
    """
```

### Class And Constructor

```python
class ExampleModel(Model):
    """A model that calculates an example FOXES quantity."""

    def __init__(self, variable: str, scale: float = 1.0) -> None:
        """Initialize the example model.

        Parameters
        ----------
        variable
            The input ``FV`` variable.
        scale
            The dimensionless factor applied to the input values.
        """
```

### Property

```python
@property
def initialized(self) -> bool:
    """Whether the model has completed initialization.

    Returns
    -------
    initialized
        ``True`` after successful initialization and before finalization.
    """
```

Templates illustrate structure, not required wording. Replace every generic
phrase with the concrete scientific contract.

## Review Checklist

Before closing a change that touches a public API, confirm:

- [ ] The summary describes the current public contract.
- [ ] Signature names and docstring parameter names match exactly.
- [ ] Annotations contain types; prose does not duplicate them.
- [ ] Units, dimensions, `FC` axes, `FV` variables, and ordering are explicit.
- [ ] Mutation, lifecycle, chunking, caching, and optional dependencies are
      described where observable.
- [ ] Semantic returns, intentional exceptions, and warning conditions are
      complete.
- [ ] Examples use public APIs and runnable public or synthetic data.
- [ ] Removed behavior and legacy names are absent.
- [ ] Related base classes, overrides, API docs, examples, and architecture
      records remain consistent.

Run the closure commands in [development](development.md#development-closure).
A Sphinx build is not part of default closure. To inspect rendered docstrings or
AutoAPI pages explicitly, run:

```console
uv sync --extra test --extra doc
uv run sphinx-build -E -b html docs/source docs/build/html
```

Generated AutoAPI pages under `docs/source/_*` and HTML under `docs/build/` are
outputs, not sources. Inspect them, but do not edit or commit them.
