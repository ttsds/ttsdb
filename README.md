# TTSDB

TTSDB is a monorepo of small, installable Python packages:

- `core/` (`ttsdb-core`): shared base classes and utilities
- `models/*/` (`ttsdb_<model>`): model-specific adapters that may vendor upstream research code in `src/<pkg>/_vendor/`

## Quickstart

- **List models**:
  - `just models`
- **Create a new model package from templates**:
  - Start pinned to one minor line (recommended when you only know one working version):
    - `just init "MyModel" python_venv=3.11`
  - Or specify an explicit support range:
    - `just init "MyModel" python_venv=3.11 python_requires=">=3.10,<3.12"`
- **Set up a model dev environment**:
  - `just setup maskgct cpu`
  - Override the interpreter used for the venv: `just setup maskgct python=3.11`

## Python version policy (per model)

Model packages often depend on upstream research code with tight version constraints. Each model’s `config.yaml` supports:

- `dependencies.python`: **supported versions** as a PEP 440 specifier string (e.g. `">=3.10,<3.12"` or `"==3.10.*"`)
- `dependencies.python_tested`: optional **explicit list of tested interpreters** (human-facing)
- `dependencies.python_venv`: **concrete interpreter version** used by `just setup` to create the venv (override via `just setup <model> python=...`)

The model’s `pyproject.toml` should mirror `dependencies.python` via `requires-python`.

