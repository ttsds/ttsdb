## `ttsdb-core`

`ttsdb-core` provides the shared “adapter” surface that every model package implements.

### Key pieces

- **`VoiceCloningTTSBase`**: abstract base class for voice-cloning style TTS adapters.
  - Subclasses implement `_load_model()` and `_synthesize()` and use `synthesize()` for the public API.
- **`ModelConfig`**: helper for loading a model package’s bundled `config.yaml` (supports editable installs too).
- **Vendoring utilities**: helpers to make upstream research code importable at runtime:
  - `setup_vendor_path(package_name)`: prepends `src/<pkg>/_vendor/source[/code.root]` to `sys.path`
  - `vendor_context(...)`: optional context manager for research code that assumes a working directory or env vars

### Model package conventions

Each model package lives in `models/<name>/` and typically includes:

- `pyproject.toml`: packaging metadata (including `requires-python`)
- `config.yaml`: model metadata + dependency constraints + pointers to upstream code/weights
- `src/ttsdb_<name>/__init__.py`: the adapter implementation
- `tests/`: lightweight unit tests and optional `integration` tests (weights required)

### Python version constraints

Because upstream research code often depends on specific Python versions, model packages express **supported Python versions** in two places:

- `pyproject.toml` → `requires-python` (PEP 440 specifier string)
- `config.yaml` → `dependencies.python` (same specifier string)

For local development automation, `config.yaml` may also include:

- `dependencies.python_venv`: a concrete interpreter version used by `just setup` when creating a venv
- `dependencies.python_tested`: optional explicit list of tested interpreters (human-facing)

