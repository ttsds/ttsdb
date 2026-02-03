set dotenv-load
set shell := ["bash", "-euc"]

default_python := `cat .python-version`
hf_repo := env_var_or_default("TTSDB_HF_REPO", "ttsds")

models := `ls models`

# ---- Bootstrap & Development Setup ----

# Install all dependencies (uv, root venv, pre-commit hooks). Run once after cloning.
bootstrap:
    #!/usr/bin/env bash
    set -euo pipefail

    # Install uv if not present
    if ! command -v uv &> /dev/null; then
        echo "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        echo "✓ uv installed. You may need to restart your shell or run: source ~/.local/bin/env"
    else
        echo "✓ uv already installed ($(uv --version))"
    fi

    # Create root venv and install builder dependencies
    echo "Creating root venv..."
    uv venv
    source .venv/bin/activate
    echo "Installing builder dependencies..."
    uv pip install -r requirements.txt

    # Install pre-commit if not present
    if ! command -v pre-commit &> /dev/null; then
        echo "Installing pre-commit..."
        uv tool install pre-commit
    else
        echo "✓ pre-commit already installed ($(pre-commit --version))"
    fi

    # Install pre-commit hooks
    echo "Installing pre-commit hooks..."
    pre-commit install

    echo "✓ Bootstrap complete!"
    echo "  Activate root venv with: source .venv/bin/activate"

# Run linters (ruff check + format) via pre-commit
lint:
    pre-commit run --all-files

# Run unit tests for the core package
test-core:
        source .venv/bin/activate && \
            uv pip install "torch" "torchaudio" --index-url https://download.pytorch.org/whl/cpu && \
            uv pip install -e core/ && \
            pytest core/tests/ -v -rs

# Run unit tests in CI for all models (setup + test in each model's venv)
test-ci:
    #!/usr/bin/env bash
    set -euo pipefail
    for model in $(ls models); do
        echo "=== Testing $model ==="
        just setup "$model" cpu
        just test "$model"
    done

# List available models
models:
    @ls models

# Regenerate the repository README.md
readme:
    .venv/bin/python builder/root_readme.py

# Synthesize datasets across models
# Usage: just synth-dataset run --gpus 0,1 --models xtts vibevoice
synth-dataset *args:
    .venv/bin/python builder/synthesize_dataset.py {{args}}

# Initialize a new model from templates
#
# Python version knobs:
# - python_venv: concrete interpreter for local venv creation (default: .python-version)
# - python_requires: PEP 440 specifier for supported versions (default: ==<python_venv>.*)
init name python_venv=default_python python_requires="" torch=">=2.0.0":
    .venv/bin/python builder/init_model.py "{{name}}" --python-venv {{python_venv}} --python-requires "{{python_requires}}" --hf-repo "{{hf_repo}}" --torch "{{torch}}"

# Initialize a new model (dry run)
init-dry name python_venv=default_python python_requires="" torch=">=2.0.0":
    .venv/bin/python builder/init_model.py "{{name}}" --python-venv {{python_venv}} --python-requires "{{python_requires}}" --hf-repo "{{hf_repo}}" --torch "{{torch}}" --dry-run

# Fetch external source code for a model
fetch model:
    .venv/bin/python builder/vendor.py "models/{{model}}"

# Set up a model's development environment (fetch vendor, create venv, install deps)
# Usage: just setup <model> [cpu|gpu] [torch_version] [python]
#   - python: optional concrete interpreter version for venv (e.g. "3.11")
setup model device="cpu" torch_version="" python="":
    #!/usr/bin/env bash
    set -euo pipefail
    cd models/{{model}}

    # Pick Python interpreter for venv
    # Prefer CLI arg > dependencies.python_venv > default (.python-version)
    PYTHON_VERSION="{{python}}"
    if [ -z "$PYTHON_VERSION" ]; then
        PYTHON_VERSION=$(
            grep -A5 "^dependencies:" config.yaml 2>/dev/null \
              | grep -E '^[[:space:]]*python_venv:' \
              | sed -E 's/.*python_venv:[[:space:]]*"?([^"]*)"?/\1/' \
              || true
        )
    fi
    PYTHON_VERSION=${PYTHON_VERSION:-"{{default_python}}"}
    echo "Setting up {{model}} with Python ${PYTHON_VERSION} ({{device}})..."

    # Fetch vendor code when config has code.url and not package.pypi (vendor.py no-ops when PyPI-only or no code)
    ../../.venv/bin/python ../../builder/vendor.py .

    # Create venv with specified Python version (--clear to replace existing)
    echo "Creating venv with Python ${PYTHON_VERSION}..."
    uv venv --python ${PYTHON_VERSION} --clear

    # Activate the venv for subsequent commands
    source .venv/bin/activate

    # Get torch version: CLI arg > config.yaml > default
    TORCH_VERSION="{{torch_version}}"
    if [ -z "$TORCH_VERSION" ]; then
        # Try to read from config.yaml
        TORCH_VERSION=$(grep -A1 "dependencies:" config.yaml | grep "torch:" | sed 's/.*torch: *"\([^"]*\)".*/\1/' || echo ">=2.0.0")
        TORCH_VERSION=${TORCH_VERSION:-">=2.0.0"}
    fi

    # Set PyTorch index URL based on device
    if [ "{{device}}" = "cpu" ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        echo "Installing PyTorch${TORCH_VERSION} + torchaudio (CPU-only)..."
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        echo "Installing PyTorch${TORCH_VERSION} + torchaudio (CUDA 12.1)..."
    fi

    # Install PyTorch and torchaudio first from the appropriate index
    uv pip install "torch${TORCH_VERSION}" "torchaudio" --index-url ${TORCH_INDEX}

    # Install ttsdb-core and model dependencies
    echo "Installing dependencies..."
    uv pip install -e ../../core/
    uv pip install -e .

    echo "✓ Setup complete! Activate with: source models/{{model}}/.venv/bin/activate"

# Run unit tests for a model
test model:
    cd models/{{model}} && source .venv/bin/activate && uv pip install -e ../../core/ && uv pip install -e ".[dev]" && pytest tests/ -v -rs

# Run integration tests for a model (requires weights: just hf-weights-prepare <model>)
# Writes models/<model>/integration_result.json on success/failure.
test-integration model:
    cd models/{{model}} && source .venv/bin/activate && uv pip install -e ../../core/ && uv pip install -e ".[dev]" && pytest tests/ -v -rs -m integration

# Run integration tests for all models, then aggregate status and update README badges.
test-integration-all:
    #!/usr/bin/env bash
    set -euo pipefail
    for m in $(ls models); do
        just test-integration "$m" || true
    done
    just status-integration --readme

# Aggregate models/*/integration_result.json into status/ and optionally update README badges.
# Usage: just status-integration  (aggregate only) or  just status-integration --readme
status-integration *args:
    .venv/bin/python builder/aggregate_integration_status.py {{args}}

# Build a specific model
build model:
    @echo "Building {{model}}..."
    cd models/{{model}} && uv build
    docker build -t ttsdb-{{model}} models/{{model}}

# Build all models (using a tiny bit of bash)
build-all:
    for m in {{models}}; do \
        just build $m; \
    done

# HuggingFace operations are split into weights and spaces, mirroring the PyPI pattern.
#
# Weights: prepare/readme/upload/publish
hf-weights-prepare model *args:
    .venv/bin/python builder/huggingface.py prepare "models/{{model}}" {{args}}

hf-weights-readme model *args:
    .venv/bin/python builder/huggingface.py readme "models/{{model}}" {{args}}

hf-weights-upload model *args:
    .venv/bin/python builder/huggingface.py upload "models/{{model}}" {{args}}

hf-weights-publish model repo=hf_repo force="false" dry_run="false":
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{force}}" = "true" ]; then
        just hf-weights-prepare "{{model}}" --force
    else
        just hf-weights-prepare "{{model}}"
    fi
    just hf-weights-readme "{{model}}"
    if [ "{{dry_run}}" = "true" ]; then
        just hf-weights-upload "{{model}}" --repo "{{repo}}" --dry-run
    else
        just hf-weights-upload "{{model}}" --repo "{{repo}}"
    fi

hf-weights-publish-all repo=hf_repo force="false" dry_run="false":
    for m in {{models}}; do \
        just hf-weights-publish $m --repo {{repo}} --force={{force}} --dry-run={{dry_run}}; \
    done

# Spaces: generate/upload/publish
hf-space-generate model *args:
    .venv/bin/python builder/huggingface.py space "models/{{model}}" {{args}}

hf-space-upload model *args:
    .venv/bin/python builder/huggingface.py space-upload "models/{{model}}" {{args}}

hf-space-publish model repo=hf_repo *args:
    just hf-space-generate {{model}} {{args}}
    just hf-space-upload {{model}} --repo {{repo}} {{args}}

hf-space-publish-all repo=hf_repo *args:
    for m in {{models}}; do \
        just hf-space-publish $m --repo {{repo}} {{args}}; \
    done

# Run a model's space locally for testing
hf-space-run model:
    #!/usr/bin/env bash
    set -euo pipefail
    cd models/{{model}}

    # Generate space files if they do not exist
    if [ ! -f space/app.py ]; then
        echo "Generating space files..."
        ../../.venv/bin/python ../../builder/huggingface.py space .
    fi

    # Activate the venv
    source .venv/bin/activate

    # Install gradio if not present
    uv pip install gradio

    # Run the space
    echo "Starting {{model}} space at http://localhost:7860..."
    cd space && python app.py

# Build a model package for PyPI
pypi-build model:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building {{model}} package..."
    cd "models/{{model}}"
    # Ensure local version does not clash with PyPI
    ../../.venv/bin/python ../../builder/bump_pypi_version.py .
    # Ensure ttsdb-core dependency is pinned to the latest local version
    ../../.venv/bin/python ../../builder/sync_core_requirement.py .
    rm -rf dist/
    # Vendor upstream code when config has code.url and not package.pypi (vendor.py no-ops when PyPI-only or no code)
    ../../.venv/bin/python ../../builder/vendor.py .
    # Build wheel only; building sdists can omit vendored code depending on source filtering.
    uv build --wheel --no-sources --verbose

# Upload a model package to PyPI (requires UV_PUBLISH_TOKEN or --token)
# Usage: just pypi-upload <model> [--token <token>]
pypi-upload model *args:
    @echo "Uploading {{model}} to PyPI..."
    cd models/{{model}} && uv publish {{args}}

# Upload a model package to TestPyPI for testing
pypi-upload-test model *args:
    @echo "Uploading {{model}} to TestPyPI..."
    cd models/{{model}} && uv publish --publish-url https://test.pypi.org/legacy/ {{args}}

# Build and upload a model to PyPI in one step
pypi-publish model *args:
    just pypi-build {{model}}
    just pypi-upload {{model}} {{args}}

# Build and upload all models to PyPI
pypi-publish-all *args:
    for m in {{models}}; do \
        just pypi-publish $m {{args}}; \
    done

# ---- Core package (ttsdb-core) PyPI publishing ----

# Build the core package for PyPI
pypi-build-core:
    @echo "Building ttsdb-core package..."
    cd core && ../../.venv/bin/python ../builder/bump_pypi_version.py . && rm -rf dist/ && uv build --no-sources --verbose

# Upload the core package to PyPI (requires UV_PUBLISH_TOKEN or --token)
pypi-upload-core *args:
    @echo "Uploading ttsdb-core to PyPI..."
    cd core && uv publish {{args}}

# Upload the core package to TestPyPI
pypi-upload-core-test *args:
    @echo "Uploading ttsdb-core to TestPyPI..."
    cd core && uv publish --publish-url https://test.pypi.org/legacy/ {{args}}

# Build and upload the core package in one step
pypi-publish-core *args:
    just pypi-build-core
    just pypi-upload-core {{args}}
