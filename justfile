models := `ls models`

# Initialize a new model from templates
init name python="3.10" torch=">=2.0.0":
    python builder/init_model.py "{{name}}" --python {{python}} --torch "{{torch}}"

# Initialize a new model (dry run)
init-dry name python="3.10" torch=">=2.0.0":
    python builder/init_model.py "{{name}}" --python {{python}} --torch "{{torch}}" --dry-run

# Fetch external source code for a model
fetch model:
    python builder/vendor.py "models/{{model}}"

# Set up a model's development environment (fetch vendor, create venv, install deps)
# Usage: just setup <model> [cpu|gpu] [torch_version]  (default: cpu, version from config.yaml)
setup model device="cpu" torch_version="":
    #!/usr/bin/env bash
    set -euo pipefail
    cd models/{{model}}
    
    # Read Python version from config.yaml
    PYTHON_VERSION=$(grep -A2 "dependencies:" config.yaml | grep "python:" | sed 's/.*python: *"\([^"]*\)".*/\1/' || echo "3.10")
    PYTHON_VERSION=${PYTHON_VERSION:-"3.10"}
    echo "Setting up {{model}} with Python ${PYTHON_VERSION} ({{device}})..."
    
    # Fetch vendor code if config has code.url
    if grep -q "code:" config.yaml && grep -q "url:" config.yaml 2>/dev/null; then
        echo "Fetching vendor code..."
        python ../../builder/vendor.py .
    fi
    
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
        echo "Installing PyTorch${TORCH_VERSION} (CPU-only)..."
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        echo "Installing PyTorch${TORCH_VERSION} (CUDA 12.1)..."
    fi
    
    # Install PyTorch first from the appropriate index
    uv pip install "torch${TORCH_VERSION}" --index-url ${TORCH_INDEX}
    
    # Install ttsdb-core and model dependencies
    echo "Installing dependencies..."
    uv pip install -e ../../core/
    uv pip install -e .
    
    echo "✓ Setup complete! Activate with: source models/{{model}}/.venv/bin/activate"

# Run tests for a model
test model:
    cd models/{{model}} && source .venv/bin/activate && uv pip install -e ".[dev]" && pytest tests/ -v

# Build a specific model
build model:
    @echo "Building {{model}}..."
    python builder/generate_files.py --model {{model}}
    cd models/{{model}} && uv build
    docker build -t ttsdb-{{model}} models/{{model}}

# Build all models (using a tiny bit of bash)
build-all:
    for m in {{models}}; do \
        just build $m; \
    done

# Generate HuggingFace README for a model
hf-readme model:
    python builder/huggingface.py readme "models/{{model}}"

# Upload weights to HuggingFace (ttsds/models by default)
hf-upload model repo="ttsds/models":
    python builder/huggingface.py upload "models/{{model}}" --repo "{{repo}}"

# Preview HuggingFace upload (dry run)
hf-upload-dry model repo="ttsds/models":
    python builder/huggingface.py upload "models/{{model}}" --repo "{{repo}}" --dry-run

# Generate HuggingFace README for all models
hf-readme-all:
    for m in {{models}}; do \
        just hf-readme $m; \
    done

# Upload all models to HuggingFace
hf-upload-all repo="ttsds/models":
    for m in {{models}}; do \
        just hf-upload $m {{repo}}; \
    done