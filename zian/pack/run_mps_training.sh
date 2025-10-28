#!/bin/bash
# Helper script to run MPS training with proper environment setup

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if .venv exists
if [ -d "../../.venv" ]; then
    echo "[MPS] Activating virtual environment..."
    source ../../.venv/bin/activate
elif [ -d ".venv" ]; then
    echo "[MPS] Activating virtual environment..."
    source .venv/bin/activate
else
    echo "[MPS] Warning: No virtual environment found. Using system Python."
fi

# Set MPS fallback (in case it's not auto-set in the script)
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "[MPS] Starting training..."
echo ""

# Run the training with all arguments passed to this script
python -m pqcqec.train_lelzz_mps "$@"

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "[MPS] Training completed successfully!"
else
    echo "[MPS] Training failed with exit code: $exit_code"
fi

exit $exit_code
