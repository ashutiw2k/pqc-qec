#!/bin/bash
# Linux equivalent of vsdev_python.cmd
# Conda + CUDA wrapper for Python development

# 1. Activate conda environment if available
CONDA_BASE="/home/ubuntu/zian/ENTER"
if [ -d "$CONDA_BASE" ]; then
    # 使用 conda init 的方式来确保正确激活
    source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || true
    # 然后激活特定环境（如果存在）
    if [ -d "$CONDA_BASE/envs/pc2" ]; then
        conda activate pc2 2>/dev/null || source "$CONDA_BASE/bin/activate" pc2 2>/dev/null || true
        echo "[INFO] Activated conda environment pc2"
    else
        source "$CONDA_BASE/bin/activate" 2>/dev/null || true
        echo "[INFO] Activated conda base environment at $CONDA_BASE"
    fi
else
    echo "[WARN] Conda environment not found at $CONDA_BASE"
fi

# 2. Set CUDA environment if available
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
    export CUDA_PATH="$CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    echo "[INFO] CUDA_HOME set to $CUDA_HOME"
elif [ -d "/opt/cuda" ]; then
    export CUDA_HOME="/opt/cuda"
    export CUDA_PATH="$CUDA_HOME"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    echo "[INFO] CUDA_HOME set to $CUDA_HOME"
else
    echo "[WARN] CUDA installation not found"
fi

# 2.5. Set CUDA optimizations for GH200
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[INFO] PyTorch CUDA memory optimizations enabled for GH200"

# 3. Set TORCH_CUDA_ARCH_LIST if not already set
if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
    export TORCH_CUDA_ARCH_LIST="9.0"  # GH200 architecture
    echo "[INFO] TORCH_CUDA_ARCH_LIST set to $TORCH_CUDA_ARCH_LIST (GH200)"
fi

# 4. Print diagnostics (first run only)
if [ -z "$_VSDEV_WRAPPER_SHOWN" ]; then
    echo "[INFO] vsdev_python.sh initialized. Using:"
    if command -v nvcc >/dev/null 2>&1; then
        echo "    nvcc: $(which nvcc)"
    else
        echo "    nvcc: NOT FOUND"
    fi
    if command -v python >/dev/null 2>&1; then
        echo "    python: $(which python)"
    else
        echo "    python: NOT FOUND"
    fi
    echo "    TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
    if [ -n "$CUDA_HOME" ]; then
        echo "    CUDA_HOME=$CUDA_HOME"
    fi
    export _VSDEV_WRAPPER_SHOWN=1
fi

# 5. Delegate to python with all original args
exec python "$@"