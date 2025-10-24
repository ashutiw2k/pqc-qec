# Running ZZ-Ring PQC Training

## Quick Start

```bash
# 1. Navigate to zian/pack directory
cd /Users/ashutoshtiwari/Desktop/Research-Code/pqc-qec/zian/pack

# 2. Activate virtual environment (if not already)
source ../../.venv/bin/activate

# 3. Set PYTHONPATH and run
PYTHONPATH="${PWD}:${PYTHONPATH}" python -m pqcqec.train_lelzz \
    --data-path ../../data/json_data/3q_10g_5blk_data/good_fidelity \
    --n-qubits 3 \
    --gate-blocks 5 \
    --epochs 100 \
    --batch-size 32
```

## Architecture

The ZZ-ring PQC architecture per block:
- **Pre-local**: RZ-RX-RZ on each qubit (3×Q angles)
- **ZZ-ring**: CNOT-RZ-CNOT between adjacent pairs in a ring (Q angles)  
- **Post-local**: RZ-RX-RZ on each qubit (3×Q angles)

**Total**: 7×Q angles per block for Q qubits

## Command Options

```bash
--data-path         # Path to dataset directory (required)
                    # Use: .../good_fidelity or .../poor_fidelity subdirectories

--n-qubits 3        # Number of qubits (must match data, default: 2)

--epochs 100        # Training epochs (default: 100)

--batch-size 32     # Batch size (default: 32)

--lr 5e-4           # Learning rate (default: 5e-4)

--k-random 32       # Number of random initial states (default: 32)

--gate-blocks 5     # Base gates per PQC block (default: 5)

--num-sample 1000   # Limit dataset size for testing (optional)

--no-detach-base-noise  # Don't detach base circuit gradients
```

## Examples

### Quick Test Run (small dataset)
```bash
cd /Users/ashutoshtiwari/Desktop/Research-Code/pqc-qec/zian/pack
PYTHONPATH="${PWD}:${PYTHONPATH}" python -m pqcqec.train_lelzz \
    --data-path ../../data/json_data/3q_10g_5blk_data/good_fidelity \
    --n-qubits 3 \
    --gate-blocks 5 \
    --epochs 2 \
    --batch-size 4 \
    --num-sample 100
```

### Full Training Run (3 qubits)
```bash
cd /Users/ashutoshtiwari/Desktop/Research-Code/pqc-qec/zian/pack
PYTHONPATH="${PWD}:${PYTHONPATH}" python -m pqcqec.train_lelzz \
    --data-path ../../data/json_data/3q_10g_5blk_data/good_fidelity \
    --n-qubits 3 \
    --gate-blocks 5 \
    --epochs 100 \
    --batch-size 32 \
    --lr 5e-4
```

### Training on 5 qubits
```bash
cd /Users/ashutoshtiwari/Desktop/Research-Code/pqc-qec/zian/pack
PYTHONPATH="${PWD}:${PYTHONPATH}" python -m pqcqec.train_lelzz \
    --data-path ../../data/json_data/5q_20g_10blk_data/good_fidelity \
    --n-qubits 5 \
    --gate-blocks 10 \
    --epochs 100 \
    --batch-size 16
```

## Output

Training will print:
```
[LELZZ] Training ZZ-ring PQC: n_qubits=3, gate_blocks=5
[LELZZ] Angles per block: 7*3 = 21
[LELZZ] Filtered dataset: 4270 circuits with 3 qubits
[LELZZ] Train: 3843, Val: 427
[LELZZ] Epoch 1/100 | LR=0.000100 | Train Loss=0.608359 | Val Fid=0.453729
[LELZZ] Epoch 2/100 | LR=0.000150 | Train Loss=0.575235 | Val Fid=0.471918
...
```

## Files

- `train_lelzz.py` - Main training script with ZZRingAnglePredictor model
- `simulator_lelzz.py` - ZZ-ring PQC block simulator
- `simulator_core.py` - Core quantum simulation primitives
- `dataset.py` - Dataset constants
- `precision.py` - AMP/mixed precision settings

## Troubleshooting

**Error: No module named pqcqec**
- Make sure you're in `/path/to/pqc-qec/zian/pack`
- Set PYTHONPATH: `export PYTHONPATH="${PWD}:${PYTHONPATH}"`

**Error: No circuits with n_qubits=X found**
- Check that `--n-qubits` matches your dataset
- Verify you're pointing to the correct subdirectory (good_fidelity or poor_fidelity)

**Error: base_gates not found**
- Make sure you're pointing to a subdirectory with .json files
- Path should be: `.../3q_10g_5blk_data/good_fidelity` (not just `.../3q_10g_5blk_data`)
