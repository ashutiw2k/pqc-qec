# Fine-tune Transformer Predictions with Gate Sequence Noise

## Overview

The `finetune_transformer_predictions_mp.py` script has been updated to support the new gate sequence noise model. You can now choose between:

1. **Traditional rotation noise** (RxRz gates)
2. **Gate sequence noise** (HH→HX transformations)
3. **Both** (gate sequence + rotation)

## New Command-Line Arguments

### `--noise-type {rotation,gate_sequence,both}`
Choose the noise model type (default: `rotation`)

- `rotation`: Traditional RxRz noise gates after each base gate
- `gate_sequence`: Coherent gate transformations (HH→HX, XX→XZ, ZZ→ZH)
- `both`: Apply gate sequence transformations AND rotation noise

### `--gate-noise-prob PROB`
Probability [0-1] for applying gate sequence transformations (default: `1.0`)

- `1.0`: Deterministic - all matching pairs transformed
- `<1.0`: Probabilistic - random subset transformed

### `--gate-noise-rules JSON`
Custom transformation rules as JSON string (default: None - uses HH→HX, XX→XZ, ZZ→ZH)

Format: `'{"HH": "HX", "XX": "YZ", "ZZ": "ZH"}'`

## Usage Examples

### Example 1: Traditional Rotation Noise (Default)

```bash
.venv/bin/python scripts/finetune_transformer_predictions_mp.py \
  -i nogit/transformer_predictions.jsonl \
  -o nogit/finetune_results \
  -q 1 -g 10 -k 10 \
  -n 1000 -t 100 -b 10 -e 5 \
  --noise-type rotation
```

### Example 2: Pure Gate Sequence Noise

```bash
.venv/bin/python scripts/finetune_transformer_predictions_mp.py \
  -i nogit/transformer_predictions.jsonl \
  -o nogit/finetune_results \
  -q 1 -g 10 -k 10 \
  -n 1000 -t 100 -b 10 -e 5 \
  --noise-type gate_sequence
```

This will apply coherent gate errors (HH→HX, XX→XZ, ZZ→ZH) without adding rotation noise.

### Example 3: Both Noise Types

```bash
.venv/bin/python scripts/finetune_transformer_predictions_mp.py \
  -i nogit/transformer_predictions.jsonl \
  -o nogit/finetune_results \
  -q 1 -g 10 -k 10 \
  -n 1000 -t 100 -b 10 -e 5 \
  --noise-type both
```

This applies gate sequence transformations THEN adds rotation noise.

### Example 4: Probabilistic Gate Sequence Noise

```bash
.venv/bin/python scripts/finetune_transformer_predictions_mp.py \
  -i nogit/transformer_predictions.jsonl \
  -o nogit/finetune_results \
  -q 1 -g 10 -k 10 \
  -n 1000 -t 100 -b 10 -e 5 \
  --noise-type gate_sequence \
  --gate-noise-prob 0.3
```

Only 30% of matching gate pairs will be transformed (models intermittent errors).

### Example 5: Custom Transformation Rules

```bash
.venv/bin/python scripts/finetune_transformer_predictions_mp.py \
  -i nogit/transformer_predictions.jsonl \
  -o nogit/finetune_results \
  -q 1 -g 10 -k 10 \
  -n 1000 -t 100 -b 10 -e 5 \
  --noise-type gate_sequence \
  --gate-noise-rules '{"HH": "HS", "XX": "XY", "HX": "HZ"}'
```

Uses custom rules: HH→HS, XX→XY, HX→HZ instead of defaults.

## Output Changes

### Results JSONL File

Each result now includes:
```json
{
  "circuit_idx": 0,
  "test_fidelity_pqc_mean": 0.9876,
  ...,
  "noise_type": "gate_sequence",
  "gate_sequence_noise_prob": 1.0
}
```

### Console Output

The script now displays noise configuration:
```
============================================================
Fine-tune Transformer PQC Predictions
============================================================
Input file: nogit/transformer_predictions.jsonl
Output directory: nogit/finetune_results
Circuit: 1q, 10g, 10 blocks
Training: 1000 samples, 10 batch, 5 epochs
Test: 100 samples
Noise model: gate_sequence
  Gate sequence noise probability: 1.0
  Default rules: HH→HX, XX→XZ, ZZ→ZH
Multiprocessing: 4 processes
============================================================
```

## Comparing Noise Models

To compare the effectiveness of different noise models:

```bash
# Run with rotation noise
.venv/bin/python scripts/finetune_transformer_predictions_mp.py \
  -i input.jsonl -o results/rotation \
  --noise-type rotation -q 1 -g 10

# Run with gate sequence noise
.venv/bin/python scripts/finetune_transformer_predictions_mp.py \
  -i input.jsonl -o results/gate_sequence \
  --noise-type gate_sequence -q 1 -g 10

# Run with both
.venv/bin/python scripts/finetune_transformer_predictions_mp.py \
  -i input.jsonl -o results/both \
  --noise-type both -q 1 -g 10

# Compare results
python -c "
import json
for noise_type in ['rotation', 'gate_sequence', 'both']:
    with open(f'results/{noise_type}/finetuned_1q_10g_summary.json') as f:
        data = json.load(f)
        print(f'{noise_type:15s}: {data[\"finetuned_fidelity_mean\"]:.4f}')
"
```

## Expected Behavior

### Circuit Size Impact

- **`rotation`**: Circuit grows ~3x (adds 2 noise gates per base gate)
- **`gate_sequence`**: Circuit size unchanged (modifies gates in-place)
- **`both`**: Circuit grows ~3x (modified gates + rotation noise)

### Training Differences

- **Rotation noise**: Models incoherent decoherence errors
- **Gate sequence noise**: Models coherent systematic calibration errors
- **Both**: Comprehensive error model

PQC may learn gate sequence errors more effectively since they are coherent and deterministic.

## Implementation Details

The script now:

1. Parses `--noise-type`, `--gate-noise-prob`, and `--gate-noise-rules` arguments
2. Passes noise configuration to `hyperparams` dict
3. Initializes `ZXZInterleavedAngleCustomStatevecModel` with:
   - `noise_type=...`
   - `gate_sequence_noise_rules=...`
   - `gate_sequence_noise_prob=...`
   - `noise_seed=circuit_idx`
4. Model applies gate sequence transformations before building templates
5. Results include noise configuration for tracking

## Backward Compatibility

✅ **Fully backward compatible** - existing commands work unchanged (default `--noise-type rotation`)

## Files Modified

- `scripts/finetune_transformer_predictions_mp.py` - Added noise type support

## See Also

- `GATE_SEQUENCE_NOISE.md` - Theory and implementation
- `PQCMODELBASE_NOISE_INTEGRATION.md` - Model integration details
- `example_gate_sequence_noise_integration.py` - Examples
