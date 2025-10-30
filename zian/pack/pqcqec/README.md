All commands should be executed under pack folder

## 1) Train the transformer from scratch

Trains the AnglePredictor transformer on a fast synthetic 1‑qubit, length‑N setup.

```bash
python -m pqcqec.opt_transformer_new `
  --synthetic-1q5 `
  --synthetic-base-len 10 `
  --synthetic-enum-all `
  --epochs 500 `
  --batch-size 64 `
  --synthetic-train-frac 0.5 `   
  --synthetic-split-seed 42
```

Output: a timestamped checkpoint in `pack/models/`.

## 2) Load a checkpoint, predict, and further fine-tune

Evaluates a trained checkpoint on all 1‑qubit base circuits of length N with K random initial states and default noise (x/z = π/10, delta = 0).

```bash
python -m pqcqec.load_start_checkpoint `
  --ckpt [model file path] `
  --eval-synthetic `
  --synthetic-base-len 10 `
  --k-random 100
```

The script prints average fidelity and saves an N+3 JSONL with predicted angles.

## 3) Per-circuit angle optimization on N+3 JSONL

Reads an N+3 JSONL and optimizes the appended PQC block angles per circuit with gradient descent under default noise (x/z = π/10, delta = 0). It prints pre/post optimization average fidelity and saves the updated dataset to `--save-out` or a timestamped `angle_refine_[YYYYMMDD_HHMMSS].jsonl` next to the input if not specified.

```bash
python -m pqcqec.optimize_refined_circuits `
  --input [data file path from the 2nd step] `
  --batch-size 256 `
  --k-random 100 `
  --epochs 500 `
  --lr 0.0001
```

Outputs:
- Optimized JSONL: `angle_refine_[timestamp].jsonl` (or the path from `--save-out`)
- Printed average fidelity before and after optimization

## 4) Supervised fine-tuning on refined angles (tsf)

Fine-tunes the transformer to match ground-truth PQC angles from an N+3 JSONL like `angle_refine_*.jsonl`. The model predicts one final PQC block (rz, rx, rz) appended to the base circuit, trained with angular L2 loss (wrapped to (-π, π]). After training, it saves a final checkpoint and runs a synthetic enum‑all fidelity evaluation using default noise (π/10, delta=0) with K=100.

```bash
python -m pqcqec.tsf `
  --input [data file path from the 3rd step] `
  --ckpt [path of the transformer] `
  --epochs 500 `
  --batch-size 256 `
  --lr 1e-5 `
  --val-frac 0.1
```

Outputs:
- Final model: `models/tdf_[YYYYMMDD_HHMMSS].pt`
- Printed synthetic enum‑all average fidelity (N=base_len, K=100, default noise)

## Notes

- Default noise is enabled and set to x/z = π/10 radians, delta = 0 unless explicitly changed.
