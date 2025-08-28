#!/usr/bin/env bash

# Run train_tokenize_circuits_mp.py over a seed range derived from counters.
# Usage:
#   scripts/run_train_tokenize_seed_range.sh <config_path> <counter1> [counter2]
# Behavior:
#   - With two args: seeds from 10000 to (counter1*10000) inclusive.
#   - With three args: seeds from (counter1*10000) to (counter2*10000) inclusive.

set -uo pipefail

usage() {
  echo "Usage: $0 <config_path> <counter1> [counter2]" >&2
  echo "  Two args: seeds 10000 .. counter1*10000" >&2
  echo "  Three args: seeds counter1*10000 .. counter2*10000" >&2
  exit 1
}

if (( $# < 2 || $# > 3 )); then
  usage
fi

CONFIG_PATH="$1"
COUNTER1="$2"
COUNTER2="${3-}"

# Resolve script directory to locate the Python module reliably
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/train_tokenize_circuits_mp.py"

# Basic validations
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: config file not found: $CONFIG_PATH" >&2
  exit 2
fi

if ! [[ "$COUNTER1" =~ ^[0-9]+$ ]]; then
  echo "Error: counter1 must be a non-negative integer: $COUNTER1" >&2
  exit 3
fi

if [[ -n "$COUNTER2" ]] && ! [[ "$COUNTER2" =~ ^[0-9]+$ ]]; then
  echo "Error: counter2 must be a non-negative integer: $COUNTER2" >&2
  exit 4
fi

SCALE=10000

if [[ -n "$COUNTER2" ]]; then
  START=$(( COUNTER1 * SCALE ))
  END=$(( COUNTER2 * SCALE ))
else
  START=$SCALE
  END=$(( COUNTER1 * SCALE ))
fi

if (( START > END )); then
  echo "Error: start ($START) is greater than end ($END)." >&2
  exit 5
fi

echo "Config: $CONFIG_PATH"
echo "Seed range: $START .. $END (inclusive)"
echo

FAILS=0
TOTAL=0

for (( seed = START; seed <= END; seed++ )); do
  ((TOTAL++))
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running seed $seed ..."
  # --force=no ensures only this seed is processed in the MP script
  if ! python -u "$PY_SCRIPT" --config "$CONFIG_PATH" --seed "$seed" --force=no; then
    echo "  -> Failed for seed $seed" >&2
    ((FAILS++))
  fi
done

echo
echo "Completed: $TOTAL runs; Failures: $FAILS"
exit $(( FAILS > 0 ? 1 : 0 ))

