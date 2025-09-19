# Memory Issue Explanation

In the old implementation (`old_training.py`), the memory explosion problem comes from the function **simulate_interleaved_with_params**.  
This function is called when calculating the fidelity loss.

---

## Problem in `old_training.py`

At line 401, the code constructs a full unitary matrix:

```python
Ufull = build_full_single(U1, int(q), n_qubits, device)
```

- `Ufull` has a size of 2^(n_qubits) × 2^(n_qubits).  
- We then use `Ufull` to calculate intermediate states (`st`), which are eventually collected into `new_states`.  
- These states flow into `state`, which is then used to compute the fidelity and finally the fidelity loss.

Because PyTorch needs to **track gradients**, it keeps all intermediate matrices in memory, including these large `Ufull` matrices for *every PQC gate* and base gates. 
As the number of gates grows, this leads to **memory explosion**.

---

## Solution in `optimized_training.py`

In the optimized version, we avoid constructing the massive `Ufull`.  
Instead, we **decompose** the operations into small unitary matrices:

- 2 × 2 for single-qubit gates  
- 4 × 4 for two-qubit gates  

By using these smaller matrices directly in the computation, we significantly reduce the memory footprint while keeping the same functionality.  

This optimization ensures that training remains efficient and feasible for larger circuits.
