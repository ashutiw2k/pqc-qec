# ZZ-Ring PQC Training: Complete Step-by-Step Breakdown

## Table of Contents
1. [Model Initialization](#model-initialization)
2. [Training Preparation](#training-preparation)
3. [Training Loop Iteration](#training-loop-iteration)
4. [Helper Functions](#helper-functions)

---

## Model Initialization

### Step 1: Create `ZZRingAnglePredictor` Model

```python
model = ZZRingAnglePredictor(gate_blocks=5, n_qubits=3).to(device)
```

#### Constructor Execution (`__init__`)

**Input Parameters:**
- `gate_blocks`: 5 (number of base circuit gates per PQC block)
- `n_qubits`: 3 (number of qubits in circuits)

**Computation Steps:**

1. **Calculate `angles_per_block`**:
   ```python
   self.angles_per_block = 7 * n_qubits = 7 * 3 = 21
   ```
   - 3 angles (RZ-RX-RZ) × 3 qubits = 9 pre-local angles
   - 1 angle × 3 pairs = 3 ZZ-ring angles
   - 3 angles × 3 qubits = 9 post-local angles
   - **Total: 21 angles per block**

2. **Calculate `max_blocks`**:
   ```python
   self.max_blocks = ceil(MAX_BASE_LEN / gate_blocks) = ceil(1000 / 5) = 200
   ```
   - Maximum possible blocks in any circuit

3. **Calculate input feature dimension**:
   ```python
   feat_dim = 3 + angles_per_block * PREV_K
            = 3 + 21 * 1 = 24
   ```
   - 3 statistics: [gate_count, cumulative_count, block_index]
   - 21 angles from PREV_K=1 previous blocks

**Architecture Components Created:**

1. **Input Projection (`self.in_proj`)**:
   ```
   Sequential(
       Linear(24 → 768),     # Project features to hidden dimension
       GELU(),               # Activation
       Dropout(0.1),         # Regularization
       LayerNorm(768)        # Normalization
   )
   ```

2. **Positional Embeddings (`self.pos_emb`)**:
   ```
   Embedding(200, 768)  # Learnable position embeddings for 200 blocks
   ```

3. **Transformer Encoder (`self.encoder`)**:
   ```
   TransformerEncoder(
       num_layers=8,
       layer=TransformerEncoderLayer(
           d_model=768,
           nhead=12,
           dim_feedforward=3072,  # 768 * 4
           dropout=0.1,
           batch_first=True,
           norm_first=True  # Pre-norm architecture
       )
   )
   ```

4. **Output Head (`self.head_ln`, `self.head`)**:
   ```
   head_ln = LayerNorm(768)
   head = Linear(768 → 42)  # 2 values per angle (x,y on unit circle)
   ```
   - Outputs 42 values = 2 × 21 angles
   - Each angle represented as (x, y) on S¹ (unit circle)

**Head Initialization**:
```python
# Initialize to predict identity (all angles = 0)
head.weight = 0
head.bias = [1, 0, 1, 0, ..., 1, 0]  # 21 times (x=1, y=0 → angle=0)
```

**Total Parameters:** ~100M parameters in transformer

---

## Training Preparation

### Step 2: Load and Prepare Dataset

#### 2.1 Load Dataset (`CircuitDataset`)

```python
ds_full = CircuitDataset(data_path='...3q_10g_5blk_data/good_fidelity', num_sample=100)
```

**What happens inside `CircuitDataset.__init__`:**

**Input:** Directory path with JSON files

**Process:**
1. Iterate through all `.json` files in directory
2. For each file, read first line (one circuit per file)
3. Parse JSON with keys: `['seed', 'fidelity', 'pqc_params', 'base_circuit_tokens', 'pqc_circuit_tokens', ...]`

**Data Format:**
```json
{
  "base_circuit_tokens": [
    ["h", [0], []],        // Gate name, qubits, parameters
    ["cx", [0, 1], []],
    ["x", [2], []]
  ],
  "pqc_circuit_tokens": [...],
  "n_qubits": 3
}
```

**Output per circuit:**
```python
{
    'idx': 0,                           # Unique circuit index
    'base_gates': ['h', 'cx', 'x'],    # Gate names
    'base_q1': [0, 0, 2],              # Primary qubit
    'base_q2': [-1, 1, -1],            # Secondary qubit (-1 for 1q gates)
    'param_gates': [],                  # (not used in lelzz mode)
    'param_qubits': [],
    'after': [],
    'param_angles_gt': [],
    'n_qubits': 3                       # Number of qubits
}
```

**Result:** `ds_full.items` = list of 100 circuit dictionaries

#### 2.2 Filter by `n_qubits`

```python
filtered_items = [item for item in ds_full.items if item['n_qubits'] == 3]
ds = FilteredDataset(filtered_items)  # Wrapper class
```

**Output:** 100 circuits with exactly 3 qubits

#### 2.3 Build Caches (`build_base_cache_vectorized`)

```python
init_cache, ref_cache, noise_schedules = build_base_cache_vectorized(
    ds, k_random=32, device=device, noise=None
)
```

**Purpose:** Pre-compute initial states and reference states for all circuits

**Detailed Execution:**

**Step 2.3.1: Group circuits by `n_qubits`**
```python
groups = {3: [circuit1, circuit2, ..., circuit100]}
```

**Step 2.3.2: Create random initial states (for n=3)**
```python
K = 32  # k_random
dim = 2^3 = 8  # Hilbert space dimension
states_init = []  # Will contain 32 random states

for k in range(32):
    st = [1, 0, 0, 0, 0, 0, 0, 0]  # |000⟩ state (complex)
    
    # Randomize each qubit independently
    for q in [0, 1, 2]:
        r = random()
        if r < 0.33:
            pass           # Keep |0⟩
        elif r < 0.66:
            apply_X(st, q)  # Flip to |1⟩
        else:
            apply_H(st, q)  # Superposition
    
    states_init.append(st)

init_cache[3] = torch.stack(states_init)  # Shape: [32, 8]
```

**Example states:**
- `|000⟩`: [1, 0, 0, 0, 0, 0, 0, 0]
- `|111⟩`: [0, 0, 0, 0, 0, 0, 0, 1]
- `H|0⟩ ⊗ |1⟩ ⊗ |0⟩`: [0, 1/√2, 0, 1/√2, 0, 0, 0, 0]

**Step 2.3.3: Simulate base circuits to get reference states**

For each circuit:
```python
# Start with 32 random initial states
states = init_cache[3].unsqueeze(0).expand(100, -1, -1).clone()
# Shape: [100 circuits, 32 initial states, 8 amplitudes] = [100, 32, 8]

# Apply base gates sequentially
for t in range(num_gates):
    gate = base_gates[t]    # e.g., 'cx'
    q1 = base_q1[t]         # e.g., 0
    q2 = base_q2[t]         # e.g., 1
    
    if gate == 'h':
        # Hadamard on qubit q1
        # States at indices [i where bit q1 is 0] and [i where bit q1 is 1]
        splits = _split_indices(q1)  # e.g., q1=0: ([0,2,4,6], [1,3,5,7])
        i0, i1 = splits
        states[:, :, i0] = (states[:, :, i0] + states[:, :, i1]) / sqrt(2)
        states[:, :, i1] = (states[:, :, i0] - states[:, :, i1]) / sqrt(2)
    
    elif gate == 'cx':
        # CNOT from q1 (control) to q2 (target)
        # Find indices where control is |1⟩ and target is |0⟩
        # Swap with indices where control is |1⟩ and target is |1⟩
        cx_swap = _get_cx_indices(q1, q2)  # e.g., swap(3,7), swap(1,5)
        i0, i1 = cx_swap
        temp = states[:, :, i0].clone()
        states[:, :, i0] = states[:, :, i1]
        states[:, :, i1] = temp
    
    # Similar for 'x', 'z', 'cz'

# Store reference states
ref_cache['tensor'] = states  # [100, 32, 8]
ref_cache['idx2row'] = {circuit_idx: row_number}  # Mapping
```

**Output:**
- `init_cache`: `{3: Tensor[32, 8]}` - Initial states
- `ref_cache`: `{'tensor': Tensor[100, 32, 8], 'idx2row': dict}` - Target states after base circuit
- `noise_schedules`: Empty (no noise in this run)

#### 2.4 Create Data Loaders

```python
train_loader = DataLoader(ds_train, batch_size=4, shuffle=True, collate_fn=collate)
```

**Collate Function (`collate`)**:

**Input:** List of 4 circuit dictionaries (batch_size=4)

**Process:**
```python
# Find max lengths in batch
max_base = max(len(c['base_gates']) for c in batch)  # e.g., 15
max_n_qubits = max(c['n_qubits'] for c in batch)     # e.g., 3

# Create padded tensors
base_g_batch = torch.full((4, max_base), PAD_ID)      # [4, 15]
base_q1_batch = torch.full((4, max_base), -1)         # [4, 15]
base_q2_batch = torch.full((4, max_base), -1)         # [4, 15]
base_len_batch = torch.tensor([len(c['base_gates']) for c in batch])  # [4]
n_qubits_batch = torch.tensor([c['n_qubits'] for c in batch])         # [4]
idx_batch = torch.tensor([c['idx'] for c in batch])                   # [4]

# Fill with actual data (pad shorter sequences)
for i, circuit in enumerate(batch):
    L = len(circuit['base_gates'])
    base_g_batch[i, :L] = convert_gates_to_ids(circuit['base_gates'])
    base_q1_batch[i, :L] = circuit['base_q1']
    base_q2_batch[i, :L] = circuit['base_q2']
```

**Output:** `Batch` object with fields:
```python
Batch(
    base_g=[4, 15],      # Gate IDs
    base_q1=[4, 15],     # Primary qubits
    base_q2=[4, 15],     # Secondary qubits
    base_len=[4],        # Actual lengths
    n_qubits=[4],        # Number of qubits per circuit
    idx=[4],             # Circuit indices
    # ... other fields not used in lelzz mode
)
```

---

## Training Loop Iteration

### Step 3: One Training Iteration

#### 3.1 Get Batch from Data Loader

```python
for batch in train_loader:
    batch = batch.to(device)  # Move to GPU/CPU
```

**Batch contents (example):**
```python
batch.base_g = [[H, CX, X, Z, H, ...],      # Circuit 1: 10 gates
                [H, H, CZ, X, CX, ...],      # Circuit 2: 12 gates
                [CX, H, X, CX, Z, ...],      # Circuit 3: 11 gates
                [X, Z, H, CX, CX, ...]]      # Circuit 4: 9 gates
                # Shape: [4, 15] (padded to max)

batch.base_len = [10, 12, 11, 9]
batch.n_qubits = [3, 3, 3, 3]
batch.idx = [17, 42, 8, 91]  # Original dataset indices
```

#### 3.2 Forward Pass: `model(batch, device)`

##### **3.2.1 Compute Block Statistics**

```python
B = 4  # Batch size
Lb_max = 12  # max(batch.base_len) = max(10, 12, 11, 9)
max_blocks = ceil(12 / 5) = 3  # Blocks needed

# Count gates per block
counts = torch.zeros(4, 3)  # [B, max_blocks]

# Example for circuit 0 (Lb=10 gates, gate_blocks=5):
# Block 0: gates [0:5] → count = 5
# Block 1: gates [5:10] → count = 5
# Block 2: gates [10:10] → count = 0
counts[0] = [5, 5, 0]

# Example for circuit 1 (Lb=12 gates):
# Block 0: gates [0:5] → count = 5
# Block 1: gates [5:10] → count = 5
# Block 2: gates [10:12] → count = 2
counts[1] = [5, 5, 2]

# Similarly for all circuits:
counts = [[5, 5, 0],   # Circuit 0: 10 gates
          [5, 5, 2],   # Circuit 1: 12 gates
          [5, 5, 1],   # Circuit 2: 11 gates
          [5, 4, 0]]   # Circuit 3: 9 gates

# Cumulative counts
cum = counts.cumsum(dim=1)
cum = [[5, 10, 10],
       [5, 10, 12],
       [5, 10, 11],
       [5, 9, 9]]
```

##### **3.2.2 Autoregressive Block-by-Block Prediction**

```python
# Initialize buffers
prev_buf = torch.zeros(4, 1, 21)  # [B, PREV_K=1, 21 angles]
Y = torch.zeros(4, 3, 21)         # [B, 3 blocks, 21 angles]

# Create causal attention mask (upper triangular)
attn_mask = [[False, True,  True],   # Block 0 can see only itself
             [False, False, True],   # Block 1 can see 0,1
             [False, False, False]]  # Block 2 can see 0,1,2
```

**Block-by-Block Loop:**

**Iteration t=0 (Predict Block 0):**

```python
L = 1  # Sequence length so far

# Build features for block 0
prev_seq[:, 0, :] = prev_buf.flatten()  # All zeros (no previous angles)

feats = torch.cat([
    counts[:, :1].unsqueeze(-1),    # [4, 1, 1]: [[5], [5], [5], [5]]
    cum[:, :1].unsqueeze(-1),       # [4, 1, 1]: [[5], [5], [5], [5]]
    [[0]], [[0]], [[0]], [[0]],     # [4, 1, 1]: Block index 0
    prev_seq[:, :1, :]              # [4, 1, 21]: All zeros
], dim=-1)  # Result: [4, 1, 24] features

# Project to hidden dimension
x = self.in_proj(feats)  # [4, 1, 24] → [4, 1, 768]

# Add positional embedding
x = x + self.pos_emb(torch.arange(1))  # Add pos_emb[0]

# Apply transformer (1 position, can only attend to itself)
h = self.encoder(x, mask=attn_mask[:1, :1])  # [4, 1, 768]

# Predict angles from last (only) position
h_last = self.head_ln(h[:, 0, :])      # [4, 768]
logits_t = self.head(h_last)           # [4, 42] (2 per angle)

# Convert S¹ representation to angles
logits_t = [[x0, y0, x1, y1, ..., x20, y20],  # 42 values per sample
            [...],
            [...],
            [...]]  # Shape: [4, 42]

xy = logits_t.view(4, 21, 2)  # [4, 21, 2]
x_vals = xy[:, :, 0]          # [4, 21]
y_vals = xy[:, :, 1]          # [4, 21]

# Normalize to unit circle
r = sqrt(x^2 + y^2)
x_norm = x / r
y_norm = y / r

# Convert to angle
theta = atan2(y_norm, x_norm)  # [4, 21] angles in [-π, π]

# Example output (first sample):
# theta[0] = [-0.1, 0.05, -0.02, 0.3, ..., 0.15]  # 21 angles

# Sanitize
theta = clamp(theta, -π+ε, π-ε)

# Store
Y[:, 0, :] = theta  # Save block 0 predictions

# Update prev buffer for next block
prev_buf[:, 0, :] = theta  # Store these angles as "previous"
```

**Iteration t=1 (Predict Block 1):**

```python
L = 2  # Now processing blocks 0 and 1

# prev_seq already has block 0's previous (zeros)
# Update block 1's previous to be block 0's output
prev_seq[:, 1, :] = prev_buf.flatten()  # Block 0's angles

feats = torch.cat([
    counts[:, :2].unsqueeze(-1),    # [4, 2, 1]: [[5,5], [5,5], [5,5], [5,4]]
    cum[:, :2].unsqueeze(-1),       # [4, 2, 1]: [[5,10], [5,10], [5,10], [5,9]]
    [[0,1], [0,1], [0,1], [0,1]],   # [4, 2, 1]: Block indices
    prev_seq[:, :2, :]              # [4, 2, 21]: Block 0 prev=zeros, Block 1 prev=block0_angles
], dim=-1)  # [4, 2, 24]

# Project
x = self.in_proj(feats)  # [4, 2, 768]

# Add positional embeddings
x = x + self.pos_emb([0, 1]).unsqueeze(0)  # Add pos_emb[0] and pos_emb[1]

# Apply transformer with causal mask
# Block 0 position can only see itself
# Block 1 position can see both block 0 and 1
h = self.encoder(x, mask=attn_mask[:2, :2])  # [4, 2, 768]

# Predict angles from LAST position (block 1)
h_last = self.head_ln(h[:, 1, :])    # [4, 768]
logits_t = self.head(h_last)         # [4, 42]

# Convert to angles (same as before)
theta = _angles_from_s1(logits_t)    # [4, 21]

# Store
Y[:, 1, :] = theta

# Update prev buffer (sliding window of size 1)
prev_buf = roll(prev_buf, shifts=-1, dims=1)  # Shift out oldest
prev_buf[:, 0, :] = theta  # Block 1 angles become "previous"
```

**Iteration t=2 (Predict Block 2):**

Similar process, but now:
- `feats` includes stats for blocks 0, 1, 2
- `prev_seq[:, 2, :]` = Block 1's angles (from prev_buf)
- Transformer sees 3 positions with causal mask
- Predict from position 2

##### **3.2.3 Reshape Output**

```python
# Y shape: [4, 3, 21]
# Reshape to [B, max_blocks * angles_per_block, 1]
logits = Y.reshape(4, 3*21, 1)  # [4, 63, 1]
```

**Output:** Tensor of shape `[4, 63, 1]` with predicted angles

#### 3.3 Compute Loss: `simulate_loss_lelzz_blocks()`

```python
loss = simulate_loss_lelzz_blocks(
    batch, logits, init_cache, ref_cache,
    noise_schedules, gate_blocks=5, device, detach_base_noise=True
)
```

##### **3.3.1 Initialize States**

```python
B = 4
n = 3  # All circuits have 3 qubits

# Get initial states
states = init_cache[3]  # [32, 8]
states = states.unsqueeze(0).expand(4, -1, -1).clone()
# Shape: [4, 32, 8] = [4 circuits, 32 random inits, 8 amplitudes]
```

##### **3.3.2 Get Reference States**

```python
# Map circuit indices to rows in ref_cache
rows = [ref_cache['idx2row'][17],   # Circuit index 17 → row e.g. 12
        ref_cache['idx2row'][42],   # Circuit index 42 → row e.g. 35
        ref_cache['idx2row'][8],    # Circuit index 8 → row e.g. 5
        ref_cache['idx2row'][91]]   # Circuit index 91 → row e.g. 78

ref = ref_cache['tensor'][rows]  # [4, 32, 8]
# These are the target states after applying base circuit
```

##### **3.3.3 Reshape Predicted Angles**

```python
Lb = 12  # max(base_len)
blocks_needed = ceil(12 / 5) = 3

angles_flat = logits[:, :63, 0]  # [4, 63]
angles_blk = angles_flat.view(4, 3, 21)  # [4, 3 blocks, 21 angles]

# Example angles_blk[0, 0, :] (circuit 0, block 0, all 21 angles):
# [θ0, θ1, ..., θ20] for first PQC block
```

##### **3.3.4 Apply Base Circuit + PQC Blocks**

```python
# Get base circuit info
gate_ids = batch.base_g[:, :12]  # [4, 12] gate IDs
q1 = batch.base_q1[:, :12]       # [4, 12] primary qubits
q2 = batch.base_q2[:, :12]       # [4, 12] secondary qubits

# Get quantum structure
splits = _split_indices(3, device)
# splits[q] = (indices where qubit q is |0⟩, indices where qubit q is |1⟩)
# splits[0] = ([0,2,4,6], [1,3,5,7])
# splits[1] = ([0,1,4,5], [2,3,6,7])
# splits[2] = ([0,1,2,3], [4,5,6,7])

cx_swap = _get_two_qubit_struct(3)
# cx_swap[(c,t)] = (indices to swap for CNOT control c, target t)
```

**Main simulation loop:**

```python
t = 0          # Current gate index
blk_idx = 0    # Current block index

while t < Lb:  # While gates remain
    # Determine segment boundaries
    t_end = min(12, (blk_idx + 1) * 5)  # Block boundary
    # Block 0: t=0, t_end=5  (gates 0-4)
    # Block 1: t=5, t_end=10 (gates 5-9)
    # Block 2: t=10, t_end=12 (gates 10-11)
    
    # Apply base gates in this segment
    for tt in range(t, t_end):
        g_t = gate_ids[:, tt]     # [4] gate IDs for all 4 circuits
        q1_t = q1[:, tt]          # [4] primary qubits
        q2_t = q2[:, tt]          # [4] secondary qubits
        
        # Apply gate to all circuits in parallel
        _apply_base_step_batched(states, g_t, q1_t, q2_t, splits, cx_swap)
        # states shape: [4, 32, 8] - modified in place
    
    t = t_end
    
    # Detach gradients after first segment (optional)
    if blk_idx == 0 and detach_base_noise:
        states = states.detach()  # Don't backprop through base
    
    # Apply PQC block
    angs_block = angles_blk[:, blk_idx]  # [4, 21] angles for this block
    _apply_lelzz_pqc_block(states, angs_block, 3, splits, cx_swap, device)
    
    blk_idx += 1
```

**Detailed PQC Block Application (`_apply_lelzz_pqc_block`):**

```python
# Input: states [4, 32, 8], angs_block [4, 21]

# Extract angle groups
pre_angles = angs_block[:, :9].view(4, 3, 3)    # [4, 3 qubits, 3 angles]
theta_zz = angs_block[:, 9:12]                   # [4, 3 pairs]
post_angles = angs_block[:, 12:21].view(4, 3, 3) # [4, 3 qubits, 3 angles]

# 1. Pre-local: RZ-RX-RZ on each qubit
for q in [0, 1, 2]:
    i0, i1 = splits[q]  # Indices where qubit q is |0⟩ and |1⟩
    
    # Get angles for this qubit
    a_rz1 = pre_angles[:, q, 0]  # [4] RZ1 angles
    a_rx  = pre_angles[:, q, 1]  # [4] RX angles
    a_rz2 = pre_angles[:, q, 2]  # [4] RZ2 angles
    
    # Apply fused RZ-RX-RZ rotation
    _apply_rzrxrz_fused_pairs(states, i0, i1, a_rz1, a_rx, a_rz2)
    # This is a 3-angle rotation: U = RZ(a_rz2) RX(a_rx) RZ(a_rz1)

# States now have pre-local rotations applied

# 2. ZZ-ring: CNOT-RZ-CNOT in a ring
for q in [0, 1, 2]:
    q0 = q
    q1 = (q + 1) % 3  # Ring: 0→1, 1→2, 2→0
    
    # Pair 0: CNOT(0,1) - RZ(θ0) on qubit 1 - CNOT(0,1)
    # Pair 1: CNOT(1,2) - RZ(θ1) on qubit 2 - CNOT(1,2)
    # Pair 2: CNOT(2,0) - RZ(θ2) on qubit 0 - CNOT(2,0)
    
    _apply_cx(states, q0, q1)  # First CNOT
    
    # Apply RZ rotation on target qubit q1
    K = 32
    theta_expanded = theta_zz[:, q].unsqueeze(1).expand(4, 32).reshape(128)
    # Shape: [128] = [4 circuits × 32 states]
    
    states_flat = states.view(128, 8)  # Flatten batch and K dimensions
    
    i0, i1 = splits[q1]
    em = exp(-0.5j * theta_expanded).unsqueeze(-1)  # [128, 1]
    ep = exp(0.5j * theta_expanded).unsqueeze(-1)   # [128, 1]
    
    states_flat[:, i0] *= em  # Apply phase -θ/2 to |0⟩ subspace
    states_flat[:, i1] *= ep  # Apply phase +θ/2 to |1⟩ subspace
    
    states.copy_(states_flat.view(4, 32, 8))  # Reshape back
    
    _apply_cx(states, q0, q1)  # Second CNOT

# States now have ZZ-ring entanglement

# 3. Post-local: RZ-RX-RZ on each qubit (same as pre-local)
for q in [0, 1, 2]:
    i0, i1 = splits[q]
    a_rz1 = post_angles[:, q, 0]
    a_rx  = post_angles[:, q, 1]
    a_rz2 = post_angles[:, q, 2]
    _apply_rzrxrz_fused_pairs(states, i0, i1, a_rz1, a_rx, a_rz2)
```

##### **3.3.5 Compute Fidelity Loss**

```python
# After all base gates and PQC blocks applied:
# states: [4, 32, 8] - final predicted states
# ref: [4, 32, 8] - target reference states

# Compute overlap (inner product)
ov = (ref.conj() * states).sum(dim=-1)  # [4, 32]
# For each of 4 circuits and 32 initial states, compute ⟨ψ_ref|ψ_pred⟩

# Compute fidelity
F_per_init = ov.abs() ** 2  # [4, 32] - fidelity for each initial state
F = F_per_init.mean()        # Scalar - average over all circuits and inits

# Loss is 1 - fidelity
loss = 1 - F  # Scalar tensor

# Example:
# F = 0.47 → loss = 0.53
```

**Output:** Scalar loss tensor (e.g., `0.53`)

#### 3.4 Backward Pass

```python
# Clear gradients
opt.zero_grad(set_to_none=True)

# Backpropagation
if scaler is not None:  # Using AMP
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
else:  # No AMP
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
```

**What happens in `loss.backward()`:**

1. **Start from loss scalar**: `loss = 0.53`

2. **Gradient flows back through fidelity computation**:
   ```
   ∂loss/∂F = -1
   ∂F/∂ov = 2 * ov.conj()
   ∂ov/∂states = ref.conj()
   ```

3. **Gradients flow through quantum gates (PQC blocks)**:
   - `_apply_rzrxrz_fused_pairs`: Computes ∂states/∂angles
   - Each gate applies a unitary: `|ψ'⟩ = U(θ)|ψ⟩`
   - Gradient: `∂|ψ'⟩/∂θ = ∂U/∂θ |ψ⟩`
   - For RZ(θ): `∂RZ/∂θ = -i/2 * Z * RZ(θ)`
   - For RX(θ): `∂RX/∂θ = -i/2 * X * RX(θ)`

4. **Gradients flow through transformer**:
   - Through `self.head`: `∂logits/∂h_last`
   - Through `self.head_ln`: `∂h_last/∂h`
   - Through `self.encoder`: Multi-head attention gradients
   - Through `self.in_proj`: Input projection gradients

5. **Clip gradients**: Ensure max norm is 1.0
   ```python
   total_norm = sqrt(sum(p.grad.norm(2)^2 for p in model.parameters()))
   if total_norm > 1.0:
       for p in model.parameters():
           p.grad *= (1.0 / total_norm)
   ```

6. **Update parameters**:
   ```python
   # AdamW update rule
   for param in model.parameters():
       # Compute first moment (mean)
       m = β1 * m + (1-β1) * grad
       # Compute second moment (variance)
       v = β2 * v + (1-β2) * grad^2
       # Bias correction
       m_hat = m / (1 - β1^t)
       v_hat = v / (1 - β2^t)
       # Update with weight decay
       param -= lr * (m_hat / (sqrt(v_hat) + ε) + λ * param)
   ```

#### 3.5 Accumulate Statistics

```python
total_loss += loss.item() * batch.base_g.size(0)  # 0.53 * 4 = 2.12
count += batch.base_g.size(0)  # += 4
```

---

## Helper Functions

### `_split_indices(n, device)`

**Purpose:** Pre-compute amplitude indices for qubit states

**Input:** `n=3` qubits

**Output:** List of tuples `[(i0, i1), ...]` for each qubit

**Logic:**
```python
# For 3 qubits, states are indexed 0-7 (binary 000-111)
# Qubit 0 (rightmost bit):
#   i0 = [0, 2, 4, 6]  # Where qubit 0 is |0⟩ (even indices)
#   i1 = [1, 3, 5, 7]  # Where qubit 0 is |1⟩ (odd indices)

# Qubit 1 (middle bit):
#   i0 = [0, 1, 4, 5]  # Where qubit 1 is |0⟩
#   i1 = [2, 3, 6, 7]  # Where qubit 1 is |1⟩

# Qubit 2 (leftmost bit):
#   i0 = [0, 1, 2, 3]  # Where qubit 2 is |0⟩
#   i1 = [4, 5, 6, 7]  # Where qubit 2 is |1⟩

splits = []
for q in range(n):
    mask_bit = 1 << q
    all_indices = torch.arange(2^n)
    i0 = all_indices[(all_indices & mask_bit) == 0]
    i1 = all_indices[(all_indices & mask_bit) != 0]
    splits.append((i0, i1))
```

### `_apply_rzrxrz_fused_pairs(states, i0, i1, a_rz1, a_rx, a_rz2)`

**Purpose:** Apply RZ-RX-RZ rotation sequence

**Input:**
- `states`: `[4, 32, 8]` quantum states
- `i0`, `i1`: Index tensors for qubit subspaces
- `a_rz1`, `a_rx`, `a_rz2`: `[4]` angle tensors (one per circuit)

**Logic:**
```python
# Apply U = RZ(a_rz2) RX(a_rx) RZ(a_rz1)

# First RZ rotation
em1 = exp(-0.5j * a_rz1).unsqueeze(-1).unsqueeze(-1)  # [4, 1, 1]
ep1 = exp(0.5j * a_rz1).unsqueeze(-1).unsqueeze(-1)   # [4, 1, 1]
states[:, :, i0] *= em1  # Broadcast over K=32 and indices
states[:, :, i1] *= ep1

# RX rotation
c = cos(0.5 * a_rx).unsqueeze(-1).unsqueeze(-1)
s = -1j * sin(0.5 * a_rx).unsqueeze(-1).unsqueeze(-1)
s0 = states[:, :, i0].clone()
s1 = states[:, :, i1].clone()
states[:, :, i0] = c * s0 + s * s1
states[:, :, i1] = s * s0 + c * s1

# Second RZ rotation
em2 = exp(-0.5j * a_rz2).unsqueeze(-1).unsqueeze(-1)
ep2 = exp(0.5j * a_rz2).unsqueeze(-1).unsqueeze(-1)
states[:, :, i0] *= em2
states[:, :, i1] *= ep2
```

### `_apply_cx(states, control, target)`

**Purpose:** Apply CNOT gate

**Input:**
- `states`: `[4, 32, 8]`
- `control`: Qubit index (e.g., 0)
- `target`: Qubit index (e.g., 1)

**Logic:**
```python
# CNOT flips target qubit when control is |1⟩
# Find indices where control=|1⟩, target=|0⟩ and target=|1⟩
# Swap amplitudes at those index pairs

# Example: control=0, target=1
# Swap: 0b001 ↔ 0b011 (indices 1 ↔ 3)
#       0b101 ↔ 0b111 (indices 5 ↔ 7)

control_bit = 1 << control
target_bit = 1 << target

all_idx = torch.arange(2^n)
# Find where control is 1 and target is 0
i0 = all_idx[(all_idx & control_bit != 0) & (all_idx & target_bit == 0)]
# Find where control is 1 and target is 1
i1 = all_idx[(all_idx & control_bit != 0) & (all_idx & target_bit != 0)]

# Swap amplitudes
temp = states[:, :, i0].clone()
states[:, :, i0] = states[:, :, i1]
states[:, :, i1] = temp
```

---

## Summary

### Key Data Flow:

1. **Input:** Batch of 4 circuits with 3 qubits, ~10-12 base gates each

2. **Model Forward:**
   - Compute gate statistics per block (counts, cumulative)
   - Autoregressively predict 21 angles for each of 3 blocks
   - Use transformer to capture dependencies
   - Output: `[4, 63, 1]` angles

3. **Simulation:**
   - Start with 32 random initial states per circuit
   - Apply base circuit gates (H, X, Z, CX, CZ)
   - Interleave PQC blocks (pre-local, ZZ-ring, post-local)
   - End with final quantum states `[4, 32, 8]`

4. **Loss:**
   - Compare with reference states (pre-computed)
   - Compute fidelity: `|⟨ψ_ref|ψ_pred⟩|²`
   - Loss = `1 - fidelity` (minimize)

5. **Optimization:**
   - Backpropagate through quantum gates and transformer
   - Update ~100M parameters with AdamW
   - Clip gradients for stability

### Computational Complexity:

- **Forward pass:** O(B × K × 2^n × T × L) where:
  - B=4 (batch), K=32 (inits), n=3 (qubits), T=3 (blocks), L=8 (layers)
  - ≈ 4 × 32 × 8 × 3 × 8 ≈ 24k operations

- **Transformer:** O(B × T² × d) for self-attention
  - 4 × 9 × 768 ≈ 28k operations

- **Backward pass:** ~2× forward pass cost

### Memory Usage:

- Model: ~400 MB (100M params × 4 bytes)
- States: ~10 KB (4 × 32 × 8 × 16 bytes)
- Gradients: ~400 MB (same as params)
- Activations: ~50 MB (cached for backprop)

**Total:** ~850 MB GPU memory per training iteration
