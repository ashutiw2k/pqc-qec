from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader    
import json

import os
import sys
from pathlib import Path
# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))  # Makes other scripts and functions discoverable



from pqcqec.utils.constants import QUBITS_FOR_GATES, QISKIT_GATES, GATE_IS_DIRECTIONAL
from pqcqec.training.tokenizer import SimpleCircuitTokenizer
from pqcqec.training.torch_loss_functions import torch_fidelity_loss
from pqcqec.training.torch_train_functions import train_torch


from pqcqec.models.transformers import SimpleTransformer
from pqcqec.noise.simple_noise import PennylaneNoisyGates
from pqcqec.simulate.simulate import run_circuit_with_noise_model_torch
from pqcqec.circuits.modify import pennylane_state_embedding, interleave_tensor_pqc_in_circuit_torch


class SimpleCircuitDataset(Dataset):
    def __init__(self, tokenizer: SimpleCircuitTokenizer, circuit_data: List):
        """
        Args:
            tokenizer: An instance of SimpleCircuitTokenizer to encode circuit operations.
            circuit_data: A list of circuits, pqc params, and the measured fidelity where each circuit is a list of operations.
                         Each operation is a tuple (gate_name, qubit_indices, params).
                         Each PQC param is of a fixed shape. 
                         Each Fidelity is a single float value <= 1.0. 
        """
        self.tokenizer = tokenizer
        self.circuit_data = circuit_data

    def __len__(self) -> int:
        return len(self.circuit_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple containing:
            - Encoded tokens as a tensor of shape [L], where L is the sequence length.
            - Attention mask as a tensor of shape [L], where 1 indicates valid tokens and 0 indicates padding.
        """
        circuit = self.circuit_data[idx]
        # print(circuit)
        input_ids = self.tokenizer.encode(circuit)  # Encode the circuit into token IDs
        attention_mask = [1 for x in input_ids if x != self.tokenizer.pad_id]  # Create an attention mask with 1s for valid tokens

        # Convert to tensors
        transformer_dict = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }
        # params_tensor = torch.tensor(params, dtype=torch.float)
        # fidelity_tensor = torch.tensor(fidelity, dtype=torch.float)

        return transformer_dict
    

# CONSTANTS
PQC_GATES = ['rz', 'rx', 'rz']
DATA_PATH = 'nogit/no_uncomp/5q_500g_circuit_data/'
GOOD_DATA_PATH = DATA_PATH + 'per_seed_data/'
BAD_DATA_PATH = DATA_PATH + 'poor_fidelity/'
CONFIG_PATH = DATA_PATH + 'config.json'

with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)


NUM_QUBITS = CONFIG.get("qubits", 3)[0]
NUM_GATES = CONFIG.get("gates", 4)[0] * 2 # Multiply by 2 for uncomp gates. 
GATE_BLOCKS = CONFIG.get("gate_blocks", 4)
VALID_GATES = CONFIG.get("gate_dist", QISKIT_GATES)
if VALID_GATES:
    VALID_GATES = list(VALID_GATES.keys())
else:
    VALID_GATES = ['x', 'z', 'h', 'cx', 'cz']

print(f"Using {NUM_QUBITS} qubits, {NUM_GATES} gates, {GATE_BLOCKS} gate blocks, {VALID_GATES} valid gates.")

NOISE_DIST = {"x_rad": 0.01, "z_rad": 0.01, "delta_x": 0, "delta_z": 0}

PAD_ID = 0
UNDIRECTED_GATES = [gate for gate in VALID_GATES if not GATE_IS_DIRECTIONAL.get(gate, False)]
print(UNDIRECTED_GATES)

TRAIN_SZ = 0.8
VAL_SZ = 0.1
TEST_SZ = 0.1
BATCH_SIZE = 64

LEARNING_RATE = 5e-6
WEIGHT_DECAY = 1e-3

NUM_STATES = 100  # Number of random input states to use for fidelity estimation during training

def main():

    good_data = []

    for i, filename in enumerate(os.listdir(GOOD_DATA_PATH)):
        if i > 10000:
            break
        with open(GOOD_DATA_PATH + filename, 'r') as f:
            token_dict = json.load(f)
            good_data.append(token_dict['base_circuit_tokens'])
            f.close()


    print(f"Number of good data samples: {len(good_data)}")

    qc_tokenizer = SimpleCircuitTokenizer(gateset=VALID_GATES, 
                                      num_qubits=NUM_QUBITS, 
                                      undirected_gates=UNDIRECTED_GATES, 
                                      qubits_for_gates=QUBITS_FOR_GATES)
    
    qc_good_dataset = SimpleCircuitDataset(qc_tokenizer, good_data)

    total_size = len(qc_good_dataset)
    train_size = int(TRAIN_SZ * total_size)
    val_size = int(VAL_SZ * total_size)
    test_size = total_size - train_size - val_size
    qc_train_dataset, qc_val_dataset, qc_test_dataset = torch.utils.data.random_split(qc_good_dataset, [train_size, val_size, test_size])

    qc_train_dataloader = DataLoader(qc_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    qc_val_dataloader = DataLoader(qc_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)
    qc_test_dataloader = DataLoader(qc_test_dataset, shuffle=False, num_workers=12)

    simple_model = SimpleTransformer(vocab_size=qc_tokenizer.vocab_size,
                                  out_shape=(int(NUM_GATES//GATE_BLOCKS), NUM_QUBITS, len(PQC_GATES)),
                                  d_model=512,
                                  nhead=4,
                                  num_encoder_layers=4,
                                  dim_feedforward=1024,
                                  dropout=0.1,
                                  pad_id=qc_tokenizer.pad_id,
                                  max_len=1024)
    
    print("Transformer Model:", simple_model, sep='\n')

    optimizer = torch.optim.AdamW(simple_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


    device_str = 'cuda' if torch.cuda.is_available() else ('mps' if torch.mps.is_available() else 'cpu')
    device = torch.device(device_str)

    print(f'Using device : {device}')

    simple_model.to(device)

    loss_fn = torch_fidelity_loss
    # loss_fn = nn.MSELoss()

    train_hist, val_hist = train_torch(
        model=simple_model,
        train_loader=qc_train_dataloader,
        val_loader=qc_val_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=5,
        grad_clip=1.0,
        noise_dist=NOISE_DIST,
        tokenizer=qc_tokenizer,
        num_input_states=NUM_STATES
    )

    # Need to plot loss curves HERE.



    noise_model = PennylaneNoisyGates(**NOISE_DIST)
    no_noise_model = PennylaneNoisyGates(0,0,0,0)
    # NUM_QUBITS = 5
    ZERO_STATE = torch.zeros((2 ** NUM_QUBITS,), dtype=torch.complex64)
    ZERO_STATE[0] = 1.0

    simple_model.eval()  # Set the model to evaluation mode
    test_fid_model = []

    for i, data in enumerate(qc_test_dataloader):

        data = data

        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        # pqc_params = params[0]

        circuit_ops = qc_tokenizer.decode(input_ids[0].cpu().numpy().tolist())


        pred_theta = simple_model(input_ids, attention_mask=attention_mask)[0].to('cpu')

        circuit_interleaved = interleave_tensor_pqc_in_circuit_torch(circuit_ops, NUM_QUBITS, GATE_BLOCKS, PQC_GATES, pred_theta)

        ideal_out_state = run_circuit_with_noise_model_torch(circuit_ops, ZERO_STATE, no_noise_model, NUM_QUBITS)

        measured_modelPQC = run_circuit_with_noise_model_torch(circuit_interleaved, ZERO_STATE, noise_model, NUM_QUBITS)

        fidelity_modelPQC = torch_fidelity_loss(ideal_out_state, measured_modelPQC)

        test_fid_model.append(fidelity_modelPQC)



if __name__ == "__main__":
    main()
    