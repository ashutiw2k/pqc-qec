import torch
from torch import nn
from typing import List, Tuple, Optional
from tqdm.auto import tqdm

from .tokenizer import SimpleCircuitTokenizer
from .torch_loss_functions import torch_fidelity_loss

from ..simulate.simulate import run_circuit_with_noise_model_torch
from ..circuits.modify import interleave_tensor_pqc_in_circuit_torch
from ..noise.simple_noise import PennylaneNoisyGates

@torch.no_grad()
def evaluate_epoch_torch(model: nn.Module, dataloader, loss_fn, device: torch.device,
                         noise_dist: dict, tokenizer: SimpleCircuitTokenizer) -> float:
    """Evaluates the model for one epoch using PyTorch-native simulation."""
    model.eval()
    total_loss = 0.0
    no_noise_model = PennylaneNoisyGates(0, 0, 0, 0)
    noise_model = PennylaneNoisyGates(**noise_dist)

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        pred = model(input_ids, attention_mask=attention_mask)

        circuit_ops_list = [tokenizer.decode(ids.cpu().numpy()) for ids in input_ids]
        
        input_state = torch.randn((input_ids.shape[0], 2**tokenizer.num_qubits),
                                   dtype=torch.complex64, device=device)
        # input_state[:, 0] = 1.0

        # Run ideal circuits (no PQC, no noise)
        ideal_outputs_list = [
            run_circuit_with_noise_model_torch(ops, state.unsqueeze(0), no_noise_model, tokenizer.num_qubits)
            for ops, state in zip(circuit_ops_list, input_state)
        ]
        ideal_outputs_torch = torch.cat(ideal_outputs_list, dim=0)

        # Run noisy circuits with PQC parameters
        blocks = pred.shape[1]
        noisy_outputs_list = [
            run_circuit_with_noise_model_torch(
                interleave_tensor_pqc_in_circuit_torch(
                    ops, tokenizer.num_qubits, blocks, ['rz', 'rx', 'rz'], pred_thetas
                ),
                state.unsqueeze(0), noise_model, tokenizer.num_qubits
            ) for ops, state, pred_thetas in zip(circuit_ops_list, input_state, pred)
        ]
        noisy_outputs_torch = torch.cat(noisy_outputs_list, dim=0)

        loss = loss_fn(ideal_outputs_torch, noisy_outputs_torch)
        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_epoch_torch(model: nn.Module, dataloader,
                      optimizer: torch.optim.Optimizer,
                      loss_fn, device: torch.device,
                      noise_dist: dict, tokenizer: SimpleCircuitTokenizer,
                      num_input_states: int = 10,
                      grad_clip: Optional[float] = None) -> float:
    """Trains the model for one epoch using PyTorch-native simulation."""
    model.train()
    total_loss = 0.0
    no_noise_model = PennylaneNoisyGates(0, 0, 0, 0)
    noise_model = PennylaneNoisyGates(**noise_dist)

    for batch in tqdm(dataloader, desc="Training Epoch"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Get circuit structure
        circuit_ops_list = [tokenizer.decode(ids.cpu().numpy()) for ids in input_ids]

        input_state_shape = (int(input_ids.shape[0]), int(num_input_states), int(2**tokenizer.num_qubits))
        input_state = torch.randn(input_state_shape, dtype=torch.complex64, device=device)
        # input_state[:, 0] = 1.0

        # --- Forward pass ---
        optimizer.zero_grad(set_to_none=True)
        
        # This is the key part: `pred` remains a torch tensor with gradients
        pred = model(input_ids, attention_mask=attention_mask)

        # Calculate ideal outputs without tracking gradients for efficiency
        with torch.no_grad():
            ideal_outputs_list = [
                run_circuit_with_noise_model_torch(ops, state, no_noise_model, tokenizer.num_qubits)
                for ops, state in zip(circuit_ops_list, input_state)
            ]
            ideal_outputs_torch = torch.cat(ideal_outputs_list, dim=0)

        # Calculate noisy outputs WITH gradient tracking through `pred`
        blocks = pred.shape[1]
        noisy_outputs_list = [
            run_circuit_with_noise_model_torch(
                interleave_tensor_pqc_in_circuit_torch(
                    ops, tokenizer.num_qubits, blocks, ['rz', 'rx', 'rz'], pred_thetas
                ),
                state, noise_model, tokenizer.num_qubits
            ) for ops, state, pred_thetas in zip(circuit_ops_list, input_state, pred)
        ]
        noisy_outputs_torch = torch.cat(noisy_outputs_list, dim=0)

        loss = loss_fn(ideal_outputs_torch, noisy_outputs_torch)

        # --- Backward pass ---
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_torch(model: nn.Module, train_loader, val_loader,
                optimizer: torch.optim.Optimizer, loss_fn,
                device: torch.device, noise_dist: dict,
                tokenizer: SimpleCircuitTokenizer, epochs: int = 10,
                grad_clip: Optional[float] = None) -> Tuple[List[float], List[float]]:
    """Main training loop using the rewritten PyTorch-native functions."""
    train_hist, val_hist = [], []
    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch_torch(model, train_loader, optimizer, loss_fn, device,
                                  noise_dist, tokenizer, grad_clip)
        va_loss = evaluate_epoch_torch(model, val_loader, loss_fn, device,
                                     noise_dist, tokenizer)
        train_hist.append(tr_loss)
        val_hist.append(va_loss)
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.6f} | val_loss={va_loss:.6f}")
    return train_hist, val_hist