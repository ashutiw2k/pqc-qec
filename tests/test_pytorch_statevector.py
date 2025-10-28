"""
Tests for PyTorch statevector simulator.
"""

import torch
import pytest
from pqcqec.simulate.pytorch_statevector import (
    torch_create_zero_state,
    torch_create_ones_state,
    torch_run_circuit_with_state,
    torch_run_many_states,
    build_torch_circuit,
    apply_x,
    apply_z,
    apply_h,
    apply_rx,
    apply_ry,
    apply_rz,
    apply_cx,
    apply_cz,
)


def test_create_zero_state():
    """Test creation of |0...0⟩ state."""
    n_qubits = 3
    state = torch_create_zero_state(n_qubits)
    
    assert state.shape == (2**n_qubits,)
    assert state.dtype == torch.complex64
    assert torch.isclose(state[0], torch.tensor(1.0+0.0j))
    assert torch.allclose(state[1:], torch.zeros(2**n_qubits - 1, dtype=torch.complex64))


def test_create_ones_state():
    """Test creation of |1...1⟩ state."""
    n_qubits = 3
    state = torch_create_ones_state(n_qubits)
    
    assert state.shape == (2**n_qubits,)
    assert state.dtype == torch.complex64
    assert torch.isclose(state[-1], torch.tensor(1.0+0.0j))
    assert torch.allclose(state[:-1], torch.zeros(2**n_qubits - 1, dtype=torch.complex64))


def test_apply_x_gate():
    """Test X gate flips computational basis states."""
    n_qubits = 1
    
    # X|0⟩ = |1⟩
    state0 = torch_create_zero_state(n_qubits)
    state1 = apply_x(state0, n_qubits, 0)
    expected = torch.tensor([0.0+0.0j, 1.0+0.0j], dtype=torch.complex64)
    assert torch.allclose(state1, expected)
    
    # X|1⟩ = |0⟩
    state1_in = torch_create_ones_state(n_qubits)
    state0_out = apply_x(state1_in, n_qubits, 0)
    expected = torch.tensor([1.0+0.0j, 0.0+0.0j], dtype=torch.complex64)
    assert torch.allclose(state0_out, expected)


def test_apply_z_gate():
    """Test Z gate applies phase."""
    n_qubits = 1
    
    # Z|0⟩ = |0⟩
    state0 = torch_create_zero_state(n_qubits)
    result = apply_z(state0, n_qubits, 0)
    assert torch.allclose(result, state0)
    
    # Z|1⟩ = -|1⟩
    state1 = torch_create_ones_state(n_qubits)
    result = apply_z(state1, n_qubits, 0)
    expected = torch.tensor([0.0+0.0j, -1.0+0.0j], dtype=torch.complex64)
    assert torch.allclose(result, expected)


def test_apply_h_gate():
    """Test Hadamard gate creates superposition."""
    n_qubits = 1
    
    # H|0⟩ = (|0⟩ + |1⟩)/√2
    state0 = torch_create_zero_state(n_qubits)
    result = apply_h(state0, n_qubits, 0)
    expected = torch.tensor([1.0, 1.0], dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0))
    assert torch.allclose(result, expected, atol=1e-6)


def test_apply_rx_gate():
    """Test RX rotation gate."""
    n_qubits = 1
    theta = torch.tensor(torch.pi / 2)
    
    # RX(π/2)|0⟩
    state0 = torch_create_zero_state(n_qubits)
    result = apply_rx(state0, n_qubits, 0, theta)
    
    # Should be approximately (|0⟩ - i|1⟩)/√2
    expected = torch.tensor([1.0/torch.sqrt(torch.tensor(2.0)) + 0.0j, 
                            0.0 - 1j/torch.sqrt(torch.tensor(2.0))], dtype=torch.complex64)
    assert torch.allclose(result, expected, atol=1e-6)


def test_apply_ry_gate():
    """Test RY rotation gate."""
    n_qubits = 1
    theta = torch.tensor(torch.pi / 2)
    
    # RY(π/2)|0⟩
    state0 = torch_create_zero_state(n_qubits)
    result = apply_ry(state0, n_qubits, 0, theta)
    
    # Should be approximately (|0⟩ + |1⟩)/√2
    expected = torch.tensor([1.0, 1.0], dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0))
    assert torch.allclose(result, expected, atol=1e-6)


def test_apply_rz_gate():
    """Test RZ rotation gate."""
    n_qubits = 1
    theta = torch.tensor(torch.pi)
    
    # RZ(π) is essentially Z gate up to global phase
    state = torch.tensor([1.0+0.0j, 1.0+0.0j], dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0))
    result = apply_rz(state, n_qubits, 0, theta)
    
    # Should apply phases e^(-iπ/2) to |0⟩ and e^(iπ/2) to |1⟩
    # This is -i|0⟩ + i|1⟩ up to normalization
    assert torch.isclose(torch.abs(result[0]), torch.tensor(1.0/torch.sqrt(torch.tensor(2.0))), atol=1e-6)
    assert torch.isclose(torch.abs(result[1]), torch.tensor(1.0/torch.sqrt(torch.tensor(2.0))), atol=1e-6)


def test_apply_cx_gate():
    """Test CNOT gate."""
    n_qubits = 2
    
    # CX|00⟩ = |00⟩
    state00 = torch_create_zero_state(n_qubits)
    result = apply_cx(state00, n_qubits, 0, 1)
    assert torch.allclose(result, state00)
    
    # CX|10⟩ = |11⟩
    # |10⟩ is index 2 in 2-qubit system
    state10 = torch.zeros(4, dtype=torch.complex64)
    state10[2] = 1.0 + 0.0j
    result = apply_cx(state10, n_qubits, 0, 1)
    expected = torch.zeros(4, dtype=torch.complex64)
    expected[3] = 1.0 + 0.0j  # |11⟩
    assert torch.allclose(result, expected)


def test_apply_cz_gate():
    """Test CZ gate."""
    n_qubits = 2
    
    # CZ|11⟩ = -|11⟩
    state11 = torch.zeros(4, dtype=torch.complex64)
    state11[3] = 1.0 + 0.0j
    result = apply_cz(state11, n_qubits, 0, 1)
    expected = torch.zeros(4, dtype=torch.complex64)
    expected[3] = -1.0 + 0.0j
    assert torch.allclose(result, expected)


def test_build_torch_circuit():
    """Test circuit builder."""
    circuit_ops = [
        ('h', [0], []),
        ('cx', [0, 1], []),
        ('rz', [1], [torch.pi/4]),
    ]
    
    gate_ids, wire1s, wire2s, thetas = build_torch_circuit(circuit_ops)
    
    assert gate_ids.shape[0] == 3
    assert wire1s.shape[0] == 3
    assert wire2s.shape[0] == 3
    assert thetas.shape[0] == 3
    
    # Check dtypes
    assert gate_ids.dtype == torch.int32
    assert wire1s.dtype == torch.int32
    assert wire2s.dtype == torch.int32


def test_run_circuit_with_state():
    """Test running a simple circuit."""
    n_qubits = 2
    
    # Build circuit: H on qubit 0, then CX(0,1)
    # This should create a Bell state
    circuit_ops = [
        ('h', [0], []),
        ('cx', [0, 1], []),
    ]
    
    gate_ids, wire1s, wire2s, thetas = build_torch_circuit(circuit_ops)
    state0 = torch_create_zero_state(n_qubits)
    final_state = torch_run_circuit_with_state(state0, n_qubits, gate_ids, wire1s, wire2s, thetas)
    
    # Bell state: (|00⟩ + |11⟩)/√2
    expected = torch.zeros(4, dtype=torch.complex64)
    expected[0] = 1.0 / torch.sqrt(torch.tensor(2.0))
    expected[3] = 1.0 / torch.sqrt(torch.tensor(2.0))
    
    assert torch.allclose(final_state, expected, atol=1e-6)


def test_run_many_states():
    """Test running circuit on batch of states."""
    n_qubits = 1
    batch_size = 3
    
    # Simple circuit: X gate
    circuit_ops = [('x', [0], [])]
    gate_ids, wire1s, wire2s, thetas = build_torch_circuit(circuit_ops)
    
    # Create batch of |0⟩ states
    states_in = torch.stack([torch_create_zero_state(n_qubits) for _ in range(batch_size)])
    states_out = torch_run_many_states(n_qubits, gate_ids, wire1s, wire2s, thetas, states_in)
    
    # All should be |1⟩ states
    expected = torch.stack([torch_create_ones_state(n_qubits) for _ in range(batch_size)])
    assert torch.allclose(states_out, expected)


def test_gradient_flow():
    """Test that gradients flow through the circuit."""
    n_qubits = 1
    
    # Create a parameterized circuit with RY gate
    theta = torch.tensor([torch.pi / 4], requires_grad=True)
    circuit_ops = [('ry', [0], [theta[0]])]
    
    gate_ids, wire1s, wire2s, thetas = build_torch_circuit(circuit_ops)
    
    # Run circuit
    state0 = torch_create_zero_state(n_qubits)
    final_state = torch_run_circuit_with_state(state0, n_qubits, gate_ids, wire1s, wire2s, thetas)
    
    # Compute some loss (e.g., expectation value of Z)
    # Z = [[1, 0], [0, -1]], so <Z> = |a0|^2 - |a1|^2
    probs = torch.abs(final_state) ** 2
    loss = probs[0] - probs[1]
    
    # Backpropagate
    loss.backward()
    
    # Check that gradient exists and is non-zero
    assert theta.grad is not None
    assert not torch.allclose(theta.grad, torch.zeros_like(theta.grad))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
