"""Test the new PQC type system in templates."""
import numpy as np
from pqcqec.circuits.templates import build_pqc_circuit_template


def test_pqc_type_none():
    """Test that pqc_type='none' creates no PQC layers."""
    base_ops = [('x', [0], []), ('h', [1], [])]
    template = build_pqc_circuit_template(
        base_ops, num_qubits=2, num_gate_blocks=1, 
        add_noise=False, pqc_type="none"
    )
    
    param_dict = {'base': np.array([0.0, 0.0])}
    circuit_ops = template.instantiate(param_dict)
    
    # Should only have the 2 base gates
    assert len(circuit_ops) == 2
    assert circuit_ops[0][0] == 'x'
    assert circuit_ops[1][0] == 'h'


def test_pqc_type_rzrxrz_only():
    """Test that pqc_type='rzrxrz' creates only local gates (no entanglement)."""
    base_ops = [('x', [0], [])]
    template = build_pqc_circuit_template(
        base_ops, num_qubits=2, num_gate_blocks=1, 
        add_noise=False, pqc_type="rzrxrz"
    )
    
    param_dict = {
        'base': np.array([0.0]),
        'pre_params': np.random.randn(1, 2, 3)
    }
    circuit_ops = template.instantiate(param_dict)
    
    # Should have: 1 base gate + 6 PQC gates (3 per qubit, 2 qubits)
    assert len(circuit_ops) == 7
    
    # Check structure: x, rz, rx, rz (qubit 0), rz, rx, rz (qubit 1)
    gate_names = [op[0] for op in circuit_ops]
    assert gate_names == ['x', 'rz', 'rx', 'rz', 'rz', 'rx', 'rz']
    
    # No CNOT gates (no entanglement)
    assert 'cx' not in gate_names


def test_pqc_type_rzrxrz_zz_ring():
    """Test that pqc_type='rzrxrz_zz_ring' creates full LEL-ZZ structure."""
    base_ops = [('x', [0], [])]
    template = build_pqc_circuit_template(
        base_ops, num_qubits=2, num_gate_blocks=1, 
        add_noise=False, pqc_type="rzrxrz_zz_ring"
    )
    
    param_dict = {
        'base': np.array([0.0]),
        'pre_params': np.random.randn(1, 2, 3),
        'theta_zz': np.random.randn(1, 2),
        'post_params': np.random.randn(1, 2, 3)
    }
    circuit_ops = template.instantiate(param_dict)
    
    # Should have: 1 base + 6 pre + 6 entangling (2*(cx+rz+cx)) + 6 post = 19 gates
    assert len(circuit_ops) == 19
    
    # Check that CNOT gates are present (entanglement)
    gate_names = [op[0] for op in circuit_ops]
    assert gate_names.count('cx') == 4  # 2 CNOTs per qubit in ring


def test_pqc_type_rxrzry_only():
    """Test that pqc_type='rxrzry' creates Rx-Rz-Ry local gates."""
    base_ops = [('x', [0], [])]
    template = build_pqc_circuit_template(
        base_ops, num_qubits=2, num_gate_blocks=1, 
        add_noise=False, pqc_type="rxrzry"
    )
    
    param_dict = {
        'base': np.array([0.0]),
        'pre_params': np.random.randn(1, 2, 3)
    }
    circuit_ops = template.instantiate(param_dict)
    
    # Should have: 1 base gate + 6 PQC gates
    assert len(circuit_ops) == 7
    
    # Check structure: x, rx, rz, ry (qubit 0), rx, rz, ry (qubit 1)
    gate_names = [op[0] for op in circuit_ops]
    assert gate_names == ['x', 'rx', 'rz', 'ry', 'rx', 'rz', 'ry']


def test_pqc_type_rxrzry_zz_ring():
    """Test that pqc_type='rxrzry_zz_ring' creates Rx-Rz-Ry + entanglement."""
    base_ops = [('x', [0], [])]
    template = build_pqc_circuit_template(
        base_ops, num_qubits=2, num_gate_blocks=1, 
        add_noise=False, pqc_type="rxrzry_zz_ring"
    )
    
    param_dict = {
        'base': np.array([0.0]),
        'pre_params': np.random.randn(1, 2, 3),
        'theta_zz': np.random.randn(1, 2),
        'post_params': np.random.randn(1, 2, 3)
    }
    circuit_ops = template.instantiate(param_dict)
    
    # Should have full structure with Rx-Rz-Ry gates
    assert len(circuit_ops) == 19
    
    # Check that we have the right gate types
    gate_names = [op[0] for op in circuit_ops]
    assert gate_names.count('rx') == 4  # 2 pre + 2 post
    assert gate_names.count('rz') == 6  # 2 pre + 2 ZZ + 2 post
    assert gate_names.count('ry') == 4  # 2 pre + 2 post
    assert gate_names.count('cx') == 4  # Entangling CNOTs


def test_pqc_type_single_qubit_no_entanglement():
    """Test that single-qubit circuits don't add entanglement even with _zz_ring."""
    base_ops = [('x', [0], [])]
    template = build_pqc_circuit_template(
        base_ops, num_qubits=1, num_gate_blocks=1, 
        add_noise=False, pqc_type="rzrxrz_zz_ring"
    )
    
    param_dict = {
        'base': np.array([0.0]),
        'pre_params': np.random.randn(1, 1, 3)
    }
    circuit_ops = template.instantiate(param_dict)
    
    # Should have: 1 base + 3 pre (no entanglement or post for single qubit)
    assert len(circuit_ops) == 4
    
    # No CNOT gates
    gate_names = [op[0] for op in circuit_ops]
    assert 'cx' not in gate_names


def test_pqc_type_invalid():
    """Test that invalid PQC type raises ValueError."""
    base_ops = [('x', [0], [])]
    try:
        template = build_pqc_circuit_template(
            base_ops, num_qubits=2, num_gate_blocks=1, 
            add_noise=False, pqc_type="invalid_type"
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown PQC type" in str(e)


def test_pqc_type_case_insensitive():
    """Test that PQC type is case-insensitive."""
    base_ops = [('x', [0], [])]
    
    template1 = build_pqc_circuit_template(
        base_ops, num_qubits=2, num_gate_blocks=1, 
        add_noise=False, pqc_type="rzrxrz"
    )
    
    template2 = build_pqc_circuit_template(
        base_ops, num_qubits=2, num_gate_blocks=1, 
        add_noise=False, pqc_type="RZRXRZ"
    )
    
    # Both should produce same structure
    assert len(template1.gates) == len(template2.gates)
