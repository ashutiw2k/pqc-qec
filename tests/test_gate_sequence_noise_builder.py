import copy
from pqcqec.noise.builder import apply_gate_sequence_noise


def test_apply_gate_sequence_noise_empty_input_returns_empty_list():
    assert apply_gate_sequence_noise([]) == []


def test_apply_gate_sequence_noise_default_rules_transform_single_qubit_pair():
    base_ops = [
        ('h', [0], []),
        ('h', [0], []),
        ('x', [1], []),
    ]

    noisy_ops = apply_gate_sequence_noise(base_ops)

    assert noisy_ops is not base_ops
    assert noisy_ops[0] == ('h', [0], [])
    assert noisy_ops[1] == ('x', [0], [])
    assert noisy_ops[2] == ('x', [1], [])


def test_apply_gate_sequence_noise_case_insensitive_matching():
    base_ops = [
        ('H', [0], []),
        ('h', [0], []),
    ]

    noisy_ops = apply_gate_sequence_noise(base_ops)

    assert noisy_ops[1][0] == 'x'


def test_apply_gate_sequence_noise_prevents_overlapping_pairs():
    base_ops = [
        ('h', [0], []),
        ('h', [0], []),
        ('h', [0], []),
    ]

    noisy_ops = apply_gate_sequence_noise(base_ops)

    assert noisy_ops[0] == ('h', [0], [])
    assert noisy_ops[1] == ('x', [0], [])
    assert noisy_ops[2] == ('h', [0], [])


def test_apply_gate_sequence_noise_returns_original_when_no_rule_matches():
    base_ops = [
        ('h', [0], []),
        ('x', [0], []),
    ]

    noisy_ops = apply_gate_sequence_noise(base_ops)

    assert noisy_ops is base_ops


def test_apply_gate_sequence_noise_supports_multi_qubit_pairs():
    custom_rules = {
        ('cz', 'cz'): ('cz', 'swap'),
    }
    base_ops = [
        ('cz', [0, 1], []),
        ('cz', [0, 1], []),
        ('cz', [1, 2], []),
        ('cz', [1, 2], []),
    ]

    noisy_ops = apply_gate_sequence_noise(base_ops, noise=custom_rules)

    assert noisy_ops[1] == ('swap', [0, 1], [])
    assert noisy_ops[3] == ('swap', [1, 2], [])


def test_apply_gate_sequence_noise_transforms_disjoint_qubits_independently():
    base_ops = [
        ('h', [0], []),
        ('h', [0], []),
        ('x', [1], []),
        ('x', [1], []),
    ]

    noisy_ops = apply_gate_sequence_noise(base_ops)

    assert noisy_ops[1] == ('x', [0], [])
    assert noisy_ops[3] == ('z', [1], [])


def test_apply_gate_sequence_noise_does_not_mutate_input_sequence():
    base_ops = [
        ('h', [0], []),
        ('h', [0], []),
    ]
    original = copy.deepcopy(base_ops)

    _ = apply_gate_sequence_noise(base_ops)

    assert base_ops == original


def test_apply_gate_sequence_noise_preserves_parameter_references():
    theta = [0.123]
    phi = [0.456]
    base_ops = [
        ('rz', [0], theta),
        ('rz', [0], phi),
    ]
    custom_rules = {
        ('rz', 'rz'): ('ry', 'rz'),
    }

    noisy_ops = apply_gate_sequence_noise(base_ops, noise=custom_rules)

    assert noisy_ops[0] == ('ry', [0], theta)
    assert noisy_ops[1] == ('rz', [0], phi)
    assert noisy_ops is not base_ops


def test_apply_gate_sequence_noise_extended_form_with_custom_parameters():
    """Test extended transformation form that specifies custom parameters."""
    base_ops = [
        ('h', [0], []),
        ('h', [0], []),
    ]
    custom_rules = {
        ('h', 'h'): (('rx', [0.5]), ('rz', [0.3])),
    }

    noisy_ops = apply_gate_sequence_noise(base_ops, noise=custom_rules)

    assert noisy_ops[0] == ('rx', [0], [0.5])
    assert noisy_ops[1] == ('rz', [0], [0.3])


def test_apply_gate_sequence_noise_mixed_simple_and_extended_rules():
    """Test mixing simple (inherit params) and extended (custom params) rule forms."""
    base_ops = [
        ('h', [0], []),
        ('h', [0], []),
        ('x', [0], []),
        ('x', [0], []),
    ]
    custom_rules = {
        ('h', 'h'): ('h', 'x'),  # Simple form
        ('x', 'x'): (('rx', [0.1]), ('rz', [0.2])),  # Extended form
    }

    noisy_ops = apply_gate_sequence_noise(base_ops, noise=custom_rules)

    assert noisy_ops[0] == ('h', [0], [])
    assert noisy_ops[1] == ('x', [0], [])
    assert noisy_ops[2] == ('rx', [0], [0.1])
    assert noisy_ops[3] == ('rz', [0], [0.2])


def test_apply_gate_sequence_noise_four_consecutive_gates():
    """Test that HHHH correctly becomes HXHX (non-overlapping pairs)."""
    base_ops = [
        ('h', [0], []),
        ('h', [0], []),
        ('h', [0], []),
        ('h', [0], []),
    ]

    noisy_ops = apply_gate_sequence_noise(base_ops)

    assert noisy_ops[0] == ('h', [0], [])
    assert noisy_ops[1] == ('x', [0], [])
    assert noisy_ops[2] == ('h', [0], [])
    assert noisy_ops[3] == ('x', [0], [])


def test_apply_gate_sequence_noise_all_default_rules():
    """Test all three default transformation rules: HH→HX, XX→XZ, ZZ→ZH."""
    base_ops = [
        ('h', [0], []),
        ('h', [0], []),
        ('x', [0], []),
        ('x', [0], []),
        ('z', [0], []),
        ('z', [0], []),
    ]

    noisy_ops = apply_gate_sequence_noise(base_ops)

    assert noisy_ops[0] == ('h', [0], [])
    assert noisy_ops[1] == ('x', [0], [])
    assert noisy_ops[2] == ('x', [0], [])
    assert noisy_ops[3] == ('z', [0], [])
    assert noisy_ops[4] == ('z', [0], [])
    assert noisy_ops[5] == ('h', [0], [])


def test_apply_gate_sequence_noise_qubit_order_matters_for_multiqubit():
    """Test that multi-qubit gates require exact qubit order match."""
    custom_rules = {
        ('cz', 'cz'): ('cz', 'swap'),
    }
    base_ops = [
        ('cz', [0, 1], []),
        ('cz', [1, 0], []),  # Different order - should NOT match
    ]

    noisy_ops = apply_gate_sequence_noise(base_ops, noise=custom_rules)

    # Should return original since qubit order doesn't match
    assert noisy_ops is base_ops


def test_apply_gate_sequence_noise_multiple_rules_simultaneously():
    """Test applying multiple different transformation rules in one pass."""
    custom_rules = {
        ('h', 'h'): ('h', 'x'),
        ('x', 'x'): ('x', 'z'),
        ('z', 'z'): ('z', 'y'),
    }
    base_ops = [
        ('h', [0], []),
        ('h', [0], []),
        ('x', [1], []),
        ('x', [1], []),
        ('z', [2], []),
        ('z', [2], []),
    ]

    noisy_ops = apply_gate_sequence_noise(base_ops, noise=custom_rules)

    assert noisy_ops[1] == ('x', [0], [])
    assert noisy_ops[3] == ('z', [1], [])
    assert noisy_ops[5] == ('y', [2], [])


def test_apply_gate_sequence_noise_variable_length_2_to_3():
    """Test variable-length transformation: HH → HZX (2→3)."""
    base_ops = [
        ('h', [0], []),
        ('h', [0], []),
    ]
    custom_rules = {
        ('h', 'h'): [('h', []), ('z', []), ('x', [])],
    }

    noisy_ops = apply_gate_sequence_noise(base_ops, noise=custom_rules)

    assert len(noisy_ops) == 3
    assert noisy_ops[0] == ('h', [0], [])
    assert noisy_ops[1] == ('z', [0], [])
    assert noisy_ops[2] == ('x', [0], [])


def test_apply_gate_sequence_noise_variable_length_2_to_1():
    """Test variable-length reduction: HH → H (2→1)."""
    base_ops = [
        ('h', [0], []),
        ('h', [0], []),
    ]
    custom_rules = {
        ('h', 'h'): [('h', [])],
    }

    noisy_ops = apply_gate_sequence_noise(base_ops, noise=custom_rules)

    assert len(noisy_ops) == 1
    assert noisy_ops[0] == ('h', [0], [])


def test_apply_gate_sequence_noise_variable_length_with_params():
    """Test variable-length with explicit parameters."""
    base_ops = [
        ('h', [0], []),
        ('h', [0], []),
    ]
    custom_rules = {
        ('h', 'h'): [('h', []), ('rz', [0.1]), ('x', [])],
    }

    noisy_ops = apply_gate_sequence_noise(base_ops, noise=custom_rules)

    assert len(noisy_ops) == 3
    assert noisy_ops[0] == ('h', [0], [])
    assert noisy_ops[1] == ('rz', [0], [0.1])
    assert noisy_ops[2] == ('x', [0], [])


def test_apply_gate_sequence_noise_variable_length_param_inheritance():
    """Test parameter inheritance in variable-length transformations."""
    theta = [0.5]
    phi = [0.3]
    base_ops = [
        ('rz', [0], theta),
        ('rz', [0], phi),
    ]
    custom_rules = {
        ('rz', 'rz'): [('rz', None), ('rx', [0.1]), ('rz', None)],
    }

    noisy_ops = apply_gate_sequence_noise(base_ops, noise=custom_rules)

    assert len(noisy_ops) == 3
    assert noisy_ops[0] == ('rz', [0], theta)  # Inherits from first
    assert noisy_ops[1] == ('rx', [0], [0.1])  # Explicit param
    assert noisy_ops[2] == ('rz', [0], phi)    # Inherits from second


def test_apply_gate_sequence_noise_variable_length_non_overlap():
    """Test non-overlapping behavior with variable-length: HHH → HZX + H."""
    base_ops = [
        ('h', [0], []),
        ('h', [0], []),
        ('h', [0], []),
    ]
    custom_rules = {
        ('h', 'h'): [('h', []), ('z', []), ('x', [])],
    }

    noisy_ops = apply_gate_sequence_noise(base_ops, noise=custom_rules)

    assert len(noisy_ops) == 4  # 3 from first pair + 1 untransformed
    assert noisy_ops[0] == ('h', [0], [])
    assert noisy_ops[1] == ('z', [0], [])
    assert noisy_ops[2] == ('x', [0], [])
    assert noisy_ops[3] == ('h', [0], [])


def test_apply_gate_sequence_noise_variable_length_hhhh():
    """Test HHHH with variable-length: should become HZX + HZX (two transformations)."""
    base_ops = [
        ('h', [0], []),
        ('h', [0], []),
        ('h', [0], []),
        ('h', [0], []),
    ]
    custom_rules = {
        ('h', 'h'): [('h', []), ('z', []), ('x', [])],
    }

    noisy_ops = apply_gate_sequence_noise(base_ops, noise=custom_rules)

    assert len(noisy_ops) == 6  # Two 2→3 transformations
    assert noisy_ops[0] == ('h', [0], [])
    assert noisy_ops[1] == ('z', [0], [])
    assert noisy_ops[2] == ('x', [0], [])
    assert noisy_ops[3] == ('h', [0], [])
    assert noisy_ops[4] == ('z', [0], [])
    assert noisy_ops[5] == ('x', [0], [])


def test_apply_gate_sequence_noise_backward_compatible_with_list():
    """Test backward compatibility: 2-tuple rules still work alongside list rules."""
    base_ops = [
        ('h', [0], []),
        ('h', [0], []),
        ('x', [0], []),
        ('x', [0], []),
    ]
    custom_rules = {
        ('h', 'h'): ('h', 'x'),  # Old 2-tuple style
        ('x', 'x'): [('x', []), ('z', [])],  # New list style
    }

    noisy_ops = apply_gate_sequence_noise(base_ops, noise=custom_rules)

    assert noisy_ops[0] == ('h', [0], [])
    assert noisy_ops[1] == ('x', [0], [])
    assert noisy_ops[2] == ('x', [0], [])
    assert noisy_ops[3] == ('z', [0], [])
