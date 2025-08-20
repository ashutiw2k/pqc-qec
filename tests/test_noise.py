import jax.numpy as jnp
import pennylane as qml

from pqcqec.noise.simple_noise import PennylaneNoisyGates


def test_apply_gate_unsupported_raises():
    n = PennylaneNoisyGates()
    try:
        n.apply_gate("u2", [0])
    except ValueError as e:
        assert "not supported" in str(e)
    else:
        assert False, "Expected ValueError for unsupported gate name"


def test_apply_gate_param_required_for_pqc():
    n = PennylaneNoisyGates()
    try:
        n.apply_gate("rx", [0])  # angle missing
    except AssertionError:
        pass
    else:
        assert False, "Expected AssertionError when angle not provided for parameterized gate"


def test_zero_noise_behaves_as_ideal_x():
    # With zero noise bounds, noisy X should equal ideal X
    n = PennylaneNoisyGates(x_rad=0.0, z_rad=0.0, delta_x=0.0, delta_z=0.0)

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit():
        n.apply_gate("x", [0])
        return qml.state()

    state = circuit()
    # Expect |1> state (up to numerical tolerance)
    expected = jnp.array([0.0 + 0.0j, 1.0 + 0.0j])
    assert jnp.allclose(state, expected, atol=1e-7)


def test_pqc_parameterized_rotation_executes():
    # Ensure parameterized PQC gates route correctly and execute
    n = PennylaneNoisyGates(x_rad=0.0, z_rad=0.0, delta_x=0.0, delta_z=0.0)
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(theta):
        # Start in |0>, apply RX(theta); when theta=pi, expect |1>
        n.apply_gate("rx", 0, angle=theta)
        return qml.state()

    state = circuit(jnp.pi)
    expected = jnp.array([0.0 + 0.0j, 1.0 + 0.0j])
    # RX(pi)|0> == -i|1>, which differs by a global phase; compare via fidelity
    state = state / (jnp.linalg.norm(state) + 1e-12)
    expected = expected / (jnp.linalg.norm(expected) + 1e-12)
    fidelity = jnp.abs(jnp.vdot(expected, state)) ** 2
    assert float(fidelity) > 1 - 1e-6
