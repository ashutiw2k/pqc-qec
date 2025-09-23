import torch

def torch_cosine_loss(pred_angles: torch.Tensor, target_angles: torch.Tensor) -> torch.Tensor:
    """
    Computes cosine loss between predicted and target angles.
    """
    
    cos_loss = 1 - torch.cos(pred_angles - target_angles)
    return cos_loss.mean()


def torch_pure_state_fidelity(psi: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Computes batch fidelity F = |⟨ψ|φ⟩|² between two state-vector tensors.

    Args:
        psi: Complex tensor of shape (batch_size, 2**n)
        phi: Complex tensor of shape (batch_size, 2**n)

    Returns:
        Scalar fidelity tensor of shape (batch_size,) with values in [0, 1]
    """
    # Ensure the tensors are complex floats
    psi = psi.to(torch.complex64)
    phi = phi.to(torch.complex64)

    # Normalize each state vector in the batch along the last dimension
    psi_norm = torch.linalg.norm(psi, dim=-1, keepdim=True) + 1e-12
    phi_norm = torch.linalg.norm(phi, dim=-1, keepdim=True) + 1e-12
    psi_normalized = psi / psi_norm
    phi_normalized = phi / phi_norm

    # Compute the batch dot product.
    # The conjugate of psi_normalized is taken element-wise.
    # The sum is over the last dimension to compute the dot product for each
    # state in the batch.
    overlap = torch.sum(psi_normalized.conj() * phi_normalized, dim=-1)
    
    fidelity = torch.abs(overlap) ** 2

    # Clip to [0, 1] to handle potential numerical floating point errors
    return torch.clamp(fidelity.real, 0.0, 1.0)


def torch_fidelity_loss(ideal: torch.Tensor, measured: torch.Tensor) -> torch.Tensor:
    """
    Calculates the fidelity-based loss for a batch of states.
    Loss = 1 - F(ideal, measured)

    Args:
        ideal: Ideal state vectors, shape (batch_size, 2**n)
        measured: Measured state vectors, shape (batch_size, 2**n)

    Returns:
        The mean loss over the batch as a single scalar tensor.
    """
    fidelities = torch_pure_state_fidelity(ideal, measured)
    
    # Calculate loss for each item in the batch and then take the mean
    loss = 1.0 - fidelities
    return torch.mean(loss)


