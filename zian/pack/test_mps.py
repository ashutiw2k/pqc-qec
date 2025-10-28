#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify MPS (Apple Silicon) functionality
"""
import torch
import sys

def test_mps_availability():
    """Test if MPS backend is available."""
    print("="*70)
    print("MPS AVAILABILITY TEST")
    print("="*70)
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    
    mps_available = torch.backends.mps.is_available()
    print(f"\nMPS available: {mps_available}")
    
    if mps_available:
        mps_built = torch.backends.mps.is_built()
        print(f"MPS built: {mps_built}")
        
        if mps_built:
            print("\n✓ MPS backend is ready!")
            return True
        else:
            print("\n✗ MPS not built in this PyTorch installation")
            return False
    else:
        print("\n✗ MPS not available (requires macOS 12.3+ and Apple Silicon)")
        return False


def test_basic_operations():
    """Test basic tensor operations on MPS."""
    print("\n" + "="*70)
    print("BASIC TENSOR OPERATIONS TEST")
    print("="*70)
    
    device = torch.device('mps')
    
    try:
        # Create tensors
        print("\n1. Creating tensors on MPS...")
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        print(f"   x.shape: {x.shape}, device: {x.device}")
        print(f"   y.shape: {y.shape}, device: {y.device}")
        
        # Matrix multiplication
        print("\n2. Matrix multiplication...")
        z = torch.matmul(x, y)
        print(f"   z.shape: {z.shape}, device: {z.device}")
        
        # Element-wise operations
        print("\n3. Element-wise operations...")
        w = x + y
        print(f"   Addition: {w.shape}")
        w = x * y
        print(f"   Multiplication: {w.shape}")
        w = torch.sin(x)
        print(f"   Sin: {w.shape}")
        
        # Complex numbers
        print("\n4. Complex number operations...")
        c = torch.randn(50, 50, dtype=torch.complex64, device=device)
        c_conj = c.conj()
        c_abs = c.abs()
        print(f"   Complex tensor: {c.shape}, dtype: {c.dtype}")
        print(f"   Conjugate: {c_conj.shape}")
        print(f"   Absolute value: {c_abs.shape}")
        
        # Reductions
        print("\n5. Reduction operations...")
        s = x.sum()
        m = x.mean()
        print(f"   Sum: {s.item():.4f}")
        print(f"   Mean: {m.item():.4f}")
        
        print("\n✓ All basic operations passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_neural_network():
    """Test neural network operations on MPS."""
    print("\n" + "="*70)
    print("NEURAL NETWORK TEST")
    print("="*70)
    
    device = torch.device('mps')
    
    try:
        # Simple model
        print("\n1. Creating a simple neural network...")
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10)
        ).to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {param_count:,}")
        
        # Forward pass
        print("\n2. Forward pass...")
        x = torch.randn(32, 100, device=device)
        y = model(x)
        print(f"   Input: {x.shape}")
        print(f"   Output: {y.shape}")
        
        # Loss and backward
        print("\n3. Computing loss and backward pass...")
        target = torch.randint(0, 10, (32,), device=device)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(y, target)
        print(f"   Loss: {loss.item():.4f}")
        
        loss.backward()
        print("   ✓ Backward pass completed")
        
        # Check gradients
        print("\n4. Checking gradients...")
        has_grads = all(p.grad is not None for p in model.parameters() if p.requires_grad)
        print(f"   All parameters have gradients: {has_grads}")
        
        print("\n✓ Neural network test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transformer():
    """Test transformer operations on MPS."""
    print("\n" + "="*70)
    print("TRANSFORMER TEST")
    print("="*70)
    
    device = torch.device('mps')
    
    try:
        print("\n1. Creating transformer encoder...")
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048,
            dropout=0.1, batch_first=True,
            norm_first=True
        )
        transformer = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=4,
            enable_nested_tensor=False  # Important for MPS
        ).to(device)
        
        param_count = sum(p.numel() for p in transformer.parameters())
        print(f"   Parameters: {param_count:,}")
        
        # Create input
        print("\n2. Creating input tensor...")
        batch_size = 16
        seq_len = 32
        x = torch.randn(batch_size, seq_len, 512, device=device)
        print(f"   Input shape: {x.shape}")
        
        # Forward pass
        print("\n3. Forward pass through transformer...")
        y = transformer(x)
        print(f"   Output shape: {y.shape}")
        
        # With attention mask
        print("\n4. Forward pass with causal mask...")
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        y_masked = transformer(x, mask=mask)
        print(f"   Output shape: {y_masked.shape}")
        
        print("\n✓ Transformer test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantum_operations():
    """Test quantum-specific operations on MPS."""
    print("\n" + "="*70)
    print("QUANTUM OPERATIONS TEST")
    print("="*70)
    
    device = torch.device('mps')
    
    try:
        # Create quantum state
        print("\n1. Creating quantum state...")
        n_qubits = 3
        dim = 2 ** n_qubits
        batch_size = 8
        k_states = 16
        
        states = torch.randn(batch_size, k_states, dim, dtype=torch.complex64, device=device)
        states = states / torch.sqrt((states.abs() ** 2).sum(dim=-1, keepdim=True))
        print(f"   State shape: {states.shape}")
        print(f"   Normalized: {torch.allclose(torch.sum(states.abs()**2, dim=-1), torch.ones(batch_size, k_states, device=device))}")
        
        # Apply phase
        print("\n2. Applying RZ phase rotation...")
        angles = torch.randn(batch_size, k_states, device=device)
        phase = torch.exp(-0.5j * angles.unsqueeze(-1))
        states_rotated = states * phase
        print(f"   Rotated state shape: {states_rotated.shape}")
        
        # Compute fidelity
        print("\n3. Computing fidelity...")
        ref_states = torch.randn(batch_size, k_states, dim, dtype=torch.complex64, device=device)
        ref_states = ref_states / torch.sqrt((ref_states.abs() ** 2).sum(dim=-1, keepdim=True))
        
        overlap = (ref_states.conj() * states).sum(dim=-1)
        fidelity = (overlap.abs() ** 2).mean()
        print(f"   Fidelity: {fidelity.item():.6f}")
        
        # Gradient test
        print("\n4. Testing gradients...")
        angles_learnable = torch.randn(batch_size, k_states, device=device, requires_grad=True)
        phase_learnable = torch.exp(-0.5j * angles_learnable.unsqueeze(-1))
        states_learnable = states * phase_learnable
        overlap_learnable = (ref_states.conj() * states_learnable).sum(dim=-1)
        fidelity_learnable = (overlap_learnable.abs() ** 2).mean()
        
        fidelity_learnable.backward()
        print(f"   Gradient computed: {angles_learnable.grad is not None}")
        print(f"   Gradient norm: {angles_learnable.grad.norm().item():.6f}")
        
        print("\n✓ Quantum operations test passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("MPS FUNCTIONALITY TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Test 1: Availability
    results['availability'] = test_mps_availability()
    
    if not results['availability']:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("\n✗ MPS not available. Cannot proceed with further tests.")
        print("\nRequirements:")
        print("  - macOS 12.3 or later")
        print("  - Apple Silicon (M1/M2/M3)")
        print("  - PyTorch 2.0 or later with MPS support")
        print("\nInstall/upgrade PyTorch:")
        print("  pip install --upgrade torch torchvision")
        return
    
    # Test 2: Basic operations
    results['basic_ops'] = test_basic_operations()
    
    # Test 3: Neural network
    results['neural_network'] = test_neural_network()
    
    # Test 4: Transformer
    results['transformer'] = test_transformer()
    
    # Test 5: Quantum operations
    results['quantum_ops'] = test_quantum_operations()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"\n{test_name:20s}: {status}")
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nYou can now use train_lelzz_mps.py for training on Apple Silicon.")
        print("\nExample usage:")
        print("  python -m pqcqec.train_lelzz_mps \\")
        print("    --data-path data/json_data/3q_10g_5blk_data \\")
        print("    --n-qubits 3 \\")
        print("    --epochs 100 \\")
        print("    --batch-size 64")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease check the errors above and ensure:")
        print("  1. PyTorch is properly installed with MPS support")
        print("  2. You're running on Apple Silicon with macOS 12.3+")
        print("  3. No conflicting PyTorch installations exist")
    print("="*70)


if __name__ == "__main__":
    main()
