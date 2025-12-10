"""
Unit tests for ChiModel with variable hidden layer dimensions.

Tests backward compatibility and new variable dimension functionality.
"""

import torch
import warnings
from model import ChiModel, create_model


def test_backward_compatibility_scalar():
    """Test that old single hidden_dim (scalar) still works."""
    print("\n=== Test 1: Backward Compatibility (Scalar hidden_dim) ===")

    # Should work with deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = ChiModel(input_dim=100, hidden_dim=128, n_layers=3)

        # Verify deprecation warning was raised
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()

    # Verify it was converted to list format internally
    assert model.hidden_dims == [128, 128, 128]
    assert model.n_layers == 3

    # Test forward pass
    x = torch.randn(16, 100)
    output = model(x)
    assert output.shape == (16, 2), f"Expected shape (16, 2), got {output.shape}"

    print("✓ Scalar hidden_dim converted to [128, 128, 128]")
    print("✓ Deprecation warning raised")
    print("✓ Forward pass works correctly")
    print("PASSED")


def test_backward_compatibility_list_as_hidden_dim():
    """Test that passing list as hidden_dim parameter works."""
    print("\n=== Test 2: Backward Compatibility (List as hidden_dim) ===")

    model = ChiModel(input_dim=100, hidden_dim=[256, 128, 64], n_layers=3)

    assert model.hidden_dims == [256, 128, 64]
    assert model.n_layers == 3

    # Test forward pass
    x = torch.randn(16, 100)
    output = model(x)
    assert output.shape == (16, 2)

    print("✓ List as hidden_dim parameter works")
    print("✓ Forward pass works correctly")
    print("PASSED")


def test_variable_dimensions_new_format():
    """Test new variable dimension functionality using hidden_dims."""
    print("\n=== Test 3: Variable Dimensions (New Format) ===")

    dims = [256, 128, 64]
    model = ChiModel(input_dim=100, hidden_dims=dims, n_layers=3)

    assert model.hidden_dims == dims
    assert model.n_layers == 3

    # Test forward pass
    x = torch.randn(16, 100)
    output = model(x)
    assert output.shape == (16, 2)

    # Verify layer dimensions
    # Layer 0: input(100) -> 256
    assert model.network[0].in_features == 100
    assert model.network[0].out_features == 256

    # Layer 1: 256 -> 128
    assert model.network[3].in_features == 256
    assert model.network[3].out_features == 128

    # Layer 2: 128 -> 64
    assert model.network[6].in_features == 128
    assert model.network[6].out_features == 64

    # Output layer: 64 -> 2
    assert model.network[9].in_features == 64
    assert model.network[9].out_features == 2

    print("✓ hidden_dims parameter works")
    print("✓ Layer dimensions correct: 100→256→128→64→2")
    print("✓ Forward pass works correctly")
    print("PASSED")


def test_freeze_layers_with_variable_dims():
    """Test that layer freezing still works with variable dimensions."""
    print("\n=== Test 4: Layer Freezing with Variable Dimensions ===")

    model = ChiModel(input_dim=100, hidden_dims=[256, 128, 64], n_layers=3)

    # Initially all trainable
    trainable_params = model.get_num_trainable_params()
    total_params = model.get_num_total_params()
    assert trainable_params == total_params
    print(f"✓ Initially all {trainable_params:,} params trainable")

    # Freeze first layer
    model.freeze_n_layers(1)

    # First layer (modules 0-2) should be frozen
    assert not model.network[0].weight.requires_grad
    assert not model.network[0].bias.requires_grad

    # Second layer (modules 3-5) should be trainable
    assert model.network[3].weight.requires_grad
    assert model.network[3].bias.requires_grad

    # Third layer (modules 6-8) should be trainable
    assert model.network[6].weight.requires_grad
    assert model.network[6].bias.requires_grad

    # Output layer should be trainable
    assert model.network[9].weight.requires_grad
    assert model.network[9].bias.requires_grad

    trainable_after_freeze = model.get_num_trainable_params()
    assert trainable_after_freeze < total_params
    print(f"✓ After freeze(1): {trainable_after_freeze:,} params trainable")

    # Freeze all layers
    model.freeze_n_layers(3)
    trainable_only_output = model.get_num_trainable_params()
    assert trainable_only_output < trainable_after_freeze
    print(f"✓ After freeze(3): {trainable_only_output:,} params trainable (output only)")

    print("PASSED")


def test_create_model_factory():
    """Test the create_model factory function with both formats."""
    print("\n=== Test 5: create_model() Factory Function ===")

    # Test with new format
    model1 = create_model(
        input_dim=100,
        hidden_dims=[512, 256, 128],
        n_layers=3,
        dropout_rate=0.2,
        device=torch.device('cpu')
    )
    assert model1.hidden_dims == [512, 256, 128]
    print("✓ create_model() works with hidden_dims")

    # Test with legacy format (scalar)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        model2 = create_model(
            input_dim=100,
            hidden_dim=128,
            n_layers=3,
            dropout_rate=0.2,
            device=torch.device('cpu')
        )
    assert model2.hidden_dims == [128, 128, 128]
    print("✓ create_model() works with scalar hidden_dim")

    # Test with legacy format (list)
    model3 = create_model(
        input_dim=100,
        hidden_dim=[256, 128, 64],
        n_layers=3,
        dropout_rate=0.2,
        device=torch.device('cpu')
    )
    assert model3.hidden_dims == [256, 128, 64]
    print("✓ create_model() works with list hidden_dim")

    print("PASSED")


def test_error_handling():
    """Test that appropriate errors are raised for invalid inputs."""
    print("\n=== Test 6: Error Handling ===")

    # Test 1: No dimension specified
    try:
        model = ChiModel(input_dim=100, n_layers=3)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Must specify either hidden_dim or hidden_dims" in str(e)
        print("✓ Raises error when no dimension specified")

    # Test 2: hidden_dims length mismatch
    try:
        model = ChiModel(input_dim=100, hidden_dims=[128, 64], n_layers=3)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must match n_layers" in str(e)
        print("✓ Raises error when hidden_dims length doesn't match n_layers")

    # Test 3: hidden_dim list length mismatch
    try:
        model = ChiModel(input_dim=100, hidden_dim=[128, 64], n_layers=3)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must match n_layers" in str(e)
        print("✓ Raises error when hidden_dim list length doesn't match n_layers")

    print("PASSED")


def test_predict_chi():
    """Test the predict_chi method with variable dimensions."""
    print("\n=== Test 7: predict_chi() Method ===")

    model = ChiModel(input_dim=100, hidden_dims=[256, 128, 64], n_layers=3)
    model.eval()

    batch_size = 16
    x = torch.randn(batch_size, 100)
    T = torch.rand(batch_size) * 100 + 273.15  # Random temperatures 273-373K

    # Test with 1D temperature
    chi = model.predict_chi(x, T)
    assert chi.shape == (batch_size,)
    print("✓ predict_chi() works with 1D temperature")

    # Test with 2D temperature
    T_2d = T.unsqueeze(1)
    chi_2d = model.predict_chi(x, T_2d)
    assert chi_2d.shape == (batch_size,)
    assert torch.allclose(chi, chi_2d)
    print("✓ predict_chi() works with 2D temperature")

    print("PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("Running ChiModel Tests")
    print("=" * 80)

    tests = [
        test_backward_compatibility_scalar,
        test_backward_compatibility_list_as_hidden_dim,
        test_variable_dimensions_new_format,
        test_freeze_layers_with_variable_dims,
        test_create_model_factory,
        test_error_handling,
        test_predict_chi
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 80)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
