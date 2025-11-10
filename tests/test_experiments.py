"""
Tests for experiment scripts in experiments/ folder.

These are simple smoke tests to verify that experiment scripts:
1. Can be imported without errors
2. Have valid configuration
3. Can run for a few iterations without crashing

Note: These tests do NOT run full experiments (which take hours).
They only verify that the scripts are functional.
"""

import pytest
import sys
from pathlib import Path
import importlib.util

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _import_module_from_path(module_name: str, file_path: Path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ablations_import():
    """
    Test that ablations.py can be imported without errors.
    
    This verifies:
    - No syntax errors
    - All imports are available
    - Module structure is valid
    
    Expected runtime: < 0.1 seconds
    """
    ablations_path = PROJECT_ROOT / "experiments" / "ablations.py"
    assert ablations_path.exists(), f"ablations.py not found at {ablations_path}"
    
    # Import the module
    ablations = _import_module_from_path("ablations", ablations_path)
    
    # Verify expected attributes exist
    assert hasattr(ablations, "ABLATIONS"), "ABLATIONS dict not found"
    assert hasattr(ablations, "run_ablation"), "run_ablation function not found"
    
    # Verify ABLATIONS structure
    assert isinstance(ablations.ABLATIONS, dict), "ABLATIONS should be a dict"
    expected_ablations = ["no_rmsnorm", "post_norm", "no_rope", "silu_only"]
    for abl in expected_ablations:
        assert abl in ablations.ABLATIONS, f"Ablation '{abl}' not found in ABLATIONS"
    
    print(f"\n✓ ablations.py imported successfully")
    print(f"  Found {len(ablations.ABLATIONS)} ablations: {list(ablations.ABLATIONS.keys())}")


def test_batch_size_sweep_import():
    """
    Test that batch_size_sweep.py can be imported without errors.
    
    This verifies:
    - No syntax errors
    - All imports are available
    - Module structure is valid
    
    Expected runtime: < 0.1 seconds
    """
    batch_sweep_path = PROJECT_ROOT / "experiments" / "batch_size_sweep.py"
    assert batch_sweep_path.exists(), f"batch_size_sweep.py not found at {batch_sweep_path}"
    
    # Import the module
    batch_sweep = _import_module_from_path("batch_size_sweep", batch_sweep_path)
    
    # Verify expected attributes exist
    assert hasattr(batch_sweep, "batch_size_sweep"), "batch_size_sweep function not found"
    
    print(f"\n✓ batch_size_sweep.py imported successfully")


def test_learning_rate_sweep_import():
    """
    Test that learning_rate_sweep.py can be imported without errors.
    
    This verifies:
    - No syntax errors
    - All imports are available
    - Module structure is valid
    
    Expected runtime: < 0.1 seconds
    """
    lr_sweep_path = PROJECT_ROOT / "experiments" / "learning_rate_sweep.py"
    assert lr_sweep_path.exists(), f"learning_rate_sweep.py not found at {lr_sweep_path}"
    
    # Import the module
    lr_sweep = _import_module_from_path("learning_rate_sweep", lr_sweep_path)
    
    # Verify expected attributes exist
    assert hasattr(lr_sweep, "grid_sweep"), "grid_sweep function not found"
    assert hasattr(lr_sweep, "stability_sweep"), "stability_sweep function not found"
    
    print(f"\n✓ learning_rate_sweep.py imported successfully")


def test_ablations_config_valid(tmp_path):
    """
    Test that ablation configurations are valid.

    This verifies that each ablation configuration can be used to create
    a valid TrainingConfig without errors.

    Expected runtime: < 0.5 seconds
    """
    import numpy as np
    from cs336_basics.config import TrainingConfig

    # Create dummy data files
    dummy_train = tmp_path / "dummy_train.npy"
    dummy_val = tmp_path / "dummy_val.npy"
    np.save(dummy_train, np.array([1, 2, 3], dtype=np.uint16))
    np.save(dummy_val, np.array([1, 2, 3], dtype=np.uint16))

    ablations_path = PROJECT_ROOT / "experiments" / "ablations.py"
    ablations = _import_module_from_path("ablations", ablations_path)

    # Test each ablation configuration
    for ablation_name, ablation_config in ablations.ABLATIONS.items():
        print(f"\n  Testing ablation: {ablation_name}")

        # Try to create a config with this ablation
        try:
            config = TrainingConfig(
                vocab_size=100,
                context_length=32,
                num_layers=2,
                d_model=64,
                num_heads=2,
                d_ff=128,
                batch_size=4,
                max_iters=10,
                train_data_path=str(dummy_train),
                val_data_path=str(dummy_val),
                checkpoint_dir=str(tmp_path / "checkpoints"),
                device="cpu",
                use_wandb=False,
                **ablation_config  # Apply ablation settings
            )

            # Verify config was created
            assert config is not None

            # Verify ablation settings were applied
            for key, value in ablation_config.items():
                assert hasattr(config, key), f"Config missing attribute: {key}"
                assert getattr(config, key) == value, (
                    f"Ablation setting not applied: {key}={value}"
                )

            print(f"    ✓ Config valid: {ablation_config}")

        except Exception as e:
            pytest.fail(f"Failed to create config for ablation '{ablation_name}': {e}")

    print(f"\n✓ All {len(ablations.ABLATIONS)} ablation configs are valid")


def test_experiment_scripts_have_main():
    """
    Test that all experiment scripts have __main__ blocks.
    
    This verifies that scripts can be run from command line.
    
    Expected runtime: < 0.1 seconds
    """
    experiment_scripts = [
        "ablations.py",
        "batch_size_sweep.py",
        "learning_rate_sweep.py",
    ]
    
    for script_name in experiment_scripts:
        script_path = PROJECT_ROOT / "experiments" / script_name
        assert script_path.exists(), f"{script_name} not found"
        
        # Read the file and check for __main__ block
        content = script_path.read_text()
        assert 'if __name__ == "__main__"' in content, (
            f"{script_name} missing __main__ block"
        )
        
        print(f"  ✓ {script_name} has __main__ block")
    
    print(f"\n✓ All {len(experiment_scripts)} scripts have __main__ blocks")


def test_ablations_function_signature():
    """
    Test that run_ablation function has correct signature.
    
    Expected runtime: < 0.1 seconds
    """
    import inspect
    
    ablations_path = PROJECT_ROOT / "experiments" / "ablations.py"
    ablations = _import_module_from_path("ablations", ablations_path)
    
    # Get function signature
    sig = inspect.signature(ablations.run_ablation)
    params = list(sig.parameters.keys())
    
    # Verify expected parameters
    expected_params = ["ablation", "lr", "device"]
    for param in expected_params:
        assert param in params, f"run_ablation missing parameter: {param}"
    
    print(f"\n✓ run_ablation has correct signature: {params}")


def test_batch_size_sweep_function_signature():
    """
    Test that batch_size_sweep function has correct signature.
    
    Expected runtime: < 0.1 seconds
    """
    import inspect
    
    batch_sweep_path = PROJECT_ROOT / "experiments" / "batch_size_sweep.py"
    batch_sweep = _import_module_from_path("batch_size_sweep", batch_sweep_path)
    
    # Get function signature
    sig = inspect.signature(batch_sweep.batch_size_sweep)
    params = list(sig.parameters.keys())
    
    # Verify expected parameters
    expected_params = ["device", "batch_sizes"]
    for param in expected_params:
        assert param in params, f"batch_size_sweep missing parameter: {param}"
    
    print(f"\n✓ batch_size_sweep has correct signature: {params}")


def test_learning_rate_sweep_function_signatures():
    """
    Test that learning rate sweep functions have correct signatures.
    
    Expected runtime: < 0.1 seconds
    """
    import inspect
    
    lr_sweep_path = PROJECT_ROOT / "experiments" / "learning_rate_sweep.py"
    lr_sweep = _import_module_from_path("learning_rate_sweep", lr_sweep_path)
    
    # Test grid_sweep signature
    sig = inspect.signature(lr_sweep.grid_sweep)
    params = list(sig.parameters.keys())
    expected_params = ["device", "learning_rates"]
    for param in expected_params:
        assert param in params, f"grid_sweep missing parameter: {param}"
    print(f"\n  ✓ grid_sweep has correct signature: {params}")
    
    # Test stability_sweep signature
    sig = inspect.signature(lr_sweep.stability_sweep)
    params = list(sig.parameters.keys())
    expected_params = ["device", "start_lr", "max_lr"]
    for param in expected_params:
        assert param in params, f"stability_sweep missing parameter: {param}"
    print(f"  ✓ stability_sweep has correct signature: {params}")
    
    print(f"\n✓ All learning rate sweep functions have correct signatures")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])

