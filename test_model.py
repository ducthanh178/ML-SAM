"""Simple test script to verify model loading works."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sys
import os
# Add sam_new to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sam_new.sam_jax.models import load_model
import jax
import jax.numpy as jnp

def test_model_loading():
    """Test if we can load a model."""
    print("Testing model loading...")
    
    try:
        # Create a random key
        key = jax.random.PRNGKey(42)
        
        # Load model
        model, params, batch_stats = load_model.get_model(
            model_name='WideResnet_mini',
            batch_size=2,
            image_size=32,
            num_classes=10,
            prng_key=key
        )
        
        print("[OK] Model loaded successfully!")
        print(f"   Model type: {type(model)}")
        print(f"   Params keys: {list(params.keys())[:5]}...")  # Show first 5 keys
        
        # Test forward pass
        dummy_input = jnp.zeros((2, 32, 32, 3), dtype=jnp.float32)
        variables = {'params': params, 'batch_stats': batch_stats}
        output = model.apply(variables, dummy_input, train=False)
        
        print("[OK] Forward pass successful!")
        print(f"   Output shape: {output.shape}")
        print(f"   Output dtype: {output.dtype}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_model_loading()
    sys.exit(0 if success else 1)

