#!/usr/bin/env python3
"""
Test script to verify the key fixes to the GNN tracking pipeline.
"""

import sys
import numpy as np
from pathlib import Path
import tempfile
import tifffile

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from imageanalysis.tracking.gnn_tracking import relabel_masks, prepare_manual_correct

def test_relabel_masks_empty_handling():
    """Test that relabel_masks handles empty masks gracefully."""
    print("Testing relabel_masks with empty masks...")
    
    # Create temporary directory with test masks
    with tempfile.TemporaryDirectory() as tmp_dir:
        mask_dir = Path(tmp_dir) / "masks"
        mask_dir.mkdir()
        
        # Create test masks - some empty, some with data
        test_masks = [
            np.zeros((50, 50), dtype=np.uint16),  # Empty mask
            np.ones((50, 50), dtype=np.uint16),   # Full mask  
            np.zeros((50, 50), dtype=np.uint16),  # Another empty mask
        ]
        
        # Save test masks
        for i, mask in enumerate(test_masks):
            tifffile.imwrite(mask_dir / f"mask_{i:03d}.tif", mask)
        
        # Test metadata
        metadata = {'um_per_pixel': (0.5, 0.5), 'finterval': 1}
        
        # This should not crash
        try:
            relabel_masks(frames=3, mask_fold_src=mask_dir, 
                         channel_seg='test', metadata=metadata)
            print("✓ relabel_masks handled empty masks successfully")
            return True
        except Exception as e:
            print(f"✗ relabel_masks failed with error: {e}")
            return False

def test_prepare_manual_correct_empty_handling():
    """Test that prepare_manual_correct handles empty masks gracefully."""
    print("Testing prepare_manual_correct with empty masks...")
    
    # Create temporary directory with test masks
    with tempfile.TemporaryDirectory() as tmp_dir:
        mask_dir = Path(tmp_dir) / "masks"
        mask_dir.mkdir()
        exp_dir = Path(tmp_dir) / "experiment"
        exp_dir.mkdir()
        
        # Create test masks - some empty, some with data
        test_masks = [
            np.zeros((50, 50), dtype=np.uint16),  # Empty mask
            np.ones((50, 50), dtype=np.uint16),   # Full mask  
            np.zeros((50, 50), dtype=np.uint16),  # Another empty mask
        ]
        
        # Save test masks
        for i, mask in enumerate(test_masks):
            tifffile.imwrite(mask_dir / f"mask_{i:03d}.tif", mask)
        
        # This should not crash
        try:
            prepare_manual_correct(frames=3, mask_fold_src=mask_dir, 
                                 channel_seg='test', exp_path=exp_dir)
            print("✓ prepare_manual_correct handled empty masks successfully")
            return True
        except Exception as e:
            print(f"✗ prepare_manual_correct failed with error: {e}")
            return False

def test_imports():
    """Test that all the fixed modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test main GNN tracking module
        from imageanalysis.tracking.gnn_tracking import relabel_masks, prepare_manual_correct
        print("✓ GNN tracking module imported successfully")
        
        # Test feature extraction module
        from imageanalysis.tracking.gnn_track.feature_extraction.feature_extraction import extract_freature_metric_learning
        print("✓ Feature extraction module imported successfully")
        
        # Test prediction module
        from imageanalysis.tracking.gnn_track.prediction.prediction import predict
        print("✓ Prediction module imported successfully")
        
        # Test postprocess module
        from imageanalysis.tracking.gnn_track.postprocess.postprocess_clean import Postprocess
        print("✓ Postprocess module imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed with error: {e}")
        return False

def main():
    """Run all tests."""
    print("Running GNN tracking pipeline fixes tests...\n")
    
    tests = [
        test_imports,
        test_relabel_masks_empty_handling,
        test_prepare_manual_correct_empty_handling,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with unexpected error: {e}")
        print()
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The GNN tracking pipeline fixes are working correctly.")
    else:
        print("❌ Some tests failed. Please review the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
