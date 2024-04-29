import pytest
import numpy as np
from mask_transformation.complete_track import copy_first_last_mask, find_gaps, fill_gaps
from skimage.draw import rectangle, disk


@pytest.fixture
def mask_stack():
    m1 = np.zeros((10, 10, 10), dtype=np.uint8)
    rr, cc = disk((4, 5), 4)
    m1[:,rr, cc] = 1
    return m1

def test_copy_first_last_mask(mask_stack):
    # Test case 1: Copy first mask to start
    mask = mask_stack.copy()
    mask[:1, :, :] = 0
    expected_output = mask_stack.copy()
    assert np.array_equal(copy_first_last_mask(mask, copy_first_to_start=True), expected_output)

    # Test case 2: Copy last mask to end
    mask = mask_stack.copy()
    mask[-2:, :, :] = 0
    expected_output = mask_stack.copy()
    assert np.array_equal(copy_first_last_mask(mask, copy_last_to_end=True), expected_output)

    # Test case 3: Copy both first and last masks
    mask = mask_stack.copy()
    mask[:1, :, :] = 0
    mask[-2:, :, :] = 0
    expected_output = mask_stack.copy()
    assert np.array_equal(copy_first_last_mask(mask, copy_first_to_start=True, copy_last_to_end=True), expected_output)

    # Test case 4: No copy needed
    mask = mask_stack.copy()
    mask[:1, :, :] = 0
    mask[-2:, :, :] = 0
    expected_output = mask.copy()
    assert np.array_equal(copy_first_last_mask(mask,copy_first_to_start=False,copy_last_to_end=False), expected_output)

    # Test case 5: No copy needed, but copy flags are set
    mask = mask_stack.copy()
    expected_output = mask_stack.copy()
    assert np.array_equal(copy_first_last_mask(mask, copy_first_to_start=True, copy_last_to_end=True), expected_output)

def test_get_masks_to_morph_lst(mask_stack):
    # Test case 1: No gaps between masks
    expected_output = []
    assert sorted(find_gaps(mask_stack)) == sorted(expected_output)

    # Test case 2: Single gap between masks
    mask = mask_stack.copy()
    mask[2:4, :, :] = 0
    expected_output = [(2, 4, 2)]
    assert sorted(find_gaps(mask)) == sorted(expected_output)

    # Test case 3: Multiple gaps between masks
    mask = mask_stack.copy()
    mask[1:3, :, :] = 0
    mask[4:5, :, :] = 0
    expected_output = [(1, 3, 2), (4, 5, 1)]
    assert sorted(find_gaps(mask)) == sorted(expected_output)

    # Test case 4: Missing first mask
    mask = mask_stack.copy()
    mask[:2, :, :] = 0
    expected_output = []
    assert find_gaps(mask) == expected_output

    # Test case 5: Missing last mask
    mask = mask_stack.copy()
    mask[-2:, :, :] = 0
    expected_output = []
    assert find_gaps(mask) == expected_output

    # Test case 6: Missing first mask + gap
    mask = mask_stack.copy()
    mask[:2, :, :] = 0
    mask[4:5, :, :] = 0
    expected_output = [(4, 5, 1)]
    assert find_gaps(mask) == expected_output

    # Test case 7: Missing last mask + gap
    mask = mask_stack.copy()
    mask[-2:, :, :] = 0
    mask[4:5, :, :] = 0
    expected_output = [(4, 5, 1)]
    assert find_gaps(mask) == expected_output

    # Test case 8: Multiple gaps with first and last masks missing
    mask = mask_stack.copy()
    mask[:2, :, :] = 0
    mask[3:4, :, :] = 0
    mask[5:6, :, :] = 0
    mask[-2:, :, :] = 0
    expected_output = [(3, 4, 1), (5, 6, 1)]
    assert find_gaps(mask) == expected_output

    # Test case 9: No gaps, but first and last masks missing
    mask = mask_stack.copy()
    mask[:2, :, :] = 0
    mask[-2:, :, :] = 0
    expected_output = []
    assert find_gaps(mask) == expected_output

def test_fill_gaps(mask_stack):
    # Test case 1: No gaps, copy flags set to True
    expected_output = mask_stack.copy()
    assert np.array_equal(fill_gaps(mask_stack, copy_first_to_start=True, copy_last_to_end=True), expected_output)

    # Test case 2: No gaps, copy flags set to False
    assert np.array_equal(fill_gaps(mask_stack, copy_first_to_start=False, copy_last_to_end=False), expected_output)

    # Test case 3: Single gap, copy flags set to True
    mask = mask_stack.copy()
    mask[2:4, :, :] = 0
    output = fill_gaps(mask, copy_first_to_start=True, copy_last_to_end=True)
    assert np.array_equal(output, expected_output)

    # Test case 4: Single gap, copy flags set to False
    mask = mask_stack.copy()
    mask[1:3, :, :] = 0
    assert np.array_equal(fill_gaps(mask, copy_first_to_start=False, copy_last_to_end=False), expected_output)

    # Test case 5: Multiple gaps, copy flags set to True
    mask = mask_stack.copy()
    mask[1:3, :, :] = 0
    mask[5:6, :, :] = 0
    assert np.array_equal(fill_gaps(mask, copy_first_to_start=True, copy_last_to_end=True), expected_output)

    # Test case 6: Missing first masks, copy flags set to True
    mask = mask_stack.copy()
    mask[:2, :, :] = 0
    assert np.array_equal(fill_gaps(mask, copy_first_to_start=True, copy_last_to_end=True), expected_output)

    # Test case 7: Missing last masks, copy flags set to True
    mask = mask_stack.copy()
    mask[-2:, :, :] = 0
    assert np.array_equal(fill_gaps(mask, copy_first_to_start=True, copy_last_to_end=True), expected_output)

    # Test case 8: Missing first and last masks, copy flags set to True
    mask = mask_stack.copy()
    mask[:2, :, :] = 0
    mask[-2:, :, :] = 0
    assert np.array_equal(fill_gaps(mask, copy_first_to_start=True, copy_last_to_end=True), expected_output)

    # Test case 9: Missing first and last masks, copy flags set to False
    mask = mask_stack.copy()
    mask[:2, :, :] = 0
    mask[-2:, :, :] = 0
    expected_output = mask.copy()
    assert np.array_equal(fill_gaps(mask, copy_first_to_start=False, copy_last_to_end=False), expected_output)

    # Test case 10: Missing first and last masks, copy flags start to True and end to False
    mask = mask_stack.copy()
    mask[:2, :, :] = 0
    mask[-2:, :, :] = 0
    expected_output = mask_stack.copy()
    expected_output[-2:, :, :] = 0
    assert np.array_equal(fill_gaps(mask, copy_first_to_start=True, copy_last_to_end=False), expected_output)

    # Test case 11: Missing first and last masks, copy flags start to False and end to True
    mask = mask_stack.copy()
    mask[:2, :, :] = 0
    mask[-2:, :, :] = 0
    expected_output = mask_stack.copy()
    expected_output[:2, :, :] = 0
    assert np.array_equal(fill_gaps(mask, copy_first_to_start=False, copy_last_to_end=True), expected_output)
    
    # Test case 12: Missing first and last masks + single gap, copy flags set to False
    mask = mask_stack.copy()
    mask[:2, :, :] = 0
    mask[5:6, :, :] = 0
    mask[-2:, :, :] = 0
    expected_output = expected_output = mask_stack.copy()
    expected_output[:2, :, :] = 0
    expected_output[-2:, :, :] = 0
    assert np.array_equal(fill_gaps(mask, copy_first_to_start=False, copy_last_to_end=False), expected_output)
    
