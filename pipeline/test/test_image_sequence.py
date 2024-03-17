
import numpy as np
from tifffile import imwrite
from pre_process.image_sequence import create_img_name_list, expand_dim_tif, extract_image_params, write_tif

def test_create_img_name_list():
    meta_dict = {
        'n_series': 2,
        'n_frames': 3,
        'n_slices': 4,
        'active_channel_list': ['GFP', 'RFP']
    }
    expected_output = [
        'GFP_s01_f0001_z0001', 'GFP_s01_f0001_z0002', 'GFP_s01_f0001_z0003', 'GFP_s01_f0001_z0004',
        'GFP_s01_f0002_z0001', 'GFP_s01_f0002_z0002', 'GFP_s01_f0002_z0003', 'GFP_s01_f0002_z0004',
        'GFP_s01_f0003_z0001', 'GFP_s01_f0003_z0002', 'GFP_s01_f0003_z0003', 'GFP_s01_f0003_z0004',
        'RFP_s01_f0001_z0001', 'RFP_s01_f0001_z0002', 'RFP_s01_f0001_z0003', 'RFP_s01_f0001_z0004',
        'RFP_s01_f0002_z0001', 'RFP_s01_f0002_z0002', 'RFP_s01_f0002_z0003', 'RFP_s01_f0002_z0004',
        'RFP_s01_f0003_z0001', 'RFP_s01_f0003_z0002', 'RFP_s01_f0003_z0003', 'RFP_s01_f0003_z0004',
        'GFP_s02_f0001_z0001', 'GFP_s02_f0001_z0002', 'GFP_s02_f0001_z0003', 'GFP_s02_f0001_z0004',
        'GFP_s02_f0002_z0001', 'GFP_s02_f0002_z0002', 'GFP_s02_f0002_z0003', 'GFP_s02_f0002_z0004',
        'GFP_s02_f0003_z0001', 'GFP_s02_f0003_z0002', 'GFP_s02_f0003_z0003', 'GFP_s02_f0003_z0004',
        'RFP_s02_f0001_z0001', 'RFP_s02_f0001_z0002', 'RFP_s02_f0001_z0003', 'RFP_s02_f0001_z0004',
        'RFP_s02_f0002_z0001', 'RFP_s02_f0002_z0002', 'RFP_s02_f0002_z0003', 'RFP_s02_f0002_z0004',
        'RFP_s02_f0003_z0001', 'RFP_s02_f0003_z0002', 'RFP_s02_f0003_z0003', 'RFP_s02_f0003_z0004'
    ]
    assert sorted(create_img_name_list(meta_dict)) == sorted(expected_output)

def test_expand_dim_tif(tmpdir):
    # Create a temporary 3D numpy array
    img_data = np.random.rand(3, 10, 10)

    # Save the array to a temporary image file
    img_path = tmpdir.join("image.tif")
    imwrite(str(img_path), img_data)

    # Test case 1: All axes are present
    axes = 'CYX'
    expanded_data = expand_dim_tif(img_path, axes)
    
    # Check that the output is a 5D numpy array
    assert isinstance(expanded_data, np.ndarray)
    assert expanded_data.ndim == 5
    assert img_data.shape[0] == expanded_data.shape[2]

    # Test case 2: Missing axes
    axes = 'TYX'
    expanded_data = expand_dim_tif(img_path, axes)
    assert isinstance(expanded_data, np.ndarray)
    assert expanded_data.ndim == 5
    assert img_data.shape[0] == expanded_data.shape[0]

    # Test case 3: Missing multiple axes
    axes = 'ZYX'
    expanded_data = expand_dim_tif(img_path, axes)
    assert isinstance(expanded_data, np.ndarray)
    assert expanded_data.ndim == 5
    assert img_data.shape[0] == expanded_data.shape[1]
    
def test_extract_image_params():
    img_name = 'GFP_s01_f0001_z0001'
    full_channel_list = ['GFP', 'RFP']
    expected_output = (0, 0, 0, 0)
    assert extract_image_params(img_name, full_channel_list) == expected_output

    img_name = 'RFP_s02_f0003_z0004'
    full_channel_list = ['GFP', 'RFP']
    expected_output = (1, 2, 3, 1)
    assert extract_image_params(img_name, full_channel_list) == expected_output

    img_name = 'GFP_s02_f0002_z0003'
    full_channel_list = ['GFP', 'RFP']
    expected_output = (1, 1, 2, 0)
    assert extract_image_params(img_name, full_channel_list) == expected_output
    
