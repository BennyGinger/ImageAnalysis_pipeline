
from os import mkdir
import numpy as np
from tifffile import imread
from ImageAnalysis_pipeline.pipeline.image_extraction.image_sequence import expand_array_dim, get_img_params_lst

    
# def test_write_array(tmpdir):
#     # Create a temporary metadata dictionary
#     metadata = {
#         'exp_path_list': [tmpdir],
#         'um_per_pixel': (0.1,0.1),
#         'interval_sec': 1.0,}
    
#     # Create a temporary folder to save the image
#     mkdir(tmpdir.join("Images"))
    
#     # Create a temporary 3D numpy array
#     array = np.random.randint(1, size=(3, 10, 10))

#     # Create a temporary input data dictionary
#     input_data = {
#         'metadata': metadata,
#         'serie': 0,
#         'img_name': 'image1',
#         'array': array,
#         'array_slice': 1}

#     # Call the write_array function
#     write_array(input_data)

#     # Read the saved image
#     expected_save_path = tmpdir.join("Images","image1.tif")
#     saved_image = imread(str(expected_save_path))
    
#     # Check that the saved image is equal to the input array slice
#     assert np.array_equal(saved_image, array[1])

def test_get_img_params_lst():
    meta_dict = {
        'n_series': 1,
        'n_frames': 3,
        'n_slices': 4,
        'active_channel_list': ['GFP', 'RFP'],
        'full_channel_list': ['GFP', 'RFP']
    }
    expected_output = [{'array_slice': (0, 0, 0, 0), 'serie': 0, 'img_name': 'GFP_s01_f0001_z0001'},
                       {'array_slice': (0, 0, 0, 1), 'serie': 0, 'img_name': 'RFP_s01_f0001_z0001'},
                       {'array_slice': (0, 0, 1, 0), 'serie': 0, 'img_name': 'GFP_s01_f0001_z0002'},
                       {'array_slice': (0, 0, 1, 1), 'serie': 0, 'img_name': 'RFP_s01_f0001_z0002'},
                       {'array_slice': (0, 0, 2, 0), 'serie': 0, 'img_name': 'GFP_s01_f0001_z0003'},
                       {'array_slice': (0, 0, 2, 1), 'serie': 0, 'img_name': 'RFP_s01_f0001_z0003'},
                       {'array_slice': (0, 0, 3, 0), 'serie': 0, 'img_name': 'GFP_s01_f0001_z0004'},
                       {'array_slice': (0, 0, 3, 1), 'serie': 0, 'img_name': 'RFP_s01_f0001_z0004'},
                       {'array_slice': (0, 1, 0, 0), 'serie': 0, 'img_name': 'GFP_s01_f0002_z0001'},
                       {'array_slice': (0, 1, 0, 1), 'serie': 0, 'img_name': 'RFP_s01_f0002_z0001'},
                       {'array_slice': (0, 1, 1, 0), 'serie': 0, 'img_name': 'GFP_s01_f0002_z0002'},
                       {'array_slice': (0, 1, 1, 1), 'serie': 0, 'img_name': 'RFP_s01_f0002_z0002'},
                       {'array_slice': (0, 1, 2, 0), 'serie': 0, 'img_name': 'GFP_s01_f0002_z0003'},
                       {'array_slice': (0, 1, 2, 1), 'serie': 0, 'img_name': 'RFP_s01_f0002_z0003'},
                       {'array_slice': (0, 1, 3, 0), 'serie': 0, 'img_name': 'GFP_s01_f0002_z0004'},
                       {'array_slice': (0, 1, 3, 1), 'serie': 0, 'img_name': 'RFP_s01_f0002_z0004'},
                       {'array_slice': (0, 2, 0, 0), 'serie': 0, 'img_name': 'GFP_s01_f0003_z0001'},
                       {'array_slice': (0, 2, 0, 1), 'serie': 0, 'img_name': 'RFP_s01_f0003_z0001'},
                       {'array_slice': (0, 2, 1, 0), 'serie': 0, 'img_name': 'GFP_s01_f0003_z0002'},
                       {'array_slice': (0, 2, 1, 1), 'serie': 0, 'img_name': 'RFP_s01_f0003_z0002'},
                       {'array_slice': (0, 2, 2, 0), 'serie': 0, 'img_name': 'GFP_s01_f0003_z0003'},
                       {'array_slice': (0, 2, 2, 1), 'serie': 0, 'img_name': 'RFP_s01_f0003_z0003'},
                       {'array_slice': (0, 2, 3, 0), 'serie': 0, 'img_name': 'GFP_s01_f0003_z0004'},
                       {'array_slice': (0, 2, 3, 1), 'serie': 0, 'img_name': 'RFP_s01_f0003_z0004'}]
    
    assert sorted(get_img_params_lst(meta_dict),key=lambda d: d['img_name']) == sorted(expected_output,key=lambda d: d['img_name'])

def test_expand_dim_tif():
    # Test case 1: All axes are present
    # Create a temporary numpy array
    img_data = np.random.rand(2, 2, 4, 3, 10, 10)

    axes = 'PTZCYX'
    expanded_data = expand_array_dim(img_data, axes)
    print(expanded_data.shape)
    # Check that the output is a 5D numpy array
    assert isinstance(expanded_data, np.ndarray)
    assert expanded_data.ndim == 6
    assert img_data.shape == expanded_data.shape

    # Test case 2: Missing axes
    img_data = np.random.rand(2, 10, 10)
    axes = 'TYX'
    expanded_data = expand_array_dim(img_data, axes)
    assert isinstance(expanded_data, np.ndarray)
    assert expanded_data.ndim == 6
    assert expanded_data.shape[0] == expanded_data.shape[2] == 1
    
