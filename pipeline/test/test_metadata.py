import pytest
from pre_process.metadata import get_tif_meta, get_ND2_meta

@pytest.mark.parametrize("input_tif",
                         ['/home/Test_images/tiff/Run1/c1z25t25v1_tif.tif',
                          '/home/Test_images/tiff/Run2/c2z25t23v1_tif.tif',
                          '/home/Test_images/tiff/Run4/c4z1t91v1.tif'])

def test_get_tif_meta(input_tif):
    expected_output = ['ImageJ','images','channels','slices','frames','hyperstack','mode','unit','finterval','loop','min','max','Info','Labels','Ranges','LUTs','axes','ImageWidth','ImageLength','XResolution','YResolution','n_series']
    
    output_dict = get_tif_meta(input_tif)
    for key in output_dict.keys():
        assert key in expected_output

@pytest.mark.parametrize("input_nd2",
                         ['/home/Test_images/nd2/Run1/c1z25t25v1_nd2.nd2',
                          '/home/Test_images/nd2/Run2/c2z25t23v1_nd2.nd2',
                          '/home/Test_images/nd2/Run3/c3z1t1v3.nd2',
                          '/home/Test_images/nd2/Run4/c4z1t91v1.nd2'])  
def test_get_ND2_meta(input_nd2):
    expected_output = ['rois','height', 'width', 'date', 'fields_of_view','frames','z_levels','z_coordinates','total_images_per_channel','channels','pixel_microns','num_frames','experiment','events','x', 'y', 'c', 't', 'z', 'timesteps', 'v', 'axes']
    output_dict = get_ND2_meta(input_nd2)
    for key in output_dict.keys():
        assert key in expected_output
    
