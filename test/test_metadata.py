import pytest
from pipeline.pre_process.metadata import get_tif_meta, get_ND2_meta, update_channel_names

LIST_TIF = ['/home/Test_images/tiff/Run1/c1z25t25v1_tif.tif',
            '/home/Test_images/tiff/Run2/c2z25t23v1_tif.tif',
            '/home/Test_images/tiff/Run4/c4z1t91v1.tif']

LIST_ND2 = ['/home/Test_images/nd2/Run1/c1z25t25v1_nd2.nd2',
            '/home/Test_images/nd2/Run2/c2z25t23v1_nd2.nd2',
            '/home/Test_images/nd2/Run3/c3z1t1v3.nd2',
            '/home/Test_images/nd2/Run4/c4z1t91v1.nd2']

@pytest.mark.parametrize('input_tif',LIST_TIF)
def test_get_tif_meta(input_tif):
    key_outputs = ['img_width','img_length','n_frames','full_n_channels','n_slices','axes','interval_sec']
    
    output_dict = get_tif_meta(input_tif)
    for key in key_outputs:
        assert key in output_dict

@pytest.mark.parametrize("input_nd2",LIST_ND2)  
def test_get_ND2_meta(input_nd2):
    key_output = ['full_n_channels', 'n_slices', 'n_frames', 'n_series', 'img_width', 'img_length']
    output_dict = get_ND2_meta(input_nd2)
    for key in key_output:
        assert key in output_dict
        
@pytest.mark.parametrize("input,output",
                         [([{'full_n_channels':2},[],[]],[['C1','C2'],['C1','C2']]),
                          ([{'full_n_channels':2},['GFP','RFP'],[]],[['GFP','RFP'],['GFP','RFP']]),
                          ([{'full_n_channels':2},['GFP','RFP'],['GFP','RFP']],[['GFP','RFP'],['GFP','RFP']]),
                          ([{'full_n_channels':2},['GFP','RFP'],['RFP','GFP']],[['GFP','RFP'],['GFP','RFP']]),
                          ([{'full_n_channels':2},['GFP','RFP','DAPI'],[]],[['C1','C2'],['C1','C2']])])
def test_update_channel_names(input,output):
    input_dict, active_channel_list, full_channel_list = input
    output_dict = update_channel_names(input_dict,active_channel_list,full_channel_list)
    out_active, out_full = output
    assert output_dict['active_channel_list'] == out_active
    assert output_dict['full_channel_list'] == out_full