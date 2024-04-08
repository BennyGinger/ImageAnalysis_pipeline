from image_handeling.Experiment_Classes import Experiment, init_from_dict

def test_analyzed_channel(tmpdir):
    # Create input dictionary for the Experiment class
    input_dict = {'cellpose_seg': {'channel1': 'settings1', 'channel2': 'settings2'}, 
                  'threshold_seg': {},
                  'iou_tracking': {'channel3': 'settings3', 'channel4': 'settings4'},
                  'um_per_pixel': (1, 1), 
                  'interval_sec': 1, 
                  'file_type': 'tif', 
                  'level_0_tag': 'Exp', 
                  'level_1_tag': 'Pos',
                  'img_width':512,
                  'img_length':512,
                  'n_frames': 3, 
                  'full_n_channels':2,
                  'n_slices': 1, 
                  'n_series': 4,
                  'img_path':'path/to/img',
                  'background_sub': ["set_bg"], 
                  'channel_reg': ['green'], 
                  'frame_reg': ["set_frame"],
                  'img_blured':[],
                  'exp_path':tmpdir}

    # Create an instance of the Experiment class
    experiment = init_from_dict(input_dict)

    # Define the expected output
    expected_output = {'cellpose_seg': ['channel1', 'channel2'], 'iou_tracking': ['channel3', 'channel4']}

    # Call the analyzed_channel property and compare the result with the expected output
    assert experiment.analyzed_channels == expected_output