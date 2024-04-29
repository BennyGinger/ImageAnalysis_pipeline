from image_handeling.Experiment_Classes import Experiment, init_from_dict

def test_analyzed_channel(tmpdir):
    # Create input dictionary for the Experiment class
    input_dict = {'cellpose_seg': {'channel1': 'settings1', 'channel2': 'settings2'}, 
                  'threshold_seg': {},
                  'iou_tracking': {'channel3': 'settings3', 'channel4': 'settings4'},
                  'exp_path':tmpdir}

    # Create an instance of the Experiment class
    experiment = init_from_dict(input_dict)

    # Define the expected output
    expected_output = {'cellpose_seg': ['channel1', 'channel2'], 'iou_tracking': ['channel3', 'channel4']}

    # Call the analyzed_channel property and compare the result with the expected output
    assert experiment.analyzed_channels == expected_output