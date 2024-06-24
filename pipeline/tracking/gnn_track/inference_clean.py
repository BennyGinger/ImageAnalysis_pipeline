import os
import yaml
import torch
from pathlib import Path
from pipeline.utilities.pipeline_utility import PathType
from pipeline.tracking.gnn_track.src.models.celltrack_plmodel import CellTrackLitModel
from pipeline.tracking.gnn_track.modules.graph_dataset_inference import CellTrackDataset
import warnings
warnings.filterwarnings("ignore")


def predict(ckpt_path: PathType, save_dir: PathType, frames: int):
    """Inference with trained model.
    It loads trained model from checkpoint.
    Then it creates graph and make prediction.
    """

    config_sets = load_configuration_settings(ckpt_path, save_dir,frames)

    data_train: CellTrackDataset = CellTrackDataset(**config_sets['dataset_params'], split='test')
    data_list, df_list = data_train.all_data['test']
    test_data, df_data = data_list[0], df_list[0]
    # x, x2, edge_index, edge_feature = test_data.x, test_data.x_2, test_data.edge_index, test_data.edge_feat
    x, x2, edge_index, edge_feature = test_data
    
    
    trained_model = load_model(ckpt_path)
    # make prediction
    outputs = trained_model((x, x2), edge_index, edge_feature.float())
    
    
    os.makedirs(save_dir, exist_ok=True)
    file1 = os.path.join(save_dir, 'pytorch_geometric_data.pt')
    file2 = os.path.join(save_dir, 'all_data_df.csv')
    file3 = os.path.join(save_dir, 'raw_output.pt')
    print(f"Save inference files: \n - {file1} \n - {file2} \n - {file3}")
    df_data.to_csv(file2)
    torch.save(test_data, file1)
    torch.save(outputs, file3)

def load_configuration_settings(ckpt_path: PathType, save_dir: PathType, frames: int)-> dict:
    
    print(f"   ---> Load model from: \033[94m{ckpt_path}\033[0m")
    
    parent_path = Path(ckpt_path).parent
    config_path = parent_path.joinpath('.hydra/config.yaml')
    with open(config_path) as file:
        config_settings = yaml.full_load(file)['datamodule']

    config_settings['dataset_params']['num_frames'] = frames 
    config_settings['dataset_params']['main_path'] = save_dir
    config_settings['dataset_params']['dirs_path']['test'] = ['']
    return config_settings

def load_model(ckpt_path: PathType)-> CellTrackLitModel:
    """Load model from checkpoint.
    """
    # load model from checkpoint model __init__ parameters will be loaded from ckpt automatically you can also pass some parameter explicitly to override it
    trained_model = CellTrackLitModel.load_from_checkpoint(checkpoint_path=ckpt_path)

    # switch to evaluation mode
    trained_model.eval()
    trained_model.freeze()
    return trained_model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-mp', type=str, required=True, help='model params full path')

    parser.add_argument('-ns', type=str, required=True, help='number of sequence - string 01/02')
    parser.add_argument('-oc', type=str, required=True, help='output csv directory')

    args = parser.parse_args()

    model_path = args.mp
    num_seq = args.ns
    output_csv = args.oc
    assert num_seq == '01' or num_seq == '02'
    predict(model_path, output_csv, num_seq)
