from __future__ import annotations
import torch
from pathlib import Path
from pipeline.utilities.pipeline_utility import PathType
from pipeline.tracking.gnn_track.src.models.celltrack_plmodel import CellTrackLitModel
from pipeline.tracking.gnn_track.modules.graph_dataset_inference import CellTrackGraph
import warnings
warnings.filterwarnings("ignore")

######################## Main function ########################
def predict(ckpt_path: PathType, prediction_dir: Path, max_travel_pix: int, is_3d: bool=False, directed: bool=False)-> None:
    """ Normalize and scale extracted features and cell parameters to create the node features together with the edge indexes (all cell-cell links in the consecutive frames). This graph is then used to make predictions using the trained model."""
    
    # Create graph
    graph = get_graph(prediction_dir, max_travel_pix, is_3d, directed)
    
    # Make prediction
    node_features, edge_index = graph
    outputs = make_prediction(ckpt_path, node_features, edge_index)
    
    # Save results
    edges_path = prediction_dir.joinpath('edge_indexes.pt')
    torch.save(edge_index, edges_path)
    preds_path = prediction_dir.joinpath('raw_preds.pt')
    torch.save(outputs, preds_path)


######################## Helper functions ########################
def load_model(ckpt_path: PathType)-> CellTrackLitModel:
    """Load model from checkpoint.
    """
    # load model from checkpoint model __init__ parameters will be loaded from ckpt automatically you can also pass some parameter explicitly to override it
    trained_model = CellTrackLitModel.load_from_checkpoint(checkpoint_path=ckpt_path)

    # switch to test mode
    trained_model.eval()
    trained_model.freeze()
    return trained_model

def get_graph(save_dir: Path, max_travel_pix: int, is_3d: bool, directed: bool)-> tuple[tuple[torch.FloatTensor,torch.FloatTensor], torch.Tensor]:
    """Create the graph from the data. Normalize and scale the extracted features and cell parameters to create the node features. Also return the edge index (all cell-cell links in the consecutive frames)."""
    
    graph_data: CellTrackGraph = CellTrackGraph(save_dir,max_travel_pix,is_3d,directed)
    return graph_data.create_graph()

def make_prediction(ckpt_path: PathType, node_features: tuple[torch.FloatTensor,torch.FloatTensor], edge_index: torch.Tensor)-> torch.Tensor:
    """Predict the cell tracks using the trained model."""
    
    print(f"   ---> Load model from: \033[94m{ckpt_path}\033[0m")
    trained_model: CellTrackLitModel = load_model(ckpt_path)
    predictions: torch.Tensor = trained_model(node_features, edge_index)
    return predictions


