from __future__ import annotations
import torch
from pathlib import Path
from pipeline.utilities.pipeline_utility import PathType
from pipeline.utilities.data_utility import run_multithread
from pipeline.tracking.gnn_track.prediction.models.celltrack_plmodel import CellTrackLitModel
from dataclasses import field, dataclass
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from threading import Lock


######################## Main function ########################
def predict(ckpt_path: PathType, prediction_dir: Path, max_travel_pix: int, is_3d: bool=False, directed: bool=False)-> None:
    """ Construct the graph from the data and make the prediction using the trained model. Save the results in the prediction directory."""
    print(f" --> Predict cell track connections using the trained model: \033[94m{ckpt_path}\033[0m")
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
    graph_data: Graph = Graph(save_dir,max_travel_pix,is_3d,directed)
    return graph_data.create_graph()

def make_prediction(ckpt_path: PathType, node_features: tuple[torch.FloatTensor,torch.FloatTensor], edge_index: torch.Tensor)-> torch.Tensor:
    """Predict the cell tracks using the trained model."""
    
    print(f"  ---> Make prediction")
    trained_model: CellTrackLitModel = load_model(ckpt_path)
    predictions: torch.Tensor = trained_model(node_features, edge_index)
    return predictions

@dataclass
class Graph:
    save_dir: Path
    max_travel_pix: int
    is_3d: bool=False
    directed: bool=False
    curr_roi: dict[str,int] = field(init=False)
        
    def filter_by_distance(self, df_feat: pd.DataFrame, frame_ind: int)-> list[tuple[int,int]]:
        # Columns to consider for ROI calculation
        cents_cols = ["centroid_row", "centroid_col"]
        if self.is_3d:
            cents_cols.append("centroid_depth")
        
        # Get all the indices of the current and next frame
        curr_indices = df_feat.loc[df_feat.frame_num==frame_ind].index.values
        next_indices = df_feat.loc[df_feat.frame_num==frame_ind+1].index.values
        
        # Iterate over each cell in the current frame
        fixed_args = {'df_feat': df_feat, 'cents_cols': cents_cols, 'next_indices': next_indices}
        # Flatten the list of lists, with all the pairs of indices
        lst_of_lst = [self.calculate_distance(cell_idx, **fixed_args) for cell_idx in curr_indices]
        return [pair for lst in lst_of_lst for pair in lst]
        
    def calculate_distance(self, cell_idx: int, df_feat: pd.DataFrame, cents_cols: list[str], next_indices: np.ndarray)-> list[tuple[int,int]]:
        # Extract the centroid coordinates for the current cell
        curr_node = df_feat.loc[cell_idx, cents_cols].values
        # Extract all centroids in the next frame
        next_nodes = df_feat.loc[next_indices, cents_cols].values
        # Get the euclidean distance between the node and the possible cells to connect
        distance: np.ndarray = np.sqrt(((next_nodes - curr_node) ** 2).sum(axis=-1))
        # Filter the distance based on the max_travel_dist
        filtered_indices = next_indices[distance <= self.max_travel_pix]
        # Pair each next frame index with the current frame index
        return [(cell_idx, filtered_idx) for filtered_idx in filtered_indices]
        
    def link_all_edges(self, df_feat: pd.DataFrame)-> list[tuple[int,int]]:
        """Create the edges between the cells in the consecutive frames, meaning determine all potential links between the cells in the consecutive frames. Edges are then filtered based on the ROI (that take in account the cell size and the max_travel_pix parameter)."""
        # Loop through all the frames and link the edges
        fixed_args = {'df_feat': df_feat}
        linked_edges = run_multithread(self._link_edges, np.unique(df_feat.frame_num.values)[:-1], fixed_args)
        
        # Flatten the list of lists
        linked_edges = [item for sublist in linked_edges for item in sublist]
        return linked_edges
 
    def _link_edges(self, frame_ind: int, df_feat: pd.DataFrame, lock: Lock)-> list[tuple[int,int]]:
        with lock:
            frame_edges = self.filter_by_distance(df_feat, frame_ind)
            
        if not self.directed:
            # Add the reversed edges
            reversed_edges = [pairs[::-1] for pairs in frame_edges]
            frame_edges = list(itertools.chain.from_iterable(zip(frame_edges, reversed_edges)))
        return frame_edges

    def scale_cell_params(self, params_df: pd.DataFrame)-> np.ndarray:
        """Scale the cell parameters between 0-1, using MinMaxScaler."""
        
        # Initialize the scaler
        scaler = MinMaxScaler()
        
        # Scale the array
        return scaler.fit_transform(params_df.values)
    
    def create_graph(self)-> tuple[tuple[torch.FloatTensor,torch.FloatTensor], torch.Tensor]:
        """Create the graph from the data. Normalize and scale the extracted features and cell parameters to create the node features. Also return the edge index (all cell-cell links in the consecutive frames)."""
        
        
        # Load the data from the CSV file
        df_feat_path = Path(self.save_dir).joinpath('df_feat.csv')
        print(f"  ---> Create graph from: \033[94m{df_feat_path}\033[0m")
        
        df_feat = pd.read_csv(df_feat_path,index_col=False).reset_index(drop=True)
        
        # Create the edges and convert to torch tensor
        link_edges = self.link_all_edges(df_feat)
        edge_index = [torch.tensor([lst], dtype=torch.long) for lst in link_edges]
        edge_index = torch.cat(edge_index, dim=0).t().contiguous()

        # Remove the mask label column
        trimmed_df = df_feat.drop('seg_label', axis=1)

        # Create a mask to separate the columns into cell parameters and cell features
        separate_cols = np.array(['feat' not in name_col for name_col in trimmed_df.columns])
        
        # Create the node features tensors
        cell_params = torch.FloatTensor(self.scale_cell_params(trimmed_df.loc[:, separate_cols]))
        cell_feat = torch.FloatTensor(trimmed_df.loc[:, np.logical_not(separate_cols)].values)
        node_features = (cell_params, cell_feat)
        
        return node_features, edge_index


