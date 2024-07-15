from __future__ import annotations
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from tifffile import imwrite
import warnings

from tqdm import trange
from pipeline.utilities.data_utility import load_stack

class Postprocess():
    def __init__(self,
                 is_3d: bool,
                 seg_paths: Path,
                 preds_dir: Path,
                 decision_threshold: float,
                 merge_operation: str,
                 max_travel_dist: int,
                 directed: bool,
                 channel_to_track: str
                 ):
        self.is_3d = is_3d
        
        # Load the segmentation files
        self.seg_paths = seg_paths
        self.seg_paths_lst = sorted(list(self.seg_paths.glob('*.tif')))

        self.max_travel_dist = max_travel_dist
        self.merge_operation = merge_operation
        self.decision_threshold: float = decision_threshold
        self.directed = directed
        self.channel = channel_to_track
        
        # Load the prediction data
        self.preds_dir = preds_dir
        print(f" --> Postprocess the GNN predictions from \033[94m{self.preds_dir}\033[0m")
        edge_index = self._load_prediction_data()
        # Remove the second direction of the edge index
        if not self.directed:
            edge_index = edge_index[:, ::2]
        
        # get the connected edges
        connected_mask = self.find_connected_edges()
        self.connected_edges = edge_index[: , connected_mask]
        
    def _load_prediction_data(self)-> torch.Tensor:
        # Get the file paths
        edge_path = self.preds_dir.joinpath('edge_indexes.pt')
        df_path = self.preds_dir.joinpath('df_feat.csv')
        pred_path = self.preds_dir.joinpath('raw_preds.pt')
        
        # Load the files, for tensors, we detach and clone to avoid any in-place operations
        edge_index: torch.Tensor = self._load_tensor(edge_path).detach().clone()
        self.preds: torch.Tensor = self._load_tensor(pred_path).detach().clone()
        self.df_feat: pd.DataFrame = self._load_df(df_path)
        # Scaling predictions between 0-1
        self.preds = torch.sigmoid(self.preds)
        return edge_index
        
    @staticmethod
    def _load_df(file_path: Path)-> pd.DataFrame:
        return pd.read_csv(file_path,index_col=False).reset_index(drop=True)
    
    @staticmethod
    def _load_tensor(file_path: Path)-> torch.Tensor:
        return torch.load(file_path)

    def find_connected_edges(self)-> torch.Tensor:
        """Determines if edges are connected based on the confidence scores from the model and the decision threshold. The function returns the connected edges as a boolean-like (0-1) array."""
        
        if self.directed:
            return (self.preds >= self.decision_threshold)
        return self._merge_edges()
    
    def _merge_edges(self)-> torch.Tensor: 
        """Merge the two directions of the edge index based on the confidence scores from the model and the decision threshold. The merge operation can be 'AVG', 'OR', or 'AND', where 'AVG' takes the average of the two directions, 'OR' where at least one directions is above descision threshold, and 'AND' where both directions are above the threshold. The function returns the merged predictions as a boolean-like (0-1) array."""
        
        # Get the confidence scores for the two directions
        clock_preds = self.preds[::2]
        anticlock_preds = self.preds[1::2]

        # Remove the second direction of the predictions
        self.preds = self.preds[::2]
        
        ## Connect the edges based on the confidence scores of the two directions
        # Using the average of the two directions
        if self.merge_operation == 'AVG':
            avg_soft = (clock_preds + anticlock_preds) / 2.0
            return (avg_soft >= self.decision_threshold)
        
        clock_bool = (clock_preds >= self.decision_threshold)
        anticlock_bool = (anticlock_preds >= self.decision_threshold)
        
        # With at least one of the directions is above the threshold
        if self.merge_operation == 'OR':
            return torch.logical_or(clock_bool, anticlock_bool)
        
        # With both directions are above the threshold
        if self.merge_operation == 'AND':
            return torch.logical_and(clock_bool, anticlock_bool)
    
    def create_trajectory(self)-> tuple[np.ndarray, np.ndarray]:
        
        # Find number of frames for iterations
        frames, mask_count = np.unique(self.df_feat.frame_num, return_counts=True)
        
        # Create the trajectory matrix and set to -2 (=empty cell)
        self.trajectory_matrix = np.full((frames.shape[0], mask_count.max()),-2)
        
        # Iterate over the frames to build the trajectory matrix
        new_track_starting_ids = self._build_trajectory_matrix(list(frames))
        self.trajectory_matrix = self.trajectory_matrix.astype(int)
        
        # Give a unique label to each tracks
        self.finalised_tracks = self._finalize_tracks(new_track_starting_ids)
        
        return self.trajectory_matrix, self.finalised_tracks

    def _build_trajectory_matrix(self, frames: list[int])-> list[int]:
        
        new_track_starting_ids = []
        for frame in frames:
            # Get index of every cells, in given frame
            nodes = self.df_feat[self.df_feat.frame_num==frame].index.values

            # If first frame, fill the matrix with the starting cells
            if frame == 0:
                self.trajectory_matrix[frame, :nodes.shape[0]] = nodes
                new_track_starting_ids.extend(nodes.tolist())
            # If not first frame, find the trajectory nodes and update the new_track list with new tracks
            new_track_starting_ids.extend(self._find_trajectory_nodes(frame, nodes))
        return new_track_starting_ids

    def _find_trajectory_nodes(self, frame: int, nodes: list[int])-> list[int]:
        new_tracks = []
        for node_idx in nodes:
            # Find the next node to connect
            next_node = self._get_next_node(node_idx)
                        
            # Add the next node to the matrix
            starting_node = self._update_matrix_with_next_node(frame, node_idx, next_node)
            new_tracks.extend(starting_node)
        return new_tracks
    
    def _get_next_node(self, node_idx: int)-> int:
        # Find all potential connections
        connected_idx = np.argwhere(self.connected_edges[0, :] == node_idx)
        
        # If there are no connections for the node
        if connected_idx.size == 0:
            return -1
        
        # Get the next frame indices
        next_frame_idx = self.connected_edges[1, connected_idx][0]
        
        # Filter based on max_travel_dist
        filtered_score, distance_mask = self._calc_distance(node_idx, next_frame_idx)
        
        # Retrieve prediction scores for the potential connections
        # prediction_scores = self.preds[connected_idx].squeeze(0)
        # # print(f"{prediction_scores = }")
        # filtered_score = prediction_scores[distance_mask].numpy()
        
        # If there are no cells to connect  
        if filtered_score.size == 0:
            return -1
        
        # Find the nearest cell to connect
        # min_idx = np.argmax(filtered_score)
        min_idx = np.argmin(filtered_score)
        nearest_cell: int = np.where(distance_mask)[0][min_idx]
        next_node_ind = int(next_frame_idx[nearest_cell])
        
        # Delete already assigned nodes from the list to avoid several cells with the same ID per frame         
        assigned_node = self.connected_edges[1,:] == next_node_ind 
        self.connected_edges = self.connected_edges[:,~assigned_node]
        return next_node_ind
    
    def _calc_distance(self, node_idx: int, next_frame_ind: torch.Tensor | np.ndarray)-> tuple[np.ndarray, np.ndarray]:
        
        centroid_cols = ["centroid_depth", "centroid_row", "centroid_col"] if self.is_3d else ["centroid_row", "centroid_col"]
        
        # Extract the centroid positions
        curr_node = self.df_feat.loc[node_idx, centroid_cols].values
        next_frame = self.df_feat.loc[next_frame_ind, centroid_cols].values
        
        # Get the euclidean distance between the node and the possible cells to connect
        distance: np.ndarray = np.sqrt(((next_frame - curr_node) ** 2).sum(axis=-1))
        
        # Filter the distance based on the max_travel_dist
        distance_mask = distance < self.max_travel_dist 
        if not all(distance_mask):
            print(f"Distance: {distance[distance_mask]}")
        return distance[distance_mask], distance_mask
    
    def _update_matrix_with_next_node(self, frame_idx: int, node_idx: int, next_node: int)-> list[int]:
        """Update the trajectory matrix with the next node. If the current node is not connected, find the next node to connect or create a new track.
        Return the index of the cells, as a list, that started a new track."""
        
        # If the current node already connected, update the loacated track with the next node
        if node_idx in self.trajectory_matrix[frame_idx, :]:
            track_idx = np.argwhere(self.trajectory_matrix[frame_idx, :] == node_idx)
            self._assign_next_node(frame_idx, next_node, track_idx)
            return []
        
        # Start a new track, look for the next empty space (-2)
        track_idx = np.argwhere(self.trajectory_matrix[frame_idx, :] == -2)
        # If there's no empty space, add a new column
        if track_idx.size == 0: 
            track_idx = self._expand_matrix(frame_idx)
        # Select the nearest empty space
        track_idx = track_idx.min()
        # Add the current node to the matrix
        self.trajectory_matrix[frame_idx, track_idx] = node_idx
        self._assign_next_node(frame_idx, next_node, track_idx)
        return [node_idx]

    def _expand_matrix(self, frame_idx: int)-> np.ndarray:
        new_col = -2 * np.ones((self.trajectory_matrix.shape[0], 1), dtype=self.trajectory_matrix.dtype)
        self.trajectory_matrix = np.append(self.trajectory_matrix, new_col, axis=1)
        current_idx = np.argwhere(self.trajectory_matrix[frame_idx, :] == -2)
        return current_idx

    def _assign_next_node(self, frame_idx: int, next_node: int, current_idx: int)-> None:
        # Update, only if frame is not the last one
        if frame_idx + 1 < self.trajectory_matrix.shape[0]:
            self.trajectory_matrix[frame_idx + 1, current_idx] = next_node
    
    def _finalize_tracks(self, child_indices: list[int])-> np.ndarray:

        # Add 1 to the matrix and child_indices to avoid 0 as a valid cell (i.e. 0 = background value)
        mask = ~np.isin(self.trajectory_matrix, [-1, -2])
        self.trajectory_matrix[mask] += 1
        child_indices = [child_idx + 1 for child_idx in child_indices] 
        
        finalised_tracks = self.trajectory_matrix.copy()
        
        # Check that there are no cells with the same ID in the same frame
        _, count_vals = np.unique(finalised_tracks, return_counts=True)
        if any(count_vals[2:] > 1):
            print()
            warnings.warn(message="\033[91mThere are cells with the same ID in the same frame\033[0m")
            print()
        
        # Update the matrix with the finalised tracks
        for child_idx in child_indices:
            # Get the coordinates of the child in the trajectory matrix
            start_row, col = np.argwhere(self.trajectory_matrix == child_idx).squeeze()
            
            # Get all indices from the starting track
            col_values = self.trajectory_matrix[start_row:, col]
            
            # Determine the last row of the track and updates the values of the tracks
            end_idx = np.argwhere(col_values == -1)
            if end_idx.size != 0:
                # Get the first index of the end value
                end_idx = end_idx[0].squeeze()
                col_values = col_values[:end_idx]
            last_row = start_row + col_values.shape[0] - 1

            # Assign the end value to the all track
            track_label = col_values[-1]
            finalised_tracks[start_row:last_row + 1, col] = track_label
        
        return finalised_tracks
    
    def save_new_pred(self, new_pred, idx, save_path: Path):
        file_name = self.seg_paths_lst[idx].name
        full_dir = save_path.joinpath(file_name)
        imwrite(full_dir, new_pred.astype(np.uint16))

    def fill_mask_labels(self, save_path: Path):
    
        n_rows, _ = self.trajectory_matrix.shape
        for idx in trange(n_rows):
            pred = load_stack(self.seg_paths_lst, self.channel, idx, return_2D=not self.is_3d)
            pred_copy = pred.copy()
            curr_row = self.trajectory_matrix[idx, :]
            
            # Get all the ids that are not -1 or -2
            mask_id = ~np.isin(curr_row, [-1, -2]) #TODO add -3 for gaps?
            graph_ids = curr_row[mask_id]
            graph_true_ids = self.finalised_tracks[idx, mask_id]
            for id, true_id in zip(graph_ids, graph_true_ids):
                # -1 to account for the 0-based indexing
                val = self.df_feat.loc[id-1, "seg_label"]      
                pred_copy[pred==val]=true_id

            self.save_new_pred(pred_copy, idx, save_path)


if __name__== "__main__":
    from time import time
    from pipeline.tracking.gnn_tracking import relabel_masks
    
    preds_dir=Path('/home/Test_images/dia_fish/newtest/c1172-GCaMP-15%_Hypo-1-MaxIP_s1/gnn_files')
    save_path = preds_dir.parent.joinpath('Masks_GNN_Track')
    
    
    start = time()
    pp = Postprocess(is_3d=False,
                     seg_paths=Path('/home/Test_images/dia_fish/newtest/c1172-GCaMP-15%_Hypo-1-MaxIP_s1/Masks_Cellpose'),
                     preds_dir=preds_dir,
                     decision_threshold=0.4,
                     merge_operation='AND',
                     max_travel_dist=10,
                     directed=False,
                     channel_to_track='RFP')
    
    all_frames_traject, trajectory_same_label = pp.create_trajectory() # Several output available that are also saved in the class, if needed one day
    all_frames_path = preds_dir.joinpath(f'all_frames_traject.csv')
    traj_path = preds_dir.joinpath(f'trajectory_same_label.csv')
    np.savetxt(all_frames_path, all_frames_traject, delimiter=",")
    np.savetxt(traj_path, trajectory_same_label, delimiter=",")

    pp.fill_mask_labels(save_path=save_path)
    end = time()
    print(f"Time to postprocess: {round(end-start,ndigits=3)} sec\n")
    metadata = {'finterval':None, 'um_per_pixel':None}
    relabel_masks(76,preds_dir.parent.joinpath('Masks_GNN_Track'),'RFP',metadata,True)




