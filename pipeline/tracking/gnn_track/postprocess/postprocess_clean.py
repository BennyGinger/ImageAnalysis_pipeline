from __future__ import annotations
import os
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from tifffile import imwrite, imread
import warnings

from tqdm import trange
warnings.filterwarnings("ignore")
import imageio


class Postprocess():
    def __init__(self,
                 is_3d: bool,
                 seg_paths: Path,
                 preds_dir: Path,
                 decision_threshold: float,
                 merge_operation: str,
                 max_travel_dist: int,
                 directed: bool,
                 ):
        self.is_3d = is_3d
        
        # Load the segmentation files
        self.seg_paths = seg_paths
        self.seg_paths_lst = sorted(list(self.seg_paths.glob('*.tif')))

        # Load the prediction data
        self.preds_dir = preds_dir
        print(f" --> Postprocess the GNN predictions from \033[94m{self.preds_dir}\033[0m")
        self.load_prediction_data()
        
        
        self.max_travel_dist = max_travel_dist
        self.merge_operation = merge_operation
        self.decision_threshold: float = decision_threshold
        self.directed = directed
        self.cols = ["child_id", "parent_id", "start_frame"]
        
        self.connected_edges = self.find_connected_edges()

    def load_prediction_data(self)-> None:
        # Get the file paths
        edge_path = self.preds_dir.joinpath('edge_indexes.pt')
        df_path = self.preds_dir.joinpath('df_feat.csv')
        pred_path = self.preds_dir.joinpath('raw_preds.pt')
        # Load the files
        self.edge_index: torch.Tensor = self._load_tensor(edge_path)
        self.df_feat: pd.DataFrame = self._load_df(df_path)
        self.preds: torch.Tensor = self._load_tensor(pred_path)

    @staticmethod
    def _load_df(file_path: Path)-> pd.DataFrame:
        return pd.read_csv(file_path,index_col=False).reset_index(drop=True)
    
    @staticmethod
    def _load_tensor(file_path: Path)-> torch.Tensor:
        return torch.load(file_path)

    def find_connected_edges(self)-> torch.Tensor:
        """Determines if edges are connected based on the confidence scores from the model and the decision threshold. The function returns the connected edges as a boolean-like (0-1) array."""
        
        if self.directed:
            # Scaling the output to 0-1
            preds_soft = torch.sigmoid(self.preds) 
            return (preds_soft >= self.decision_threshold)
        return self._merge_edges()
    
    def _merge_edges(self)-> torch.Tensor: 
        """Merge the two directions of the edge index based on the confidence scores from the model and the decision threshold. The merge operation can be 'AVG', 'OR', or 'AND', where 'AVG' takes the average of the two directions, 'OR' where at least one directions is above descision threshold, and 'AND' where both directions are above the threshold. The function returns the merged predictions as a boolean-like (0-1) array."""
        
        # Remove the second direction of the edge index
        self.edge_index = self.edge_index.detach().clone()[:, ::2]
        
        # Deep copy the predictions
        self.preds = self.preds.detach().clone()
        
        # Get the confidence scores for the two directions
        clock_pred = self.preds[::2]
        clock_scaled = torch.sigmoid(clock_pred)
        anticlock_pred = self.preds[1::2]
        anticlock_scaled = torch.sigmoid(anticlock_pred)

        ## Connect the edges based on the confidence scores of the two directions
        # Using the average of the two directions
        if self.merge_operation == 'AVG':
            avg_soft = (clock_scaled + anticlock_scaled) / 2.0
            return (avg_soft >= self.decision_threshold)
        
        clock_bool = (clock_scaled >= self.decision_threshold)
        anticlock_bool = (anticlock_scaled >= self.decision_threshold)
        
        # With at least one of the directions is above the threshold
        if self.merge_operation == 'OR':
            return torch.logical_or(clock_bool, anticlock_bool)
        
        # With both directions are above the threshold
        if self.merge_operation == 'AND':
            return torch.logical_and(clock_bool, anticlock_bool)
    
    def create_trajectory(self)-> tuple[np.ndarray, np.ndarray, str]:
        self.flag_id0_terminate = False
        # extract values from arguments
        connected_indices = self.edge_index[:, self.connected_edges]
        
        # Find number of frames for iterations
        frames, mask_count = np.unique(self.df_feat.frame_num, return_counts=True)
        
        # Create the trajectory matrix and set to -2 (=empty cell)
        self.trajectory_matrix = np.zeros((frames.shape[0], mask_count.max())) 
        self.trajectory_matrix[:, :] = -2
        
        # Iterate over the frames to build the trajectory matrix
        parents_df_lst = self._build_trajectory_matrix(connected_indices, list(frames))
        self.trajectory_matrix = self.trajectory_matrix.astype(int)
        
        # Create csv contains all the relevant information for the res_track.txt
        df_parents = pd.concat(parents_df_lst, axis=0).reset_index(drop=True)
        self.df_track, self.trajectory_same_label = self.set_all_info(df_parents)
        
        # Convert csv to res_track.txt
        self.file_str = self._df_to_str(self.df_track)

        return self.trajectory_matrix, self.trajectory_same_label, self.file_str

    def _build_trajectory_matrix(self, connected_indices: torch.Tensor, frames: list[int])-> list[pd.DataFrame]:
        
        df_parents = []
        for frame in frames:
            # Select all the nodes in the current frame
            nodes = self.df_feat[self.df_feat.frame_num==frame].index.values

            # If first frame, fill the matrix and the first frame parents
            if frame == 0:
                self.trajectory_matrix[frame, :nodes.shape[0]] = nodes
                df = self._initialize_parent_df(nodes)
                df_parents.append(df)

            new_tracks = self._find_trajectory_nodes(connected_indices, frame, nodes)
            if new_tracks:
                df = self._find_parent_cell(frame, new_tracks)
                df_parents.append(df)
                
        return df_parents

    def _find_trajectory_nodes(self, connected_indices: torch.Tensor, frame: int, nodes: list[int])-> list[int]:
        new_tracks = []
        for node_idx in nodes:
            # Find the next node to connect
            next_node = self._get_next_node(connected_indices, node_idx)
                        
            # Add the next node to the matrix
            starting_node = self._update_matrix_with_next_node(frame, node_idx, next_node)
            new_tracks.extend(starting_node)
        return new_tracks
    
    def _get_next_node(self, connected_indices: torch.Tensor, node_idx: int)-> int:
        # If there are no connections for the node
        if node_idx not in connected_indices[0, :]:
            # FIXME: I'm not sure this is relevent as we set all cells from frame 0 as starting cells, so I don't see how we can have a situation where a starting cell is not connected
            if node_idx == 0:
                self.flag_id0_terminate = True
            return -1
        
        # Find all potential connections
        connected_idx = np.argwhere(connected_indices[0, :] == node_idx)
        next_frame_idx = connected_indices[1, connected_idx][0]
                
        # Get the euclidean distance between the node and the possible cells to connect
        filtered_distance, distance_mask = self._calc_distance(node_idx, next_frame_idx)
        
        # If there are no cells to connect  
        if filtered_distance.size == 0:
            return -1
        
        # Find the nearest cell to connect
        min_idx = np.argmin(filtered_distance)
        nearest_cell: int = np.where(distance_mask)[0][min_idx]
        next_node_ind = int(next_frame_idx[nearest_cell])
        
        # Delete already assigned nodes from the list to avoid several cells with the same ID per frame         
        assigned_node = connected_indices[1,:] == next_node_ind 
        connected_indices = connected_indices[:,~assigned_node]
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
    
    def _initialize_parent_df(self, starting_cells: list[int])-> pd.DataFrame:
        df_parent = pd.DataFrame(index=range(len(starting_cells)), columns=self.cols)
        df_parent.loc[:, ["start_frame", "parent_id"]] = 0
        df_parent.loc[:, "child_id"] = starting_cells
        return df_parent
    
    def _df_to_str(self, df_track: pd.DataFrame)-> str:
        """
        L B E P where
        L - a unique label of the track (label of markers, 16-bit positive value)
        B - a zero-based temporal index of the frame in which the track begins
        E - a zero-based temporal index of the frame in which the track ends
        P - label of the parent track (0 is used when no parent is defined)
        """
        str_track = ''
        for i in df_track.index:
            L = df_track.loc[i, "child_id"]
            B = df_track.loc[i, "start_frame"]
            E = df_track.loc[i, "end_frame"]
            P = df_track.loc[i, "parent_id"]
            str_track += f"{L} {B} {E} {P}\n"

        return str_track
    
    def _find_parent_cell(self, frame_idx: int, new_tracks: list[int])-> pd.DataFrame: #TODO implement gap frames
        """Function that try to find the parent cell for each cell that started in the frame. Return a dataframe with the parent-child relationship."""
        # Find all the cells that ended in the frame
        end_idx = np.argwhere(self.trajectory_matrix[frame_idx, :] == -1)
        # Find the previous index that terminated
        finish_node_idx: np.ndarray = self.trajectory_matrix[frame_idx - 1, end_idx].squeeze(axis=1)
        # Create the parent dataframe
        df_parent = pd.DataFrame(index=range(len(new_tracks)), columns=self.cols)
        df_parent.loc[:, "start_frame"] = frame_idx
        
        # If there are no cells that ended in the frame, set the parent to 0
        if finish_node_idx.size == 0:
            df_parent.loc[:, "child_id"] = new_tracks
            df_parent.loc[:, "parent_id"] = 0
            return df_parent
        
        # Find the parent cell for each cell that started in the frame
        for idx, cell in enumerate(new_tracks):
            # Get the euclidean distance between the node and the possible cells to connect
            filtered_distance, distance_mask = self._calc_distance(cell, finish_node_idx)
            
            # If there are no cells to connect
            if filtered_distance.size == 0:
                df_parent.loc[idx, "child_id"] = cell
                df_parent.loc[idx, "parent_id"] = 0
                continue
            
            # Find the nearest cell to connect
            min_index = np.argmin(filtered_distance)
            nearest_cell = np.where(distance_mask)[0][min_index]
            parent_cell = int(finish_node_idx[nearest_cell])
            
            df_parent.loc[idx, "child_id"] = cell
            df_parent.loc[idx, "parent_id"] = parent_cell
            # Delete the parent cell from the list to avoid several cells with the same ID per frame
            finish_node_idx = np.delete(finish_node_idx, [nearest_cell])
        return df_parent
    
    
    
    def save_csv(self, df_file, file_name):
        full_name = os.path.join(self.preds_dir, f"postprocess_data")
        os.makedirs(full_name, exist_ok=True)
        full_name = os.path.join(full_name, file_name)
        df_file.to_csv(full_name)

    def save_txt(self, str_txt, output_folder, file_name):
        full_name = os.path.join(output_folder, file_name)
        with open(full_name, "w") as text_file:
            text_file.write(str_txt)

    def clean_repetition(self, df):
        all_childs = df.child_id.values
        unique_vals, count_vals = np.unique(all_childs, return_counts=True)
        prob_vals = unique_vals[count_vals > 1]
        for prob_val in prob_vals:
            masking = df.child_id.values == prob_val
            all_apearence = df.loc[masking, :]
            start_frame = all_apearence.start_frame.min()
            end_frame = all_apearence.end_frame.max()
            df.loc[all_apearence.index[0], ["start_frame", "end_frame"]] = start_frame, end_frame
            df = df.drop(all_apearence.index[1:])

        return df.reset_index(drop=True)

    def set_all_info(self, df_parents: pd.DataFrame)-> tuple[pd.DataFrame, np.ndarray]:

        iterate_childs = df_parents.child_id.values
        frames_traject_same_label = self.trajectory_matrix.copy()
        for ind, child_ind in enumerate(iterate_childs):
            # find the place where we store the child_ind in the trajectory matrix
            # validate that only one place exists
            coordinates_child = np.argwhere(self.trajectory_matrix == child_ind)
            n_places = coordinates_child.shape[0]

            assert n_places == 1, f"Problem! find {n_places} places which the current child appears"

            coordinates_child = coordinates_child.squeeze()
            row, col = coordinates_child
            s_frame = df_parents.loc[ind, "start_frame"]
            assert row == s_frame, f"Problem! start frame {s_frame} is not equal to row {row}"

            # take the specific col from 'row' down
            curr_col = self.trajectory_matrix[row:, col]
            last_ind = np.argwhere(curr_col == -1)
            if last_ind.size != 0:
                last_ind = last_ind[0].squeeze()
                curr_col = curr_col[:last_ind]
            e_frame = row + curr_col.shape[0] - 1

            df_parents.loc[ind, "end_frame"] = int(e_frame)
            curr_id = curr_col[-1]
            df_parents.loc[ind, "child_id"] = curr_id
            frames_traject_same_label[row:e_frame + 1, col] = curr_id

        assert not(df_parents.isnull().values.any()), "Problem! dataframe contains NaN values"
        df_parents = self.clean_repetition(df_parents.astype(int))
        return df_parents.astype(int), frames_traject_same_label
    
    def get_pred(self, idx):
        pred = None
        if len(self.seg_paths_lst):
            im_path = self.seg_paths_lst[idx]
            pred = imread(im_path) #load Image
            if self.is_3d and len(pred.shape) != 3:
                pred = np.stack(imageio.mimread(im_path))
                assert len(pred.shape) == 3, f"Expected 3d dimiension! but {pred.shape}"
        return pred

    def create_save_dir(self):
        self.save_tra_dir = self.seg_paths.parent.joinpath(f"Masks_GNN_Track")
        self.save_tra_dir.mkdir(exist_ok=True)

    def save_new_pred(self, new_pred, idx):
        file_name = self.seg_paths_lst[idx].name
        full_dir = self.save_tra_dir.joinpath(file_name)
        imwrite(full_dir, new_pred.astype(np.uint16))

    def check_ids_consistent(self, frame_ind, pred_ids, curr_ids):

        predID_not_in_currID = [x for x in pred_ids if x not in curr_ids]
        currID_not_in_predID = [x for x in curr_ids if x not in pred_ids]
        flag1 = len(predID_not_in_currID) == 1 and predID_not_in_currID[0] == 0
        flag2 = len(currID_not_in_predID) == 0
        if not flag1:
            str_print = f"Frame {frame_ind}: Find segmented cell {predID_not_in_currID} without assigned labels"
            warnings.warn(str_print)

        assert flag2, f"Frame {frame_ind}: Find assigned labels {currID_not_in_predID} " \
                      f"which are not appears in the final saved results"

        return flag1, predID_not_in_currID

    def fix_inconsistent(self, pred_prob_ids, pred):
        for id in pred_prob_ids:
            if id == 0:
                continue
            pred[pred == id] = 0
        return pred

    def fill_mask_labels(self, debug=False):
        self.create_save_dir()
        all_frames_traject, trajectory_same_label = self.trajectory_matrix, self.trajectory_same_label
        df = self.df_feat
        n_rows, _ = all_frames_traject.shape

        count_diff_vals = 0
        for idx in trange(n_rows):
            pred = self.get_pred(idx)
            pred_copy = pred.copy()
            curr_row = all_frames_traject[idx, :]
            mask_id = np.bitwise_and(curr_row != -1, curr_row != -2) #TODO add -3 for gaps?
            graph_ids = curr_row[mask_id]
            graph_true_ids = trajectory_same_label[idx, mask_id]
            frame_ids = []
            for id, true_id in zip(graph_ids, graph_true_ids):
                flag_id0 = true_id == 0
                if flag_id0:    # edge case when the cell with id=0 terminate after one frame
                    if self.flag_id0_terminate:
                        new_id = trajectory_same_label.max() + 1
                        self.df_track.child_id[self.df_track.child_id == 0] = new_id
                        self.file_str = self._df_to_str(self.df_track)
                    else:
                        assert False, "Problem!"
                val = df.loc[id, "seg_label"]        

                if flag_id0:
                    true_id = new_id
                
                pred_copy[pred==val]=true_id

                frame_ids.append(true_id)
            isOK, predID_not_in_currID = self.check_ids_consistent(idx, np.unique(pred_copy), frame_ids)
            if not debug:
                if not isOK:
                    pred_copy = self.fix_inconsistent(predID_not_in_currID, pred_copy)
                self.save_new_pred(pred_copy, idx)
        print(f"Number of different vals: {count_diff_vals}")
        self.save_txt(self.file_str, self.save_tra_dir, 'res_track.txt')


if __name__== "__main__":
    from time import time
    from pipeline.tracking.gnn_tracking import relabel_masks
    
    preds_dir=Path('/home/Test_images/dia_fish/newtest/c1172-GCaMP-15%_Hypo-1-MaxIP_s1/gnn_files')
    
    
    start = time()
    pp = Postprocess(is_3d=False,
                     seg_paths=Path('/home/Test_images/dia_fish/newtest/c1172-GCaMP-15%_Hypo-1-MaxIP_s1/Masks_Cellpose'),
                     preds_dir=preds_dir,
                     decision_threshold=0.4,
                     merge_operation='AND',
                     max_travel_dist=10,
                     directed=False)
    
    all_frames_traject, trajectory_same_label, str_track = pp.create_trajectory() # Several output available that are also saved in the class, if needed one day
    all_frames_path = preds_dir.joinpath(f'all_frames_traject.csv')
    traj_path = preds_dir.joinpath(f'trajectory_same_label.csv')
    str_track_path = preds_dir.joinpath(f'str_track.csv')
    np.savetxt(all_frames_path, all_frames_traject, delimiter=",")
    np.savetxt(traj_path, trajectory_same_label, delimiter=",")
    with open(str_track_path, 'w') as file:
        file.write(str_track)

    pp.fill_mask_labels(debug=False)
    end = time()
    print(f"Time to postprocess: {round(end-start,ndigits=3)} sec\n")
    metadata = {'finterval':None, 'um_per_pixel':None}
    relabel_masks(76,preds_dir.parent.joinpath('Masks_GNN_Track'),'RFP',metadata,True)




