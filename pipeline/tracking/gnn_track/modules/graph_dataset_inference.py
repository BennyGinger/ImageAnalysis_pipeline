from __future__ import annotations
import os
import os.path as osp
from collections.abc import Iterable

import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
from torch.nn.functional import one_hot
from torch_geometric.data import Data

import warnings
warnings.simplefilter('always')

class CellTrackDataset:
    def __init__(self,
                 num_frames,
                 type_file,
                 dirs_path,
                 main_path,
                 edge_feat_embed_dict,
                 normalize_all_cols,
                 mul_vals=[2, 2, 2],
                 produce_gt='simple',
                 split='train',
                 exp_name='',
                 overlap=1,
                 jump_frames=1,
                 filter_edges=False,
                 directed=True,
                 same_frame=True, #TODO False?
                 next_frame=True,
                 separate_models=False,
                 one_hot_label=True,
                 self_loop=True,
                 normalize=True,
                 debug_visualization=False,
                 which_preprocess='MinMax',
                 drop_feat=[]
                 ):
        # List of int, to scale up the ROI: [2, 2, 2] for 3D and [2, 2] for 2D
        self.mul_vals = mul_vals
        
        # The experiment name must contain the string '3D' or 2D
        self.is_3d = True if '3D' in exp_name else False
        
        # NOTE: Not sure what it does...
        self.filter_edges = filter_edges
        
        # attributes for visualization
        self.debug_visualization = debug_visualization
        # attributes for nodes features
        self.drop_feat = list(drop_feat)
        self.normalize = normalize
        self.which_preprocess = which_preprocess
        self.separate_models = separate_models
        # attributes for edges features
        self.edge_feat_embed_dict = edge_feat_embed_dict
        # attributes for both nodes and edges features
        self.normalize_all_cols = normalize_all_cols
        # attributes for GT construction
        self.produce_gt = produce_gt
        self.one_hot_label = one_hot_label
        # Create a dict with train or test as keys and a list of paths as values to be processed
        # BUG: Trainning only and must accept 2 files!! Change that
        self.dirs_path = dirs_path
        for k, v_list in dirs_path.items():
            for ind, val in enumerate(v_list):
                self.dirs_path[k][ind] = osp.join(main_path, val)
        
        self.save_dir = main_path        
        
        self.modes = ["train", "valid", "test"]
        self.type_file = type_file
        # attributes for graph construction
        self.same_frame = same_frame
        self.next_frame = next_frame
        self.self_loop = self_loop
        self.overlap = overlap
        self.directed = directed
        self.num_frames = num_frames
        # BUG: I don't thing we ever gonna use this
        self.jump_frames = jump_frames
        if self.jump_frames > 1:
            print(f"Pay attention! using {jump_frames} jump_frames can make problem in mitosis edges!")

        # self._process(split)


    def filter_by_roi(self, df_data_curr: pd.DataFrame, df_data_next: pd.DataFrame)-> list[tuple[int,int]]:
        # Columns to consider for ROI calculation
        cents_cols = ["centroid_row", "centroid_col"]
        if self.is_3d:
            cents_cols.append("centroid_depth")
        
        # Extract ROI columns from current and next frame data
        curr_centers, next_centers = df_data_curr.loc[:, cents_cols], df_data_next.loc[:, cents_cols]

        
        index_pairs = []

        for cell_idx in curr_centers.index.values:
            # Extract the centroid coordinates for the current cell
            row_coord, col_coord = curr_centers.centroid_row[cell_idx], curr_centers.centroid_col[cell_idx]
            max_row, min_row = row_coord + self.curr_roi['row'], row_coord - self.curr_roi['row']
            max_col, min_col = col_coord + self.curr_roi['col'], col_coord - self.curr_roi['col']

            # Create masks to find cells within the ROI in the next frame
            row_vals, col_vals = next_centers.centroid_row.values, next_centers.centroid_col.values
            mask_row = np.bitwise_and(min_row <= row_vals, row_vals <= max_row)
            mask_col = np.bitwise_and(min_col <= col_vals, col_vals <= max_col)
            mask_all = np.bitwise_and(mask_row, mask_col)

            if self.is_3d:
                depth_coord = curr_centers.centroid_depth[cell_idx]
                max_depth, min_depth = depth_coord + self.curr_roi['depth'], depth_coord - self.curr_roi['depth']
                depth_vals = next_centers.centroid_depth.values
                mask_depth = np.bitwise_and(min_depth <= depth_vals, depth_vals <= max_depth)
                mask_all = np.bitwise_and(mask_all, mask_depth)

            # Find indices of next frame cells within the ROI
            next_indices = next_centers.index[mask_all].values
            # Pair each next frame index with the current frame index
            curr_indices = np.ones_like(next_indices) * cell_idx
            index_pairs += np.concatenate((curr_indices[:, None], next_indices[:, None]), -1).tolist()
        return index_pairs

    def link_all_edges(self, df_data: pd.DataFrame)-> list[tuple[int,int]]:
        """
        doing aggregation of the same frame links + the links between 2 consecutive frames
        """
        # In the following loop- doing aggregation of the same frame links + the links between 2 consecutive frames
        linked_edges = []
        for frame_ind in np.unique(df_data.frame_num.values)[:-1]:
            # Find all cells in the given frame
            mask_frame = df_data.frame_num.isin([frame_ind])
            
            # doing aggregation of the links between 2 consecutive frames
            # FIXME: We may be able to add the gap links here...
            # Find all cells in the given consecutive frames
            mask_next_frame = df_data.frame_num.isin([frame_ind + 1])
            
            frame_edges = self.filter_by_roi(df_data.loc[mask_frame, :], df_data.loc[mask_next_frame, :])
            
            # FIXME: Might be able to use undirected edges
            if not self.directed:
                # take the opposite direction using [::-1] and merge one-by-one
                # with directed and undirected edges
                opposite_edges = [pairs[::-1] for pairs in frame_edges]
                frame_edges = list(itertools.chain.from_iterable(zip(frame_edges, opposite_edges)))
            
            linked_edges.extend(frame_edges)
        return linked_edges

    def create_gt(self, df_data, curr_frame, next_frame):
        """
        this method create gt for two consecutive frames *only*, it takes the min id and find the

        """
        start_frame_mask = df_data.frame_num.isin([curr_frame])
        next_frame_mask = df_data.frame_num.isin([next_frame])

        start_frame_ids = df_data.id.loc[start_frame_mask].values
        next_frame_ids = df_data.id.loc[next_frame_mask].reset_index().drop(['index'], axis=1)

        num_classes = next_frame_ids.index[-1] + 2  # start with zero (+1) and plus one if is not in the next frame
        next_frame_ids = next_frame_ids.values.squeeze()

        gt_list = []
        for id in start_frame_ids:
            if np.sum(id == next_frame_ids):
                gt_list.append((next_frame_ids == id).astype(int).argmax() + 1)
            else:
                gt_list.append(0)

        y = torch.tensor(gt_list)
        if self.one_hot_label:
            y = one_hot(y, num_classes=num_classes).flatten()
        return y

    def scale_cell_params(self, trimmed_params_df: pd.DataFrame)-> np.ndarray:
        # Convert df to array
        array = trimmed_params_df.values
        
        # Initialize the scaler
        if self.which_preprocess == 'Standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        # Normalize the array. NOTE: Originally 2 different way of normalization was used, but all the model uses separate_models, so I removed the other one.
        return scaler.fit_transform(array)
    
    # FIXME: I think we can adjust the size of the roi to the max_travle dist, instead of this arbitrary value
    def define_bbox_size(self, df_data: pd.DataFrame):
        if self.is_3d:
            cols = ['min_row_bb', 'min_col_bb', 'max_row_bb', 'max_col_bb',
                    'min_depth_bb', 'max_depth_bb']
        else:
            cols = ['min_row_bb', 'min_col_bb', 'max_row_bb', 'max_col_bb']

        bb_feat = df_data.loc[:, cols]
        max_row = np.abs(bb_feat.min_row_bb.values - bb_feat.max_row_bb.values).max()
        max_col = np.abs(bb_feat.min_col_bb.values - bb_feat.max_col_bb.values).max()

        self.curr_roi = {'row': max_row * self.mul_vals[0], 'col': max_col * self.mul_vals[1]}
        if self.is_3d:
            max_depth = np.abs(bb_feat.min_depth_bb.values - bb_feat.max_depth_bb.values).max()
            self.curr_roi['depth'] = max_depth * self.mul_vals[2]

    def move_roi(self, df_data, curr_dir):
        if self.is_3d:
            cols = ['centroid_row', 'centroid_col', 'centroid_depth']
            cols_new = ['diff_row', 'diff_col', 'diff_depth']
        else:
            cols = ['centroid_row', 'centroid_col']
            cols_new = ['diff_row', 'diff_col']

        df_stats = pd.DataFrame(columns=['id'] + cols_new)
        counter = 0
        for id in np.unique(df_data.id):
            mask_id = df_data.id.values == id
            df_id = df_data.loc[mask_id, ['frame_num'] + cols]
            for i in range(df_id.shape[0] - 1):
                curr_frame_ind = df_id.iloc[i, 0]
                next_frame_ind = df_id.iloc[i + 1, 0]

                if curr_frame_ind + 1 != next_frame_ind:
                    continue

                diff = df_id.iloc[i, 1:].values - df_id.iloc[i + 1, 1:].values
                df_stats.loc[counter, 'id'] = id
                df_stats.loc[counter, cols_new] = np.abs(diff)
                counter += 1

        if self.save_stats:
            path = osp.join(curr_dir, "stats")
            os.makedirs(path, exist_ok=True)
            path = osp.join(path, "df_movement_stats.csv")
            df_stats.to_csv(path)

        diff_row = np.abs(df_stats.diff_row.values)
        diff_col = np.abs(df_stats.diff_col.values)
        self.curr_roi = {'row': diff_row.max() + self.mul_vals[0] * diff_row.std(),
                         'col': diff_col.max() + self.mul_vals[1] * diff_col.std()}
        if self.is_3d:
            diff_depth = np.abs(df_stats.diff_depth.values)
            self.curr_roi['depth'] = diff_depth.max() + self.mul_vals[2] * diff_depth.std()

    def create_graph(self):
        """
        curr_dir: str : path to the directory holds CSVs files to build the graph upon
        """
        drop_col_list = []
        
        save_path = Path(self.save_dir).joinpath('all_data_df.csv')
        df_data = pd.read_csv(save_path,index_col=False).reset_index(drop=True)
        
        self.define_bbox_size(df_data)

        ## Create the edges and convert to torch tensor
        link_edges = self.link_all_edges(df_data)
        edge_index = [torch.tensor([lst], dtype=torch.long) for lst in link_edges]
        edge_index = torch.cat(edge_index, dim=0).t().contiguous()

        # Remove the mask label column
        trimmed_df = df_data.drop('seg_label', axis=1)

        # Separate the columns into cell parameters and cell features
        self.separate_cols = np.array(['feat' not in name_col for name_col in trimmed_df.columns])
        
        # Create the node features tensors
        cell_params = torch.FloatTensor(self.scale_cell_params(trimmed_df.loc[:, self.separate_cols]))
        cell_feat = torch.FloatTensor(trimmed_df.loc[:, np.logical_not(self.separate_cols)].values)
        node_features = (cell_params, cell_feat)
        
        return node_features, edge_index

    def _process(self, curr_mode: str):
        # Read data into huge `Data` list and store in dictionary.
        self.all_data = {}
        curr_dir = self.dirs_path[curr_mode]
        print(f"Start process {curr_dir} ({curr_mode})")
        data_list = []
        for dir_path in curr_dir:
            curr_dir = osp.join(dir_path, self.type_file)  # add type of the files for the folder (../{type})
            data_list += self.create_graph(curr_dir, curr_mode)    # concat all dirs graphs
            print(f"Finish process {curr_dir} ({curr_mode})")
        self.all_data[curr_mode] = data_list

