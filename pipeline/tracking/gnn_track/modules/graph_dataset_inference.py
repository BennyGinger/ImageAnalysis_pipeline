from __future__ import annotations
from dataclasses import field, dataclass
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch


@dataclass
class CellTrackGraph:
    save_dir: Path
    max_travel_pix: int
    is_3d: bool=False
    directed: bool=True
    curr_roi: dict[str,int] = field(init=False)
        
    def filter_by_roi(self, df_curr: pd.DataFrame, df_next: pd.DataFrame)-> list[tuple[int,int]]:
        """Filter the edges between the cells in the consecutive frames based on the ROI. The ROI is defined by the macx cell size and the max_travel_pix parameter. The ROI is used to filter out cells edges that are too far apart."""
        
        # Columns to consider for ROI calculation
        cents_cols = ["centroid_row", "centroid_col"]
        if self.is_3d:
            cents_cols.append("centroid_depth")
        
        # Extract ROI columns from current and next frame data
        curr_centers, next_centers = df_curr.loc[:, cents_cols], df_next.loc[:, cents_cols]

        # Iterate over each cell in the current frame
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
            pairs = list(zip([cell_idx] * len(next_indices), next_indices))
            index_pairs.extend(pairs)
        return index_pairs

    def link_all_edges(self, df: pd.DataFrame)-> list[tuple[int,int]]:
        """Create the edges between the cells in the consecutive frames, meaning determine all potential links between the cells in the consecutive frames. Edges are then filtered based on the ROI (that take in account the cell size and the max_travel_pix parameter)."""
        # In the following loop- doing aggregation of the same frame links + the links between 2 consecutive frames
        linked_edges = []
        for frame_ind in np.unique(df.frame_num.values)[:-1]:
            # Find all cells in the given frame
            mask_frame = df.frame_num.isin([frame_ind])
            
            # doing aggregation of the links between 2 consecutive frames
            # FIXME: We may be able to add the gap links here...
            # Find all cells in the given consecutive frames
            mask_next_frame = df.frame_num.isin([frame_ind + 1])
            
            frame_edges = self.filter_by_roi(df.loc[mask_frame, :], df.loc[mask_next_frame, :])
            
            # FIXME: Might be able to use undirected edges
            if not self.directed:
                # take the opposite direction using [::-1] and merge one-by-one
                # with directed and undirected edges
                opposite_edges = [pairs[::-1] for pairs in frame_edges]
                frame_edges = list(itertools.chain.from_iterable(zip(frame_edges, opposite_edges)))
            
            linked_edges.extend(frame_edges)
        return linked_edges

    def scale_cell_params(self, params_df: pd.DataFrame)-> np.ndarray:
        """Scale the cell parameters between 0-1, using MinMaxScaler."""
        
        # Convert df to array
        array = params_df.values
        
        # Initialize the scaler
        scaler = MinMaxScaler()
        
        # Scale the array
        return scaler.fit_transform(array)
    
    def define_bbox_size(self, df: pd.DataFrame)-> None:
        """Define the bounding box size for the ROI, meaning the largest cell size in the dataset. The size of the bbox is then increased by twice the max_travel_pix parameter to account for cell movement. The ROI is used to filter out cells edges that are too far apart."""
        
        if self.is_3d:
            cols = ['min_row_bb', 'min_col_bb', 'max_row_bb', 'max_col_bb',
                    'min_depth_bb', 'max_depth_bb']
        else:
            cols = ['min_row_bb', 'min_col_bb', 'max_row_bb', 'max_col_bb']

        bb_feat = df.loc[:, cols]
        max_row = np.abs(bb_feat.min_row_bb.values - bb_feat.max_row_bb.values).max()
        max_col = np.abs(bb_feat.min_col_bb.values - bb_feat.max_col_bb.values).max()

        increase_factor = self.max_travel_pix * 2
        self.curr_roi = {'row': max_row + increase_factor, 'col': max_col + increase_factor}
        if self.is_3d:
            max_depth = np.abs(bb_feat.min_depth_bb.values - bb_feat.max_depth_bb.values).max()
            self.curr_roi['depth'] = max_depth * self.max_travel_pix

    def create_graph(self)-> tuple[tuple[torch.FloatTensor,torch.FloatTensor], torch.Tensor]:
        """Create the graph from the data. Normalize and scale the extracted features and cell parameters to create the node features. Also return the edge index (all cell-cell links in the consecutive frames)."""
        
        
        # Load the data from the CSV file
        save_path = Path(self.save_dir).joinpath('all_data_df.csv')
        print(f"   ---> Create graph from: \033[94m{save_path}\033[0m")
        
        df_data = pd.read_csv(save_path,index_col=False).reset_index(drop=True)
        
        # Define the bounding box size
        self.define_bbox_size(df_data)

        # Create the edges and convert to torch tensor
        link_edges = self.link_all_edges(df_data)
        edge_index = [torch.tensor([lst], dtype=torch.long) for lst in link_edges]
        edge_index = torch.cat(edge_index, dim=0).t().contiguous()

        # Remove the mask label column
        trimmed_df = df_data.drop('seg_label', axis=1)

        # Separate the columns into cell parameters and cell features
        separate_cols = np.array(['feat' not in name_col for name_col in trimmed_df.columns])
        
        # Create the node features tensors
        cell_params = torch.FloatTensor(self.scale_cell_params(trimmed_df.loc[:, separate_cols]))
        cell_feat = torch.FloatTensor(trimmed_df.loc[:, np.logical_not(separate_cols)].values)
        node_features = (cell_params, cell_feat)
        
        return node_features, edge_index

