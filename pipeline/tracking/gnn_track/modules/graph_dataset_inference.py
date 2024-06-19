import os
import os.path as osp
from collections.abc import Iterable

import itertools
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

        self._process(split)

    # FIXME: Not used...
    def true_links(self, df_data):
        """
        Doing aggregation of the true links, i.e. which cell are truly connected
        """
        link_edges = []
        # In the following loop- doing aggregation of the true links, i.e. which cell are truly connected
        for id in np.unique(df_data.id.values):
            mask_id = df_data.id.isin([id])  # find the places containing ids
            nodes = df_data.index[mask_id].values
            frames = df_data.frame_num[mask_id].values
            for ind_node in range(0, nodes.shape[0] - 1):
                # until the -2 - since we connect nodes in the graphs,
                # so the last frame cells cant connect to the next frame's cells
                if frames[ind_node] + self.jump_frames == frames[ind_node + 1]:
                    link_edges.append([nodes[ind_node], nodes[ind_node + 1]])
                    if not self.directed:
                        link_edges.append([nodes[ind_node + 1], nodes[ind_node]])

        return link_edges

    def filter_by_roi(self, df_data_curr, df_data_next):
        cols = ["centroid_row", "centroid_col"]
        if self.is_3d:
            cols.append("centroid_depth")
        df_data_curr_ceter, df_data_next_ceter = df_data_curr.loc[:, cols], df_data_next.loc[:, cols]

        curr_list = []

        for ind in df_data_curr_ceter.index.values:
            row_coord, col_coord = df_data_curr_ceter.centroid_row[ind], df_data_curr_ceter.centroid_col[ind]
            max_row, min_row = row_coord + self.curr_roi['row'], row_coord - self.curr_roi['row']
            max_col, min_col = col_coord + self.curr_roi['col'], col_coord - self.curr_roi['col']

            row_vals, col_vals = df_data_next_ceter.centroid_row.values, df_data_next_ceter.centroid_col.values
            mask_row = np.bitwise_and(min_row <= row_vals, row_vals <= max_row)
            mask_col = np.bitwise_and(min_col <= col_vals, col_vals <= max_col)
            mask_all = np.bitwise_and(mask_row, mask_col)

            if self.is_3d:
                depth_coord = df_data_curr_ceter.centroid_depth[ind]
                max_depth, min_depth = depth_coord + self.curr_roi['depth'], depth_coord - self.curr_roi['depth']
                depth_vals = df_data_next_ceter.centroid_depth.values
                mask_depth = np.bitwise_and(min_depth <= depth_vals, depth_vals <= max_depth)
                mask_all = np.bitwise_and(mask_all, mask_depth)

            next_indices = df_data_next_ceter.index[mask_all].values
            curr_indices = np.ones_like(next_indices) * ind
            curr_list += np.concatenate((curr_indices[:, None], next_indices[:, None]), -1).tolist()
        return curr_list

    def same_next_links(self, df_data, link_edges):
        """
        doing aggregation of the same frame links + the links between 2 consecutive frames
        """
        # In the following loop- doing aggregation of the same frame links + the links between 2 consecutive frames
        same_next_edge_index = []
        iter_frames = np.unique(df_data.frame_num.values)
        for loop_ind, frame_ind in enumerate(iter_frames[:-1]):
            # find the places containing the specific frame index
            mask_frame = df_data.frame_num.isin([frame_ind])
            nodes = df_data.index[mask_frame].values.tolist()
            # doing aggregation of the same frame links
            if self.same_frame:
                if self.self_loop:
                    same_next_edge_index += [list(tup) for tup in itertools.product(nodes, nodes)]
                else:
                    same_next_edge_index += [list(tup) for tup in itertools.product(nodes, nodes) if tup[0] != tup[1]]
            # doing aggregation of the links between 2 consecutive frames
            if self.next_frame:
                if frame_ind != iter_frames[-1]:
                    # find the places containing the specific frame index
                    mask_next_frame = df_data.frame_num.isin([iter_frames[loop_ind + 1]])
                    next_nodes = df_data.index[mask_next_frame].values.tolist()
                    if self.filter_edges:
                        curr_list = self.filter_by_roi(df_data.loc[mask_frame, :], df_data.loc[mask_next_frame, :])
                        curr_list = list(filter(lambda x: not (x in link_edges), curr_list))
                    else:
                        curr_list = [list(tup) for tup in itertools.product(nodes, next_nodes)
                                     if not (list(tup) in link_edges)]
                    if not self.directed:
                        # take the opposite direction using [::-1] and merge one-by-one
                        # with directed and undirected edges
                        curr_list_opposite = [pairs[::-1] for pairs in curr_list]
                        curr_list = list(itertools.chain.from_iterable(zip(curr_list, curr_list_opposite)))
                    same_next_edge_index += curr_list
        return same_next_edge_index

    # FIXME: Not used...
    def iterator_gt_creator(self, df_data):
        frames = np.unique(df_data.frame_num)
        gt = []
        for ind in range(frames.shape[0] - 1):
            curr_frame = frames[ind]
            next_frame = frames[ind + 1]
            mask_frames = df_data.frame_num.isin([curr_frame, next_frame])
            gt.append(self.create_gt(df_data[mask_frames], curr_frame, next_frame))
        return torch.cat(gt, axis=0)

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

    def preprocess(self, dropped_df):
        array = dropped_df.values
        if self.normalize:
            array = self.normalize_array(array)
        return array

    def normalize_array(self, array):
        """
        input:
        - array (numpy.ndarray): array should be normalized
        - norm_col (numpy.ndarray): columns should be normalized
        output:
        - array (numpy.ndarray): normalized array
        """
        if self.which_preprocess == 'MinMax':
            scaler = MinMaxScaler()
        elif self.which_preprocess == 'Standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        # array[:, self.normalize_cols] = scaler.fit_transform(array[:, self.normalize_cols])
        if self.separate_models:
            array = scaler.fit_transform(array)
        else:
            array[:, self.normalize_cols] = scaler.fit_transform(array[:, self.normalize_cols])
        return array

    def edge_feat_embedding(self, x, edge_index):
        src, trg = edge_index
        sub_x = x[src] - x[trg]
        abs_sub = np.abs(sub_x)
        res = abs_sub ** 2 if self.edge_feat_embed_dict['p'] == 2 else abs_sub
        # try to preprocess edge features embedding - min-max normalization or z-score normalization ...
        if self.edge_feat_embed_dict['normalized_features']:
            res = self.normalize_array(res)
        return res

    def bb_roi(self, df_data):
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

    def find_roi(self, files, curr_dir):
        temp_data = [pd.read_csv(file) for file in files]
        df_data = pd.concat(temp_data, axis=0).reset_index(drop=True)
        self.bb_roi(df_data)

    def create_graph(self, curr_dir, mode):
        """
        curr_dir: str : path to the directory holds CSVs files to build the graph upon
        """
        data_list = []
        df_list = []
        drop_col_list = []
        is_first_time = True
        # find all the files in the curr_path
        files = [osp.join(curr_dir, f_name) for f_name in sorted(os.listdir(curr_dir)) if
                 self.type_file in f_name]
        print(f"Start with {curr_dir}")
        num_files = len(files)
        self.find_roi(files, curr_dir)

        if self.num_frames == 'all':
            num_frames = num_files
        elif isinstance(self.num_frames, int):
            num_frames = self.num_frames
        else:
            assert False, f"The provided num_frames {type(self.num_frames)} variable type is not supported"

        for ind in range(0, num_files, self.overlap):
            # break when the length of the graph is smaller than the rest number of frames
            if ind + num_frames > num_files:
                break

            # read the current frame CSVs
            temp_data = [pd.read_csv(files[ind_tmp]) for ind_tmp in range(ind, ind + num_frames, self.jump_frames)]
            df_data = pd.concat(temp_data, axis=0).reset_index(drop=True)

            link_edges = []
            if self.same_frame or self.next_frame:
                link_edges += self.same_next_links(df_data, link_edges)

            # convert to torch tensor
            edge_index = [torch.tensor([lst], dtype=torch.long) for lst in link_edges]
            edge_index = torch.cat(edge_index, dim=0).t().contiguous()

            if not ('id' in drop_col_list) and 'id' in df_data.columns:
                drop_col_list.append('id')
                warnings.warn("Find the id label as part of the features and dropped it, please be aware")
            if not ('seg_label' in drop_col_list) and 'seg_label' in df_data.columns:
                drop_col_list.append('seg_label')
                warnings.warn("Find the seg label as part of the features and dropped it, please be aware")

            dropped_df = df_data.drop(drop_col_list, axis=1)
            for feat in self.drop_feat:
                if feat in dropped_df.columns:
                    dropped_df = dropped_df.drop([feat], axis=1)

            if is_first_time:
                is_first_time = False
                print(f"features: {dropped_df.columns}")
                if self.normalize_all_cols:
                    self.normalize_cols = np.ones((dropped_df.shape[-1]), dtype=bool)
                else:
                    self.normalize_cols = np.array(
                        ['feat' != name_col[:len('feat')] for name_col in dropped_df.columns])

                if self.separate_models:
                    self.separate_cols = np.array(['feat' != name_col[:len('feat')] for name_col in dropped_df.columns])


            if not self.separate_models:
                x = self.preprocess(dropped_df)
                if self.edge_feat_embed_dict['use_normalized_x']:
                    edge_feat = self.edge_feat_embedding(x, edge_index)
                else:
                    edge_feat = self.edge_feat_embedding(dropped_df.values, edge_index)
                x = torch.FloatTensor(x)
                edge_feat = torch.FloatTensor(edge_feat)

                if torch.any(x.isnan()) or torch.any(edge_feat.isnan()):
                    assert False, "inputs contain nan values"

                data = Data(x=x, edge_index=edge_index, edge_feat=edge_feat)
            else:
                if not self.edge_feat_embed_dict['use_normalized_x']:
                    x = torch.FloatTensor(self.preprocess(dropped_df.loc[:, self.separate_cols]))
                    x_2 = torch.FloatTensor(dropped_df.loc[:, np.logical_not(self.separate_cols)].values)
                    edge_feat = self.edge_feat_embedding(dropped_df.values, edge_index)
                else:
                    x = self.preprocess(dropped_df.loc[:, self.separate_cols])
                    x_2 = dropped_df.loc[:, np.logical_not(self.separate_cols)].values
                    edge_feat = self.edge_feat_embedding(np.concatenate((x, x_2), axis=-1), edge_index)
                    x = torch.FloatTensor(x)
                    x_2 = torch.FloatTensor(x_2)

                edge_feat = torch.FloatTensor(edge_feat)
                data = Data(x=x, x_2=x_2, edge_index=edge_index,  edge_feat=edge_feat)

            data_list.append(data)
            df_list.append(df_data)
            print(f"Finish frame index {ind}")

        return data_list, df_list

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

