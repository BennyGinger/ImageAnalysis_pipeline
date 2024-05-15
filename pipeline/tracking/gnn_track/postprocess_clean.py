import os
import os.path as osp
import torch
import pandas as pd
import numpy as np
from skimage import io
import warnings
warnings.filterwarnings("ignore")
import imageio
from pathlib import Path


class Postprocess(object):
    def __init__(self,
                 is_3d,
                 type_masks,
                 merge_operation,
                 decision_threshold,
                 path_inference_output,
                 center_coord,
                 path_seg_result,
                 directed=True
                 ):

        file1 = os.path.join(path_inference_output, 'pytorch_geometric_data.pt')
        file2 = os.path.join(path_inference_output, 'all_data_df.csv')
        file3 = os.path.join(path_inference_output, 'raw_output.pt')

        self.dir_result = dir_results = path_seg_result
        self.results = []
        if os.path.exists(dir_results):
            self.results = [os.path.join(dir_results, fname) for fname in sorted(os.listdir(dir_results)) #path of segmentation masks in correct order
                            if type_masks in fname]

        self.mitosis = True #TODO bring this outside
        self.max_travel_dist = 50 #TODO bring this outside
        self.is_3d = is_3d
        self.center_coord = center_coord
        self.merge_operation = merge_operation
        self.decision_threshold = decision_threshold
        self.directed = directed
        self.path_inference_output = path_inference_output
        self.cols = ["child_id", "parent_id", "start_frame"]

        self.edge_index = (self._load_file(file1)).edge_index

        self.df_preds = self._load_file(file2)
        self.output_pred = self._load_file(file3)
        self.find_connected_edges()

    def _load_file(self, file_path):
        print(f"Load {file_path}")
        file_type = file_path.split('.')[-1]
        if file_type == 'csv':
            file = pd.read_csv(file_path, index_col=0)
        if file_type == 'pt':
            file = torch.load(file_path)
        return file

    def save_csv(self, df_file, file_name):
        full_name = os.path.join(self.path_inference_output, f"postprocess_data")
        os.makedirs(full_name, exist_ok=True)
        full_name = os.path.join(full_name, file_name)
        df_file.to_csv(full_name)

    def save_txt(self, str_txt, output_folder, file_name):
        full_name = os.path.join(output_folder, file_name)
        with open(full_name, "w") as text_file:
            text_file.write(str_txt)

    def insert_in_specific_col(self, all_frames_traject, frame_ind, curr_node, next_node):
        if curr_node in all_frames_traject[frame_ind, :]: #check if the node where we found a connection for has already a track in the earlier frames and connect the next cell to it
            flag = 0 #no start of a new track
            ind_place = np.argwhere(all_frames_traject[frame_ind, :] == curr_node)
            if frame_ind + 1 < all_frames_traject.shape[0]: #if its the last frame, we dont do a connection
                all_frames_traject[frame_ind + 1, ind_place] = next_node
        else: #if the current note was not connected before, we start a new track
            flag = 1 #flag for start of a new track
            ind_place = np.argwhere(all_frames_traject[frame_ind, :] == -2) #check for empty (-2) space to put the track
            while ind_place.size == 0: #if there is not enough space, add a new colum for that track
                new_col = -2 * np.ones((all_frames_traject.shape[0], 1), dtype=all_frames_traject.dtype)
                all_frames_traject = np.append(all_frames_traject, new_col, axis=1)
                ind_place = np.argwhere(all_frames_traject[frame_ind, :] == -2)
            ind_place = ind_place.min()
            all_frames_traject[frame_ind, ind_place] = curr_node
            if frame_ind + 1 < all_frames_traject.shape[0]: #write the connection for the next frame, if not in the last frame
                all_frames_traject[frame_ind + 1, ind_place] = next_node

        return flag, all_frames_traject #return the updated trajectorys and the flag if there was a new start of track or not

    def fill_first_frame(self, cell_starts):
        # cols = ["child_id", "parent_id", "start_frame"]
        df_parent = pd.DataFrame(index=range(len(cell_starts)), columns=self.cols)
        df_parent.loc[:, ["start_frame", "parent_id"]] = 0
        df_parent.loc[:, "child_id"] = cell_starts
        return df_parent

    def find_parent_cell(self, frame_ind, all_frames_traject, df, cell_starts): #TODO implement gap frames
        ind_place = np.argwhere(all_frames_traject[frame_ind, :] == -1) #find all indeces were a track ended
        finish_node_ids = all_frames_traject[frame_ind - 1, ind_place].squeeze(axis=1)# find the start IDs in the frame before
        # print(f"frame_ind: {frame_ind}, cell_starts: {cell_starts}, cell_ends: {finish_node_ids}")

        df_parent = pd.DataFrame(index=range(len(cell_starts)), columns=self.cols)
        df_parent.loc[:, "start_frame"] = frame_ind

        if finish_node_ids.shape[0] != 0:
            if self.is_3d:
                finish_cell = df.loc[finish_node_ids, ["centroid_depth", "centroid_row", "centroid_col"]].values
            else:
                finish_cell = df.loc[finish_node_ids, ["centroid_row", "centroid_col"]].values
            for ind, cell in enumerate(cell_starts):
                if self.is_3d:
                    curr_cell = df.loc[cell, ["centroid_depth", "centroid_row", "centroid_col"]].values
                else:
                    curr_cell = df.loc[cell, ["centroid_row", "centroid_col"]].values

                distance = np.sqrt(((finish_cell - curr_cell) ** 2).sum(axis=-1)) #get the distance from every point
                distance_mask = distance < self.max_travel_dist #check that distances are inside the max_travel distance
                filtered_distance = distance[distance_mask] #apply the filter on the array
                if filtered_distance.size == 0:
                    df_parent.loc[ind, "child_id"] = cell
                    df_parent.loc[ind, "parent_id"] = 0
                    continue
                min_index = np.argmin(filtered_distance) #get the smalest distance index in the filtered array
                nearest_cell = np.where(distance_mask)[0][min_index] #get back the index from the original array

                # nearest_cell = np.argmin(distance, axis=-1)
                parent_cell = int(finish_node_ids[nearest_cell])
                df_parent.loc[ind, "child_id"] = cell
                df_parent.loc[ind, "parent_id"] = parent_cell
                finish_node_ids = np.delete(finish_node_ids, [nearest_cell]) #make sure to not connect another track to this track
        else:
            df_parent.loc[:, "child_id"] = cell_starts
            df_parent.loc[:, "parent_id"] = 0

        return df_parent

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

    def set_all_info(self, df_parents_all, all_frames_traject):

        iterate_childs = df_parents_all.child_id.values
        frames_traject_same_label = all_frames_traject.copy()
        for ind, child_ind in enumerate(iterate_childs):
            # find the place where we store the child_ind in the trajectory matrix
            # validate that only one place exists
            coordinates_child = np.argwhere(all_frames_traject == child_ind)
            n_places = coordinates_child.shape[0]

            assert n_places == 1, f"Problem! find {n_places} places which the current child appears"

            coordinates_child = coordinates_child.squeeze()
            row, col = coordinates_child
            s_frame = df_parents_all.loc[ind, "start_frame"]
            assert row == s_frame, f"Problem! start frame {s_frame} is not equal to row {row}"

            # take the specific col from 'row' down
            curr_col = all_frames_traject[row:, col]
            last_ind = np.argwhere(curr_col == -1)
            if last_ind.size != 0:
                last_ind = last_ind[0].squeeze()
                curr_col = curr_col[:last_ind]
            e_frame = row + curr_col.shape[0] - 1

            df_parents_all.loc[ind, "end_frame"] = int(e_frame)
            curr_id = curr_col[-1]
            df_parents_all.loc[ind, "child_id"] = curr_id
            frames_traject_same_label[row:e_frame + 1, col] = curr_id

        assert not(df_parents_all.isnull().values.any()), "Problem! dataframe contains NaN values"
        df_parents_all = self.clean_repetition(df_parents_all.astype(int))
        return df_parents_all.astype(int), frames_traject_same_label

    def df2str(self, df_track):
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

    def merge_edges(self):
        in_output_pred, out_output_pred = self.match_edges()
        if self.merge_operation == 'OR' or self.merge_operation == 'AND':
            in_outputs_soft = torch.sigmoid(in_output_pred)
            in_outputs_hard = (in_outputs_soft > self.decision_threshold).int()

            out_outputs_soft = torch.sigmoid(out_output_pred)
            out_outputs_hard = (out_outputs_soft > self.decision_threshold).int()

            final_outputs_hard = np.bitwise_or(in_outputs_hard,out_outputs_hard) if self.merge_operation == 'OR'\
                else np.bitwise_and(in_outputs_hard,out_outputs_hard)

        if self.merge_operation == 'AVG':
            avg_outputs_soft = torch.sigmoid(in_output_pred) + torch.sigmoid(out_output_pred)
            avg_outputs_soft = avg_outputs_soft / 2.0
            final_outputs_hard = (avg_outputs_soft > self.decision_threshold).int()

        self.outputs_hard = final_outputs_hard
        return final_outputs_hard

    def merge_match_edges(self, edge_index, output_pred): 
        #function looks for both ways of the tensor to see if connections where established and 
        # checks for conficence in comparison to the decission threshold
        #AND, OR, AVG: how to bring the two confidence lists together
        assert torch.all(edge_index[:, ::2] == edge_index[[1, 0], 1::2]), \
            "The results don't match!"
        edge_index = edge_index[:, ::2]
        in_output_pred = output_pred[::2]
        out_output_pred = output_pred[1::2]

        if self.merge_operation == 'OR' or self.merge_operation == 'AND':
            in_outputs_soft = torch.sigmoid(in_output_pred)
            in_outputs_hard = (in_outputs_soft > self.decision_threshold).int()

            out_outputs_soft = torch.sigmoid(out_output_pred)
            out_outputs_hard = (out_outputs_soft > self.decision_threshold).int()

            final_outputs_hard = np.bitwise_or(in_outputs_hard, out_outputs_hard) if self.merge_operation == 'OR' \
                else np.bitwise_and(in_outputs_hard, out_outputs_hard)

        elif self.merge_operation == 'AVG':
            avg_outputs_soft = torch.sigmoid(in_output_pred) + torch.sigmoid(out_output_pred)
            avg_outputs_soft = avg_outputs_soft / 2.0
            final_outputs_hard = (avg_outputs_soft > self.decision_threshold).int()

        return final_outputs_hard, edge_index

    def find_connected_edges(self): #outputs are propably the confidence scores of the model, whether two cells are connected
        edge_index, outputs = self.edge_index, self.output_pred

        if not self.directed:
            final_outputs_hard, edge_index = self.merge_match_edges(edge_index.detach().clone(), outputs.detach().clone())
            self.outputs_hard = final_outputs_hard
            self.edge_index = edge_index
        else:
            outputs_soft = torch.sigmoid(outputs) #created values between 0 and 1 for the confidence
            self.outputs_hard = (outputs_soft > self.decision_threshold).int() #when confidence from the model is superior to the decision threshold, connection is accepted

    def create_trajectory(self):
        edge_index, df, outputs_hard = self.edge_index, self.df_preds, self.outputs_hard
        self.flag_id0_terminate = False
        # extract values from arguments
        connected_indices = edge_index[:, outputs_hard.bool()]

        # find number of frames for iterations
        frame_nums, counts = np.unique(df.frame_num, return_counts=True)
        all_frames_traject = np.zeros((frame_nums.shape[0], counts.max())) #crearing matrix with shape (rows=frames, column=max num of label in frame)

        # intialize the matrix with -2 meaning empty cell, -1 means end of trajectory,
        # other value means the number of node in the graph
        all_frames_traject[:, :] = -2
        str_track = ''
        df_parents = []
        for frame_ind in frame_nums: #loop through the frames
            # mask_frame_ind = df.frame_num.isin([frame_ind])  # find the places containing frame_ind
            nodes_indices = df[df.frame_num==frame_ind].index.values # find the places containing frame_ind, nodes_indices: unique value for every node
            # filter the places with the specific frame_ind and take the corresponding indices
            # nodes = df.loc[mask_frame_ind, :]
            # nodes_indices = nodes.index.values #nodes_indices: unique value for every node

            next_frame_indices = np.array([])

            if frame_ind == 0:  # for the first frame, we should fill the first row with node indices
                all_frames_traject[frame_ind, :nodes_indices.shape[0]] = nodes_indices
                df_parents.append(self.fill_first_frame(nodes_indices))

            num_starts = 0
            cell_starts = []
            for i in nodes_indices: #loop through the nodes in one frame
                if i in connected_indices[0, :]:
                    ind_place = np.argwhere(connected_indices[0, :] == i) # find all potential connections
                    
                    next_frame_ind = connected_indices[1, ind_place][0]#.numpy().squeeze() #get the ID of the potential cells in the next frame
                    if self.is_3d:
                        next_frame = df.loc[next_frame_ind, ["centroid_depth", "centroid_row", "centroid_col"]].values
                        curr_node = df.loc[i, ["centroid_depth", "centroid_row", "centroid_col"]].values
                    else:
                        next_frame = df.loc[next_frame_ind, ["centroid_row", "centroid_col"]].values #getting the centroid position for the potential connection points
                        curr_node = df.loc[i, ["centroid_row", "centroid_col"]].values #getting the original centroid
                    
                    distance = np.sqrt(((next_frame - curr_node) ** 2).sum(axis=-1)) #get the euclidean distance between the node and the possible cells to connect
                    distance_mask = distance < self.max_travel_dist #check that distances are inside the max_travel distance
                    filtered_distance = distance[distance_mask] #apply the filter on the array
                    if filtered_distance.size == 0:
                        next_node_ind = -1
                    else:
                        min_index = np.argmin(filtered_distance) #get the smalest distance index in the filtered array
                        nearest_cell = np.where(distance_mask)[0][min_index] #get back the index from the original array
                        next_node_ind = int(next_frame_ind[nearest_cell])
                        
                        
                    # #check how many potential connections one cell has
                    # if ind_place.shape[-1] > 1: # if more than one connection is possible:
                    #     next_frame_ind = connected_indices[1, ind_place].numpy().squeeze() #get the ID of the potential cells in the next frame
                    #     if self.is_3d:
                    #         next_frame = df.loc[next_frame_ind, ["centroid_depth", "centroid_row", "centroid_col"]].values
                    #         curr_node = df.loc[i, ["centroid_depth", "centroid_row", "centroid_col"]].values
                    #     else:
                    #         next_frame = df.loc[next_frame_ind, ["centroid_row", "centroid_col"]].values #getting the centroid position for the potential connection points
                    #         curr_node = df.loc[i, ["centroid_row", "centroid_col"]].values #getting the original centroid

                    #     distance = np.sqrt(((next_frame - curr_node) ** 2).sum(axis=-1)) #get the euclidean distance between the node and the possible cells to connect
                    #     nearest_cell = np.argmin(distance, axis=-1) #get the index of the closest cell
                    #     # add to the array
                    #     next_node_ind = next_frame_ind[nearest_cell]

                    # elif ind_place.shape[-1] == 1:  # one node in the next frame is connected to current node
                    #     next_node_ind = connected_indices[1, ind_place[0]]
                    # else:  # no node in the next frame is connected to current node -
                    #     # in this case we end the trajectory (-1 means stop of track)
                    #     next_node_ind = -1
                    if not self.mitosis and not next_node_ind==-1:  
                        condition = connected_indices[1,:] == next_node_ind #delete already assigned nodes from the list to avoid several cells with the same ID per frame
                        connected_indices = connected_indices[:,~condition] 
                else:
                    # we dont find the current node in the edge indices matrix - meaning we dont have a connection
                    # for the node - in this case we end the trajectory and the cell
                    if i == 0:
                        self.flag_id0_terminate = True #only needed if the cell with node 0 terminates after first frame. To make sure to give it a proper ID later
                    next_node_ind = -1

                        
                next_frame_indices = np.append(next_frame_indices, next_node_ind) #add the next node index or -1 for track stop into the next_frame_indices list
                # count the number of starting trajectories
                start, all_frames_traject = self.insert_in_specific_col(all_frames_traject, frame_ind, i, next_node_ind)
                num_starts += start

                if start == 1:  # append the id of the cell to the list
                    cell_starts.append(i)
            num_starts = 0
            if num_starts > 0:
                df_parents.append(self.find_parent_cell(frame_ind, all_frames_traject, df, cell_starts)) #TODO check this function

        all_frames_traject = all_frames_traject.astype(int)

        # create csv contains all the relevant information for the res_track.txt
        df_parents_all = pd.concat(df_parents, axis=0).reset_index(drop=True)
        df_track_res, trajectory_same_label = self.set_all_info(df_parents_all, all_frames_traject)

        # convert csv to res_track.txt and res_track_real.txt
        str_track = self.df2str(df_track_res)

        self.all_frames_traject = all_frames_traject
        self.trajectory_same_label = trajectory_same_label
        self.df_track = df_track_res
        self.file_str = str_track
        
        return all_frames_traject, trajectory_same_label, str_track

    def get_pred(self, idx):
        pred = None
        if len(self.results):
            im_path = self.results[idx]
            pred = io.imread(im_path) #load Image
            if self.is_3d and len(pred.shape) != 3:
                pred = np.stack(imageio.mimread(im_path))
                assert len(pred.shape) == 3, f"Expected 3d dimiension! but {pred.shape}"
        return pred

    def create_save_dir(self):
        # num_seq = self.dir_result.split('/')[-1][:2]
        save_tra_dir = osp.join(self.dir_result, f"../Masks_GNN_Track") 
        self.save_tra_dir =save_tra_dir
        os.makedirs(self.save_tra_dir, exist_ok=True)

    def save_new_pred(self, new_pred, idx):
        # idx_str = "%04d" % idx
        # file_name = f"mask{idx_str}.tif"
        file_name = osp.basename(self.results[idx])
        full_dir = osp.join(self.save_tra_dir, file_name)
        io.imsave(full_dir, new_pred.astype(np.uint16))

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
        all_frames_traject, trajectory_same_label = self.all_frames_traject, self.trajectory_same_label
        df = self.df_preds
        n_rows, _ = all_frames_traject.shape

        count_diff_vals = 0
        for idx in range(n_rows):
            pred = self.get_pred(idx)
            pred_copy = pred.copy()
            curr_row = all_frames_traject[idx, :]
            mask_id = np.bitwise_and(curr_row != -1, curr_row != -2) #TODO add -3 for gaps?
            graph_ids = curr_row[mask_id]
            graph_true_ids = trajectory_same_label[idx, mask_id]
            # mask_where = np.ones_like(pred) #TODO why np_ones? Are we not losing mask 1 than?
            frame_ids = []
            for id, true_id in zip(graph_ids, graph_true_ids):
                flag_id0 = true_id == 0
                if flag_id0:    # edge case when the cell with id=0 terminate after one frame
                    if self.flag_id0_terminate:
                        new_id = trajectory_same_label.max() + 1
                        self.df_track.child_id[self.df_track.child_id == 0] = new_id
                        self.file_str = self.df2str(self.df_track)
                    else:
                        assert False, "Problem!"
                if self.is_3d:
                    cell_center = df.loc[id, ["centroid_depth", "centroid_row", "centroid_col"]].values.astype(int)
                    depth_center, row_center, col_center = cell_center[0], cell_center[1], cell_center[2]
                    if self.center_coord:
                        n_depth_img, n_row_img, n_col_img = pred.shape
                        depth_center += n_depth_img // 2
                        row_center += n_row_img // 2
                        col_center += n_col_img // 2

                    val = pred[depth_center, row_center, col_center]
                    if 'seg_label' in df.columns:
                        v_old = val
                        val = df.loc[id, "seg_label"]
                        count_diff_vals += 1 if v_old != val else 0

                    if val == 0:
                        if np.any(pred[depth_center-3:depth_center+3, row_center-3:row_center+3, col_center-3:col_center+3] != 0):
                            area = pred[depth_center-3:depth_center+3, row_center-3:row_center+3, col_center-3:col_center+3]
                            unique_labels, counts = np.unique(area, return_counts=True)
                            mask = unique_labels != 0
                            unique_labels = unique_labels[mask]
                            counts = counts[mask]
                            val = unique_labels[np.argmax(counts)]
                        else:
                            print("Problem! The provided center coordinates value is zero, should be labeled with other value")
                            print(df.loc[id, ["seg_label", "frame_num",  "centroid_depth", "centroid_row", "centroid_col", "min_depth_bb",
                                              "min_row_bb", "min_col_bb", "max_depth_bb", "max_row_bb", "max_col_bb"]].astype(int))
                            continue
                else:
                    cell_center = df.loc[id, ["centroid_row", "centroid_col"]].values.astype(int)
                    row_center, col_center = cell_center[0], cell_center[1] #centroid positions of the labeled cell
                    if self.center_coord:
                        n_row_img, n_col_img = pred.shape #pred: loaded mask
                        row_center += n_row_img // 2 #BUG stupid thing
                        col_center += n_col_img // 2

                    val = pred[row_center, col_center] #get the ID of the cell in the original Mask
                    if 'seg_label' in df.columns:
                        v_old = val
                        val = df.loc[id, "seg_label"]
                        count_diff_vals += 1 if v_old != val else 0

                    if val == 0: #only if the Cell ID on the position == 0
                        if np.any(pred[row_center-3:row_center+3, col_center-3:col_center+3] != 0): # try the IDs in an area and find the most common ID
                            area = pred[row_center-3:row_center+3, col_center-3:col_center+3]
                            unique_labels, counts = np.unique(area, return_counts=True)
                            mask = unique_labels != 0
                            unique_labels = unique_labels[mask]
                            counts = counts[mask]
                            val = unique_labels[np.argmax(counts)]
                        else:
                            print("Problem! The provided center coordinates value is zero, should be labeled with other value") #print error if there is no ID
                            print(df.loc[id, ["seg_label", "frame_num", "centroid_row", "centroid_col",
                                              "min_row_bb", "min_col_bb", "max_row_bb", "max_col_bb"]].astype(int))
                            continue

                assert val != 0, "Problem! The provided center coordinates value is zero, " \
                                 "should be labeled with other value"
                if flag_id0:
                    true_id = new_id
                # mask_val = (pred_copy == val).copy() #get a mask with only the obj == val
                # mask_curr = np.logical_and(mask_val, mask_where)
                # pred_copy[mask_curr] = true_id
                # mask_where = np.logical_and(np.logical_not(mask_val), mask_where)
                
                pred_copy[pred==val]=true_id

                frame_ids.append(true_id)
            print(f'processing frame: {idx+1}')
            isOK, predID_not_in_currID = self.check_ids_consistent(idx, np.unique(pred_copy), frame_ids)
            if not debug:
                if not isOK:
                    pred_copy = self.fix_inconsistent(predID_not_in_currID, pred_copy)
                self.save_new_pred(pred_copy, idx)
        print(f"Number of different vals: {count_diff_vals}")
        self.save_txt(self.file_str, self.save_tra_dir, 'res_track.txt')



if __name__== "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-modality', type=str, required=True, help='2D/3D modality')
    parser.add_argument('-iseg', type=str, required=True, help='segmentation output directory')
    parser.add_argument('-oi', type=str, required=True, help='inference output directory')

    args = parser.parse_args()

    modality = args.modality
    assert modality == '2D' or modality == '3D'

    path_inference_output = args.oi
    path_Seg_result = args.iseg

    is_3d = '3d' in modality.lower()
    directed = True
    merge_operation = 'AND'

    pp = Postprocess(is_3d=is_3d,
                     type_masks='tif', merge_operation=merge_operation,
                     decision_threshold=0.5,
                     path_inference_output=path_inference_output, center_coord=False,
                     directed=directed,
                     path_seg_result=path_Seg_result)
    all_frames_traject, trajectory_same_label, str_track = pp.create_trajectory()
    pp.fill_mask_labels(debug=False)








