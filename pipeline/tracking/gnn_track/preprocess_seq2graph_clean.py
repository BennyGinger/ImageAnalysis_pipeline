from __future__ import annotations
from dataclasses import dataclass
from os import PathLike
import os
import os.path as op
from tifffile import imread
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
from skimage import io
from skimage.measure import regionprops, regionprops_table
import warnings

from tqdm import trange
warnings.filterwarnings("always")
from pathlib import Path
import re

from pipeline.tracking.gnn_track.modules.resnet_2d.resnet import set_model_architecture, MLP #src_metric_learning.modules.resnet_2d.resnet
from pipeline.utilities.data_utility import run_multithread, run_multiprocess
from skimage.morphology import label



def create_csv(input_images, input_seg, input_model, output_csv, channel:str):
    # Load images
    if not Path(input_images).exists():
        raise FileNotFoundError(f"Input images folder does not exist: {input_images}")
    
    img_lst = sorted([str(file) for file in Path(input_images).glob(f"*{channel}*.tif")])
    
    if not img_lst:
        raise FileNotFoundError(f"Input images folder does not contain any images with the channel: {channel}")
    
    # Load segmentation masks
    if not Path(input_seg).exists():
        raise FileNotFoundError(f"Input segmentation folder does not exist: {input_seg}")
    
    seg_lst = sorted([str(file) for file in Path(input_seg).glob(f"*{channel}*.tif")])
    
    if not seg_lst:
        raise FileNotFoundError(f"Input segmentation folder does not contain any images with the channel: {channel}")
    
    # Create dataset
    ds = TestDataset(images=img_lst, results=seg_lst)
    
    ds.preprocess_features_loop_by_results_w_metric_learning(path_to_write=output_csv, dict_path=input_model)



@dataclass
class TestDataset():
    """Example dataset class for loading images from folder."""

    images: list[PathLike]
    results: list[PathLike]

    def __getitem__(self, idx):
        
        im_path = self.images[idx]
        image = imread(im_path)
        result_path = self.results[idx]
        result = imread(result_path)
        return image, result, im_path, result_path

    def __len__(self):
        return len(self.images)

    def padding(self, img):
        if self.flag_new_roi:
            desired_size_row = self.global_delta_row
            desired_size_col = self.global_delta_col
        else:
            desired_size_row = self.roi_model['row']
            desired_size_col = self.roi_model['col']
        delta_row = desired_size_row - img.shape[0]
        delta_col = desired_size_col - img.shape[1]
        pad_top = delta_row // 2
        pad_left = delta_col // 2

        image = cv2.copyMakeBorder(img, pad_top, delta_row - pad_top, pad_left, delta_col - pad_left,
                                   cv2.BORDER_CONSTANT, value=self.pad_value)

        if self.flag_new_roi:
            image = cv2.resize(image, dsize=(self.roi_model['col'], self.roi_model['row']))

        return image

    def extract_freature_metric_learning(self, bbox, img, seg_mask, ind, normalize_type='MinMaxCell'):
        min_row_bb, min_col_bb, max_row_bb, max_col_bb = bbox
        img_patch = img[min_row_bb:max_row_bb, min_col_bb:max_col_bb]
        msk_patch = seg_mask[min_row_bb:max_row_bb, min_col_bb:max_col_bb] != ind
        img_patch[msk_patch] = self.pad_value
        img_patch = img_patch.astype(np.float32)

        if normalize_type == 'regular':
            img = self.padding(img_patch) / self.max_img
        elif normalize_type == 'MinMaxCell':
            not_msk_patch = np.logical_not(msk_patch)
            img_patch[not_msk_patch] = (img_patch[not_msk_patch] - self.min_cell) / (self.max_cell - self.min_cell)
            img = self.padding(img_patch)
        else:
            assert False, "Not supported this type of normalization"

        img = torch.from_numpy(img).float()
        with torch.no_grad():
            embedded_img = self.embedder(self.trunk(img[None, None, ...]))

        return embedded_img.numpy().squeeze()

    def find_min_max_and_roi(self):
        global_min = 2 ** 16 - 1
        global_max = 0
        global_delta_row = 0
        global_delta_col = 0
        counter = 0
        for ind_data in range(self.__len__()):
            img, result, _, _ = self[ind_data]

            for id_res in np.unique(result):
                if id_res == 0:
                    continue

                properties = regionprops(np.uint8(result == id_res), img)[0]
                min_row_bb, min_col_bb, max_row_bb, max_col_bb = properties.bbox
                delta_row = np.abs(max_row_bb - min_row_bb)
                delta_col = np.abs(max_col_bb - min_col_bb)

                if (delta_row > self.roi_model['row']) or (delta_col > self.roi_model['col']):
                    counter += 1

                global_delta_row = max(global_delta_row, delta_row)
                global_delta_col = max(global_delta_col, delta_col)

            res_bin = result != 0
            min_curr = img[res_bin].min()
            max_curr = img[res_bin].max()

            global_min = min(global_min, min_curr)
            global_max = max(global_max, max_curr)
        print(f'{counter=}')
        print(f"global_delta_row: {global_delta_row}")
        print(f"global_delta_col: {global_delta_col}")
        self.min_cell = global_min
        self.max_cell = global_max

        self.global_delta_row = global_delta_row
        self.global_delta_col = global_delta_col

    def preprocess_features_loop_by_results_w_metric_learning(self, path_to_write, dict_path):
        dict_params = torch.load(dict_path)
        self.roi_model = dict_params['roi']
        self.find_min_max_and_roi()
        self.flag_new_roi = self.global_delta_row > self.roi_model['row'] or self.global_delta_col > self.roi_model['col']
        if self.flag_new_roi:
            self.global_delta_row = max(self.global_delta_row, self.roi_model['row'])
            self.global_delta_col = max(self.global_delta_col, self.roi_model['col'])
            print("Assign new region of interest")
            print(f"old ROI: {self.roi_model}, new: row: {self.global_delta_row}, col : {self.global_delta_col}")
        else:
            print("We don't assign new region of interest - use the old one")

        self.pad_value = dict_params['pad_value']
        # models params
        model_name = dict_params['model_name']
        mlp_dims = dict_params['mlp_dims']
        mlp_normalized_features = dict_params['mlp_normalized_features']
        # models state_dict
        trunk_state_dict = dict_params['trunk_state_dict']
        embedder_state_dict = dict_params['embedder_state_dict']

        trunk = set_model_architecture(model_name)
        trunk.load_state_dict(trunk_state_dict)
        self.trunk = trunk
        self.trunk.eval()

        embedder = MLP(mlp_dims, normalized_feat=mlp_normalized_features)
        embedder.load_state_dict(embedder_state_dict)
        self.embedder = embedder
        self.embedder.eval()

        cols = ["seg_label",
                "frame_num",
                "area",
                "min_row_bb", "min_col_bb", "max_row_bb", "max_col_bb",
                "centroid_row", "centroid_col",
                "major_axis_length", "minor_axis_length",
                "max_intensity", "mean_intensity", "min_intensity"
                ]


        cols_resnet = [f'feat_{i}' for i in range(mlp_dims[-1])]
        cols += cols_resnet
        
        fixed_args = {'obj':self,'cols':cols,'cols_resnet':cols_resnet,'path_to_write':path_to_write, 'img_lst': self.images, 'result_lst': self.results}
        
        # run_multiprocess(extract_feat, range(len(self)), fixed_args)
        
        for ind_data in trange(len(self)):
            construct_csv(ind_data,**fixed_args)
        
        full_dir = op.join(path_to_write, "csv")    
        print(f"files were saved to : {full_dir}")

    
def construct_csv(ind_data,obj,img_lst,result_lst,cols,cols_resnet,path_to_write,lock=None,firstframeflag=None,subst_value=None,):
    img_path = Path(img_lst[ind_data])
    img = imread(img_path)
    im_name = img_path.stem
    im_num = int(re.findall('f\d+', im_name)[0][1:])-1

    result_path = Path(result_lst[ind_data])
    result = imread(result_path)
    
    # num_labels = np.unique(result).shape[0] - 1

    # df = pd.DataFrame(index=range(num_labels), columns=cols)
    
    props = ['label', 'area', 'bbox', 'centroid', 'major_axis_length', 'minor_axis_length', 'max_intensity', 'mean_intensity', 'min_intensity']
    props_table = regionprops_table(result, img, properties=props)
    
    col_rename = {'label': 'seg_label',
    'bbox-0': 'min_row_bb',
    'bbox-1': 'min_col_bb',
    'bbox-2': 'max_row_bb',
    'bbox-3': 'max_col_bb',
    'centroid-0': 'centroid_row',
    'centroid-1': 'centroid_col',}
    
    df = pd.DataFrame(props_table)
    df.rename(columns=col_rename, inplace=True)
    
    bbox_lst = list(zip(df['min_row_bb'], df['min_col_bb'], df['max_row_bb'], df['max_col_bb']))
    embedded_feats = [obj.extract_freature_metric_learning(bbox, img.copy(), result.copy(), id_res) for id_res, bbox in enumerate(bbox_lst)]
    embedded_feats_df = pd.DataFrame(embedded_feats, columns=cols_resnet)
    df = pd.concat([df, embedded_feats_df], axis=1)
    
    
    
    
    # for row_ind, id_res in enumerate(np.unique(result)[1:]):
        
    #     # # extracting statistics using regionprops
    #     properties = regionprops(np.uint8(result == id_res), img)[0]

    #     embedded_feat = obj.extract_freature_metric_learning(properties.bbox, img.copy(), result.copy(), id_res)
    #     df.loc[row_ind, cols_resnet] = embedded_feat
    #     df.loc[row_ind, "seg_label"] = id_res

    #     df.loc[row_ind, "area"] = properties.area

    #     df.loc[row_ind, "min_row_bb"], df.loc[row_ind, "min_col_bb"], \
    #     df.loc[row_ind, "max_row_bb"], df.loc[row_ind, "max_col_bb"] = properties.bbox

    #     df.loc[row_ind, "centroid_row"], df.loc[row_ind, "centroid_col"] = \
    #         properties.centroid[0].round().astype(np.int16), \
    #         properties.centroid[1].round().astype(np.int16)

    #     df.loc[row_ind, "major_axis_length"], df.loc[row_ind, "minor_axis_length"] = \
    #         properties.major_axis_length, properties.minor_axis_length

    #     df.loc[row_ind, "max_intensity"], df.loc[row_ind, "mean_intensity"], df.loc[row_ind, "min_intensity"] = \
    #         properties.max_intensity, properties.mean_intensity, properties.min_intensity


    df.loc[:, "frame_num"] = im_num

    if df.isnull().values.any():
        warnings.warn("Pay Attention! there are Nan values!")

    full_dir = op.join(path_to_write, "csv")
    os.makedirs(full_dir, exist_ok=True)
    im_num=str(im_num).zfill(4)
    file_path = op.join(full_dir, f"frame_{im_num}.csv")
    df.to_csv(file_path, index=False)
    return full_dir







if __name__ == "__main__":
    from time import time
    
    start = time()
    input_images = '/home/Test_images/nd2/Run4/c4z1t91v1_s1/Images_Registered'
    input_segmentation = '/home/Test_images/nd2/Run4/c4z1t91v1_s1/Masks_Cellpose'
    model = "PhC-C2DH-U373"
    input_model = f"/home/ImageAnalysis_pipeline/pipeline/tracking/gnn_track/models/{model}/all_params.pth"

    output_csv = '/home/Test_images/nd2/Run4/c4z1t91v1_s1/gnn_files'

    create_csv(input_images, input_segmentation, input_model, output_csv, 'RFP')
    end = time()
    print(f"Time to process: {round(end-start,ndigits=3)} sec\n")