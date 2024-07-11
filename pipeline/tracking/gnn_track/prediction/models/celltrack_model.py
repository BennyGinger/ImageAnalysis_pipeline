import torch
import torch.nn as nn
from torch.nn.modules.distance import CosineSimilarity

from pipeline.tracking.gnn_track.prediction.models.mlp import MLP
from pipeline.tracking.gnn_track.prediction.models.edge_mpnn import CellTrack_GNN


class CellTrack_Model(nn.Module):
    def __init__(self,
                 hand_NodeEncoder_dic={},
                 learned_NodeEncoder_dic={},
                 intialize_EdgeEncoder_dic={},
                 message_passing={},
                 edge_classifier_dic={}
                 ):
        super(CellTrack_Model, self).__init__()
        self.distance = CosineSimilarity()
        self.handcrafted_node_embedding = MLP(**hand_NodeEncoder_dic)
        self.learned_node_embedding = MLP(**learned_NodeEncoder_dic)
        self.learned_edge_embedding = MLP(**intialize_EdgeEncoder_dic)

        self.message_passing = CellTrack_GNN(**message_passing.kwargs)

        self.edge_classifier = MLP(**edge_classifier_dic)

    def forward(self, node_features, edge_index,):
        cell_feat, cell_param = node_features
        x_init = torch.cat((cell_feat, cell_param), dim=-1)
        src, trg = edge_index
        similarity1 = self.distance(x_init[src], x_init[trg])
        abs_init = torch.abs(x_init[src] - x_init[trg])
        cell_feat = self.handcrafted_node_embedding(cell_feat)
        cell_param = self.learned_node_embedding(cell_param)
        node_features = torch.cat((cell_feat, cell_param), dim=-1)
        src, trg = edge_index
        similarity2 = self.distance(node_features[src], node_features[trg])
        edge_feat_in = torch.cat((abs_init, similarity1[:, None], node_features[src], node_features[trg], torch.abs(node_features[src] - node_features[trg]), similarity2[:, None]), dim=-1)
        edge_init_features = self.learned_edge_embedding(edge_feat_in)
        edge_feat_mp = self.message_passing(node_features, edge_index, edge_init_features)
        pred = self.edge_classifier(edge_feat_mp).squeeze()
        return pred