import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch, add_self_loops
from torch.nn import Dropout, Linear, ReLU, Sequential
import torch.nn.functional as F
from models.modules import HyperHierarchicalGRL, HyperStructLearning
from torch.nn import BatchNorm1d as BN

EPS = 1e-15


class HypSEE(torch.nn.Module):
    # branch 1: handcrafted hyperedges; branch 2: hyperedges by GNN encoder.
    def __init__(self, in_channels, hidden_channels_gnn, hidden_channels, out_channels, num_layers_gnn,
                 num_edges2, avg_num_nodes, height=3,
                 decay_rate=0.5, hgsl_constraint='sigmoid',
                 dropout=0.5, mask_empty_hyperedges=False,
                 cluster_assign='em', drop_zero_incidence=False):
        super(HypSEE, self).__init__()

        self.EPS = EPS
        self.dropout = dropout
        self.capture_incidence = False
        self.captured_incidence = None
        self.mask_empty_hyperedges = bool(mask_empty_hyperedges)
        self.keep_assignments = False
        self.view_assignment_snapshot = None
        self.cluster_temp = 10.0 if cluster_assign == 'direct_orth' else 1.0

        if hidden_channels_gnn == 0:
            hidden_channels_gnn = hidden_channels

        self.gnn_encoder, self.batch_norm, self.gnn_drop = self._build_gnn_encoder(
            in_channels, hidden_channels_gnn, hidden_channels, num_layers_gnn, dropout)
        self.gnn_encoder_S, self.batch_norm_S, self.gnn_drop_S = self._build_gnn_encoder(
            in_channels, hidden_channels_gnn, hidden_channels, num_layers_gnn, dropout)

        self.hyper_struct_learning = HyperStructLearning(
            hidden_channels, num_edges2, constraint=hgsl_constraint)

        self.hyper_hierarchical_GRL = HyperHierarchicalGRL(
            hidden_channels=hidden_channels,
            avg_num_nodes=avg_num_nodes,
            height=height,
            EPS=self.EPS,
            decay_rate=decay_rate,
            dropout=dropout,
            cluster_assign=cluster_assign,
            drop_zero_incidence=bool(drop_zero_incidence),
        )

        self.classifier = Sequential(
            Linear(hidden_channels * 2, hidden_channels * 2),
            ReLU(inplace=True),
            Dropout(dropout),
            Linear(hidden_channels * 2, out_channels),
        )

    def _build_gnn_encoder(self, in_channels, hidden_channels_gnn, hidden_channels, num_layers_gnn, dropout):
        convs = torch.nn.ModuleList()
        batch_norms = torch.nn.ModuleList()
        dropouts = torch.nn.ModuleList()
        in_ch = in_channels
        for _ in range(num_layers_gnn - 1):
            convs.append(GCNConv(in_ch, hidden_channels_gnn))
            batch_norms.append(BN(hidden_channels_gnn))
            dropouts.append(Dropout(dropout))
            in_ch = hidden_channels_gnn
        convs.append(GCNConv(in_ch, hidden_channels))
        batch_norms.append(BN(hidden_channels))
        dropouts.append(Dropout(dropout))
        return convs, batch_norms, dropouts

    def _apply_gnn_encoder(self, x, edge_index, convs, batch_norms, dropouts):
        if len(convs) == 0:
            raise RuntimeError("gnn_encoder is empty; check num_layers_gnn")
        for i, conv in enumerate(convs):
            x = conv(x, edge_index)
            x = batch_norms[i](x).relu()
            x = dropouts[i](x)
        return x

    def reset_parameters(self):
        self.hyper_struct_learning.reset_parameters()

    # Hierarchical-Aware View: s
    # Embedding-Derived View: w
    # def forward(self, X_sparse, edge_index, H_S, batch):
    def _remember_view_assignments(self, grl):
        if not self.keep_assignments:
            return None
        return {
            "assignments": grl.last_assignments,
            "node_mask": grl.last_node_mask,
        }

    def forward(self, data_S, data_T, H_S):
        if not self.keep_assignments:
            self.view_assignment_snapshot = None
        # view s:
        # print("----------------------view S--------------------")
        X_sparse_S, edge_index_S, batch_S = data_S.x, data_S.edge_index, data_S.batch
        edge_index_S, _ = add_self_loops(edge_index=edge_index_S)
        embedding_gnn_S = self._apply_gnn_encoder(
            X_sparse_S, edge_index_S, self.gnn_encoder_S, self.batch_norm_S, self.gnn_drop_S)

        embedding_gnn_S, mask_S = to_dense_batch(embedding_gnn_S, batch_S)

        bs, _num_v, _num_e = H_S.size()
        if self.mask_empty_hyperedges:
            # Batch padding adds all-zero columns. A weight of 0 keeps them out of
            # degree normalization and HSE. A real hyperedge column contains a 1.
            W_S = (H_S.sum(dim=1) > 0).to(dtype=H_S.dtype)
        else:
            W_S = torch.ones([bs, _num_e], dtype=H_S.dtype, device=H_S.device)
        grl_s = self.hyper_hierarchical_GRL
        grl_s.keep_assignments = self.keep_assignments
        pool_temp = 1.0 / self.cluster_temp
        Z_S, loss_hse_S = grl_s(embedding_gnn_S, H_S, W_S, mask_S, temp=pool_temp)
        snap_s = self._remember_view_assignments(grl_s)

        # print("----------------------view T--------------------")
        X_sparse_T, edge_index_T, batch_T = data_T.x, data_T.edge_index, data_T.batch
        edge_index_T, _ = add_self_loops(edge_index=edge_index_T)
        embedding_gnn_T = self._apply_gnn_encoder(
            X_sparse_T, edge_index_T, self.gnn_encoder, self.batch_norm, self.gnn_drop)
        embedding_gnn_T, mask_T = to_dense_batch(embedding_gnn_T, batch_T)
        H_T = self.hyper_struct_learning(embedding_gnn_T)
        if self.capture_incidence:
            self.captured_incidence = {
                "H_S": H_S.detach(),
                "H_T": H_T.detach(),
                "mask_S": mask_S.detach(),
                "mask_T": mask_T.detach(),
            }
        # W_T = torch.eye(H_T.shape[-1]).unsqueeze(0).repeat(X.shape[0], 1, 1).to(X.device)
        bs, _num_v, _num_e = H_T.size()
        W_T = torch.ones([bs, _num_e], dtype=H_T.dtype, device=H_T.device)
        grl_t = self.hyper_hierarchical_GRL
        grl_t.keep_assignments = self.keep_assignments
        Z_T, loss_hse_T = grl_t(embedding_gnn_T, H_T, W_T, mask_T, temp=pool_temp)
        if self.keep_assignments:
            self.view_assignment_snapshot = {
                "S": snap_s,
                "T": self._remember_view_assignments(grl_t),
            }

        # print("----------------------classifier--------------------")
        Z = torch.cat((Z_S, Z_T), dim=-1)
        out = self.classifier(Z)
        return Z_S, Z_T, out, loss_hse_S, loss_hse_T


    def loss_con(self, Su, Sm, Tu, Tm):
        T = 0.5
        eps = self.EPS

        Su_abs = torch.linalg.vector_norm(Su, dim=1).clamp(min=eps)
        Sm_abs = torch.linalg.vector_norm(Sm, dim=1).clamp(min=eps)
        Tu_abs = torch.linalg.vector_norm(Tu, dim=1).clamp(min=eps)
        Tm_abs = torch.linalg.vector_norm(Tm, dim=1).clamp(min=eps)

        sim_matrix_P = (torch.einsum('ik,jk->ij', Su, Sm) / torch.einsum('i,j->ij', Su_abs, Sm_abs)) / T
        Log_Pu = torch.log_softmax(sim_matrix_P, dim=-1)
        sim_matrix_Q = (torch.einsum('ik,jk->ij', Tu, Tm) / torch.einsum('i,j->ij', Tu_abs, Tm_abs)) / T
        Log_Qu = torch.log_softmax(sim_matrix_Q, dim=-1)
        KL = F.kl_div(Log_Pu, Log_Qu, reduction="batchmean", log_target=True) + F.kl_div(Log_Qu, Log_Pu, reduction="batchmean", log_target=True)
        # print(KL)
        return KL/2