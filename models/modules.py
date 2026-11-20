import torch
from torch_geometric.nn import GINConv, MLP, DenseGINConv, GCNConv, DenseGCNConv, DenseGraphConv
from torch_geometric.utils import to_dense_batch, is_undirected, to_dense_adj
from torch.nn import Linear, Parameter, ReLU, Sequential, BatchNorm1d as BN
import torch.nn.functional as F
# from hypergraph_conv import HypergraphConv
from models.hgnn import DenseHypergraphConv
from models.ClusterNet import ClusterNetPoolLayer
from math import ceil
import math
from torch_geometric.nn.models import GCN
from torch_scatter import scatter_sum
# from torch.nn import BatchNorm1d as BatchNorm

# EPS = 1e-15

CONSTRAINT_CHOICES = ('sigmoid', 'softplus')


def build_pool_layer(
    hidden_channels: int,
    num_clusters: int,
    *,
    num_iter: int = 1,
    eps: float = 1e-15,
    assign_mode: str = "em",
) -> torch.nn.Module:
    """ClusterNet soft k-means pool used by HyperHierarchicalGRL."""
    return ClusterNetPoolLayer(
        hidden_channels, num_clusters, num_iter=num_iter, eps=eps, assign_mode=assign_mode)


class HyperStructLearning(torch.nn.Module):
    def __init__(self, out_channels_gnn, num_edges2, constraint='sigmoid'):
        super(HyperStructLearning, self).__init__()
        if constraint not in CONSTRAINT_CHOICES:
            raise ValueError(
                f"constraint must be one of {CONSTRAINT_CHOICES}, got {constraint!r}")
        self.constraint = constraint
        self.num_edges2 = num_edges2
        self.lin = Linear(out_channels_gnn, num_edges2)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def _apply_constraint(self, H):
        if self.constraint == 'sigmoid':
            return torch.sigmoid(H)
        return F.softplus(H)

    def forward(self, embedding_gnn):
        embedding_gnn = embedding_gnn.unsqueeze(0) if embedding_gnn.dim() == 2 else embedding_gnn
        H = self.lin(embedding_gnn)
        return self._apply_constraint(H)


class HyperHierarchicalGRL(torch.nn.Module):
    def __init__(self, hidden_channels, avg_num_nodes, height=3, decay_rate=0.5,
                 sym_D=False, EPS=1e-15, act2=True, pool_num_iter=1,
                 dropout=0.5, cluster_assign='em', drop_zero_incidence=False):
        super(HyperHierarchicalGRL, self).__init__()

        self.height = height
        self.EPS = EPS
        self.cluster_assign = cluster_assign
        self.drop_zero_incidence = bool(drop_zero_incidence)

        hyperconv_dict = torch.nn.ModuleDict()
        pool_dict = torch.nn.ModuleDict()
        num_nodes = avg_num_nodes

        for k in range(self.height, 1, -1):
            key = str(k)
            hyperconv_dict[key] = DenseHypergraphConv(
                hidden_channels,
                hidden_channels,
                act2=act2,
                drop_rate=dropout,
                eps=EPS,
                drop_zero_incidence=self.drop_zero_incidence,
            )
            num_nodes = ceil(decay_rate * num_nodes)
            pool_dict[key] = build_pool_layer(
                hidden_channels,
                num_nodes,
                num_iter=pool_num_iter,
                eps=EPS,
                assign_mode=self.cluster_assign,
            )
        hyperconv_dict['1'] = DenseHypergraphConv(
            hidden_channels,
            hidden_channels,
            act2=act2,
            drop_rate=dropout,
            eps=EPS,
            drop_zero_incidence=self.drop_zero_incidence,
        )
        self.hyperconv_dict = hyperconv_dict
        self.pool_dict = pool_dict

        self.clu_mat = {}  # C
        self.vol_dict = {}
        self.sym_D = sym_D
        # Diagnostic only. Training leaves this off so forward does not retain assignments.
        self.keep_assignments = False
        self.last_assignments = None
        self.last_node_mask = None

    def _hyperconv(self, k, X, H, W, D, mask):
        try:
            return self.hyperconv_dict[str(k)](X, H, W, D=D, mask=mask)
        except AssertionError as err:
            raise AssertionError(f"HyperHierarchicalGRL layer k={k}: {err}") from err

    def forward(self, X, H, W, mask, temp=1.0):
        assert X.dim() == 3  # b*n*d
        assert H.dim() == 3  # b*n*k
        assert W.dim() == 2  # b*k
        assert mask.dim() == 2  # b*n
        # Drop the previous forward's cached tensors so we never hold two batches'
        # autograd graphs on the module at once.
        self.clu_mat.clear()
        self.vol_dict.clear()
        self.last_assignments = None
        self.last_node_mask = None
        input_mask = mask
        H_input = H

        for k in range(self.height, 1, -1):
            bs, _num_v, _num_e = H.size()
            if k == self.height:
                mask = mask.view([bs, _num_v, 1]).to(H.dtype).to(H.device)
            else:
                mask = None
            if self.sym_D:
                HW = torch.einsum('bij,bj->bij', H, W)
                D = torch.einsum('bij,bjk->bik', HW, H.transpose(-1, -2))
                D = torch.einsum('bii->bi', D)
            else:
                D = torch.einsum('bij,bj->bi', H, W)
            D = D.clamp(min=self.EPS)
            Z = self._hyperconv(k, X, H, W, D, mask)
            C = self.pool_dict[str(k)](Z, H=H, W=W, D=D, mask=mask, tau=temp)
            C = torch.softmax(C / temp, dim=-1)
            H = C.transpose(-1, -2).matmul(H)
            X = C.transpose(-1, -2).matmul(Z)


            self.clu_mat[k] = C
            self.vol_dict[k] = D

        k = 1
        bs, _num_v, _num_e = H.size()
        if k == self.height:
            mask = mask.view([bs, _num_v, 1]).to(H.dtype).to(H.device)
        else:
            mask = None
        if self.sym_D:
            HW = torch.einsum('bij,bj->bij', H, W)
            D = torch.einsum('bij,bjk->bik', HW, H.transpose(-1, -2))
            D = torch.einsum('bii->bi', D)
        else:
            D = torch.einsum('bij,bj->bi', H, W)
        D = D.clamp(min=self.EPS)
        Z = self._hyperconv(k, X, H, W, D, mask)
        C = torch.ones([bs, C.shape[-1], 1]).to(H.dtype).to(X.device)
        self.clu_mat[k] = C
        self.vol_dict[k] = D

        # Z = torch.sum(Z, dim=-2)
        Z = torch.mean(Z, dim=-2)
        loss_hse = self.hse_loss(H_input, W, 1e-10)
        if self.keep_assignments:
            # New dict: a shared encoder's next view must not overwrite the previous snapshot.
            self.last_assignments = {int(k): v.detach() for k, v in self.clu_mat.items()}
            self.last_node_mask = input_mask.detach()
        # ``hse_loss`` is the only consumer of these caches; release the module-level
        # references now so the graph tensors can be freed right after backward().
        self.clu_mat.clear()
        self.vol_dict.clear()
        return Z, loss_hse

    def hse_loss(self, H, W, EPS):
        bs, _num_v, _num_e = H.size()
        B = torch.einsum('bij->bj', H)  # b*k
        B = B.clamp(min=EPS)
        B_inv = 1.0 / (B + EPS)
        B_inv[B_inv == float("inf")] = 0

        # ass_mat[k] for k < height only; level ``height`` is the identity (no n×n storage).
        vol_dict = self.vol_dict
        clu_mat = self.clu_mat
        ass_mat = {self.height - 1: clu_mat[self.height]}
        for k in range(self.height - 2, 0, -1):
            ass_mat[k] = ass_mat[k + 1].matmul(clu_mat[k + 1])

        hse_loss = torch.zeros([bs], device=H.device)
        vol_H = torch.einsum('bi->b', vol_dict[self.height]).clamp(min=EPS)  # b
        vol_dict[0] = vol_H.unsqueeze(-1)
        for k in range(1, self.height + 1):
            if self.sym_D:
                CV = torch.einsum('bij,bj->bij', clu_mat[k], vol_dict[k - 1])
                vol_parent = torch.einsum('bij->bjk', CV, clu_mat[k].transpose(-1, -2))
                vol_parent = torch.einsum('bii->bi', vol_parent)
            else:
                vol_parent = torch.einsum('bij,bj->bi', clu_mat[k], vol_dict[k - 1])
            log_vol_ratio_k = torch.log2(
                (vol_dict[k].clamp(min=EPS) / vol_parent.clamp(min=EPS)).clamp(min=EPS)
            )
            if k == self.height:
                # ass_mat[height] is I; avoid materializing (B, n, n).
                # (1 - I) has zeros on the diagonal and ones off-diagonal, so
                # HS = H^T @ (1 - I) = colsum(H^T) - H^T.
                SH = H
                H_T = H.transpose(-1, -2)
                HS = H_T.sum(dim=-1, keepdim=True) - H_T
            else:
                A = ass_mat[k]
                SH = torch.einsum('bij,bjk->bik', A.transpose(-1, -2), H)  # b*n*k
                HS = torch.einsum('bij,bjk->bik', H.transpose(-1, -2), 1.0 - A)  # b*k*n
            WB = W * B_inv  # b*k
            SHWB = torch.einsum('bij,bj->bij', SH, WB)  # b*n*k
            cut_k = torch.einsum('bij,bji->bi', SHWB, HS)
            hse_loss_k = - cut_k * log_vol_ratio_k  # b*n
            hse_loss += hse_loss_k.sum(dim=-1) / vol_H
        return hse_loss.mean()


    def hse_loss_old(self, H, W, EPS):
        bs, _num_v, _num_e = H.size()
        B = torch.einsum('bij->bj', H)  # b*k
        B = B.clamp(min=EPS)
        B_inv = 1.0 / (B + EPS)
        B_inv[B_inv == float("inf")] = 0

        ass_mat = {}
        ass_mat[self.height] = torch.eye(_num_v).unsqueeze(0).repeat(bs, 1, 1).to(H.device)
        vol_dict = self.vol_dict
        clu_mat = self.clu_mat

        ass_mat[self.height - 1] = self.clu_mat[self.height]
        for k in range(self.height - 2, 0, -1):
            ass_mat[k] = ass_mat[k + 1].matmul(clu_mat[k + 1])

        hse_loss = torch.zeros([bs], device=H.device)
        vol_H = torch.einsum('bi->b', vol_dict[self.height]).clamp(min=EPS)  # b
        vol_dict[0] = vol_H.unsqueeze(-1)
        for k in range(1, self.height + 1):
            if self.sym_D:
                CV = torch.einsum('bij,bj->bij', clu_mat[k], vol_dict[k - 1])
                vol_parent = torch.einsum('bij->bjk', CV, clu_mat[k].transpose(-1, -2))
                vol_parent = torch.einsum('bii->bi', vol_parent)
            else:
                vol_parent = torch.einsum('bij,bj->bi', clu_mat[k], vol_dict[k - 1])
            log_vol_ratio_k = torch.log2(
                (vol_dict[k].clamp(min=EPS) / vol_parent.clamp(min=EPS)).clamp(min=EPS)
            )
            # if k == self.height:
            #     SH = H
            SH = torch.einsum('bij,bjk->bik', ass_mat[k].transpose(-1, -2), H)  # b*n*k
            HS = torch.einsum('bij,bjk->bik', H.transpose(-1, -2), (1.0 - ass_mat[k]))  # b*k*n
            WB = W * B_inv  # b*k
            SHWB = torch.einsum('bij,bj->bij', SH, WB)  # b*n*k
            # cut_k = torch.einsum('bij,bjk->bik', SHWB, HS)  # b*n*n
            # cut_k = torch.einsum('bii->bi', cut_k)  # b*n
            cut_k = torch.einsum('bij,bji->bi', SHWB, HS)
            hse_loss_k = - cut_k * log_vol_ratio_k  # b*n
            hse_loss += hse_loss_k.sum(dim=-1) / vol_H
        return hse_loss.mean()
