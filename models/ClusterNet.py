"""ClusterNet assigner: two-step soft k-means on embedder outputs."""
from __future__ import annotations

import torch
import torch.nn.functional as F


ASSIGN_MODES = ("em", "direct_orth")


def orthogonal_prototypes(num_clusters: int, embed_dim: int) -> torch.Tensor:
    """Orthonormal rows when the cluster count fits in the embedding dimension."""
    if num_clusters <= embed_dim:
        basis, _ = torch.linalg.qr(torch.randn(embed_dim, num_clusters))
        return basis.T.contiguous()
    basis, _ = torch.linalg.qr(torch.randn(embed_dim, embed_dim))
    extra = F.normalize(torch.randn(num_clusters - embed_dim, embed_dim), dim=-1)
    return torch.cat([basis.T, extra], dim=0)


def cluster(
    data: torch.Tensor,
    k: int,
    num_iter: int,
    *,
    init: torch.Tensor,
    tau: float,
    eps: float = 1e-15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable soft k-means on the unit sphere. Returns cluster means, soft assignments, and distances."""
    data = F.normalize(data, p=2, dim=-1)
    mu = init
    for _ in range(num_iter):
        dist = data @ mu.t()
        r = torch.softmax(dist / tau, dim=-1)
        cluster_r = r.sum(dim=-2)
        cluster_mean = r.transpose(-1, -2) @ data
        mu = cluster_mean / cluster_r.unsqueeze(-1).clamp(min=eps)
    dist = data @ mu.t()
    r = torch.softmax(dist / tau, dim=-1)
    return mu, r, dist


def _batched_cluster_init(init: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Broadcast init to (batch, k, embed_dim). Accepts (k, d) or (batch, k, d)."""
    if init.dim() == 2:
        return init.unsqueeze(0).expand(batch_size, -1, -1)
    if init.dim() == 3:
        return init
    raise ValueError(
        f"init must have shape (k, embed_dim) or (batch, k, embed_dim), got {tuple(init.shape)}"
    )


def cluster_batched(
    data: torch.Tensor,
    k: int,
    num_iter: int,
    *,
    init: torch.Tensor,
    tau: float,
    eps: float = 1e-15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched soft k-means. data: (batch, num_nodes, embed_dim), init: (k, embed_dim)."""
    data = F.normalize(data, p=2, dim=-1)
    mu = _batched_cluster_init(init, data.size(0))
    for _ in range(num_iter):
        dist = torch.bmm(data, mu.transpose(1, 2))
        r = torch.softmax(dist / tau, dim=-1)
        cluster_r = r.sum(dim=1, keepdim=True).transpose(1, 2)
        cluster_mean = torch.bmm(r.transpose(1, 2), data)
        mu = cluster_mean / cluster_r.clamp(min=eps)
    dist = torch.bmm(data, mu.transpose(1, 2))
    r = torch.softmax(dist / tau, dim=-1)
    return mu, r, dist


class ClusterNetAssigner(torch.nn.Module):
    """ClusterNet assignment head: embedder output + two-step soft k-means."""

    def __init__(
        self,
        embed_dim: int,
        num_clusters: int,
        num_iter: int = 1,
        eps: float = 1e-15,
        assign_mode: str = "em",
    ):
        super().__init__()
        if assign_mode not in ASSIGN_MODES:
            raise ValueError(f"assign_mode must be one of {ASSIGN_MODES}, got {assign_mode!r}")
        if assign_mode == "direct_orth":
            proto_init = "orthogonal"
            mean_update = False
            proto_detach = False
        else:
            proto_init = "uniform"
            mean_update = True
            proto_detach = True
        self.K = num_clusters
        self.num_iter = num_iter
        self.eps = eps
        self.assign_mode = assign_mode
        self.proto_init = proto_init
        self.mean_update = mean_update
        self.proto_detach = proto_detach
        if self.proto_init == "orthogonal":
            initial = orthogonal_prototypes(num_clusters, embed_dim)
        else:
            initial = torch.rand(num_clusters, embed_dim)
        self.init = torch.nn.Parameter(initial)

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        if not self.mean_update:
            return self._cosine_logits(x)
        if x.dim() == 2:
            return self._forward_single(x, tau)
        if x.dim() == 3:
            return self._forward_batched(x, tau)
        raise ValueError(
            f"expected x with shape (num_nodes, embed_dim) or "
            f"(batch, num_nodes, embed_dim), got {tuple(x.shape)}"
        )

    def _prototype_matrix(self) -> torch.Tensor:
        prototypes = self.init.detach() if self.proto_detach else self.init
        return F.normalize(prototypes, p=2, dim=-1)

    def _cosine_logits(self, x: torch.Tensor) -> torch.Tensor:
        nodes = F.normalize(x, p=2, dim=-1)
        prototypes = self._prototype_matrix()
        if nodes.dim() == 2:
            return nodes @ prototypes.t()
        return torch.matmul(nodes, prototypes.t())

    def _second_init(self, mu_init: torch.Tensor) -> torch.Tensor:
        if self.proto_detach:
            return mu_init.detach().clone()
        return mu_init

    def _forward_single(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        mu_init, _, _ = cluster(
            x, self.K, self.num_iter, init=self.init, tau=tau, eps=self.eps,
        )
        _, _, dist = cluster(
            x, self.K, 1, init=self._second_init(mu_init), tau=tau, eps=self.eps,
        )
        return dist

    def _forward_batched(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        mu_init, _, _ = cluster_batched(
            x, self.K, self.num_iter, init=self.init, tau=tau, eps=self.eps,
        )
        _, _, dist = cluster_batched(
            x, self.K, 1, init=self._second_init(mu_init), tau=tau, eps=self.eps,
        )
        return dist


class ClusterNetPoolLayer(torch.nn.Module):
    """Hierarchical pool layer: ClusterNet soft k-means assignment from nodes to clusters."""

    def __init__(
        self,
        hidden_channels: int,
        num_clusters: int,
        num_iter: int = 1,
        eps: float = 1e-15,
        assign_mode: str = "em",
    ):
        super().__init__()
        self.assigner = ClusterNetAssigner(
            hidden_channels, num_clusters, num_iter=num_iter, eps=eps, assign_mode=assign_mode)

    def reset_parameters(self):
        if self.assigner.proto_init == "orthogonal":
            with torch.no_grad():
                refreshed = orthogonal_prototypes(self.assigner.K, self.assigner.init.size(1))
                self.assigner.init.copy_(refreshed.to(device=self.assigner.init.device, dtype=self.assigner.init.dtype))
            return
        torch.nn.init.uniform_(self.assigner.init, 0, 1)

    def forward(
        self,
        Z: torch.Tensor,
        H: torch.Tensor | None = None,
        W: torch.Tensor | None = None,
        D: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        tau: float = 1.0,
    ) -> torch.Tensor:
        """Return cosine-similarity logits, shape (batch, num_nodes, num_clusters)."""
        del H, W, D, mask
        return self.assigner(Z, tau=tau)
