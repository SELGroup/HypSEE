import copy
import random
import re
from typing import Optional, Union, cast

import numpy as np
import torch
import torch_geometric.utils as tg_utils
from torch_scatter import scatter_add, scatter_max
from torch_geometric.data.data import BaseData
from torch_geometric.data.separate import separate
from torch_geometric.datasets import TUDataset

from datasets.feature_expansion import FeatureExpander


def _default_aug_device(device: Optional[Union[torch.device, str]] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _ensure_num_nodes(data: BaseData) -> BaseData:
    """Ensure num_nodes is stored explicitly for PyG batch collation."""
    if data.x is not None:
        data.num_nodes, _ = data.x.shape
    return data


def _cpu_adj_list(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Build an adjacency list on CPU for subgraph sampling (NetworkX-free)."""
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    ei = edge_index.detach().cpu().numpy()
    for u, v in zip(ei[0], ei[1]):
        u_i, v_i = int(u), int(v)
        adj[u_i].append(v_i)
    return adj


def _parse_feat_str(feat_str):
    tokens = set(feat_str.split("+")) if feat_str else set()
    degree = "deg" in tokens
    onehot_maxdeg = re.findall(r"odeg(\d+)", feat_str)
    onehot_maxdeg = int(onehot_maxdeg[0]) if onehot_maxdeg else None
    k = re.findall(r"an{0,1}k(\d+)", feat_str)
    k = int(k[0]) if k else 0
    centrality = "cent" in tokens
    return degree, onehot_maxdeg, k, centrality


def get_dataset(name, root=None, feat_str="deg") -> "TUDatasetExt":
    degree, onehot_maxdeg, k, centrality = _parse_feat_str(feat_str)
    pre_transform = FeatureExpander(
        degree=degree,
        onehot_maxdeg=onehot_maxdeg or 0,
        AK=k,
        centrality=centrality,
    ).transform
    return TUDatasetExt(
        root,
        name,
        pre_transform=pre_transform,
        use_node_attr=True,
        transform=None,
        pre_filter=None,
        processed_filename=f"data_{feat_str}.pt",
    )


def subset_dataset(dataset: "TUDatasetExt", index) -> "TUDatasetExt":
    """Return a dataset slice with a concrete type for static checkers."""
    return cast(TUDatasetExt, dataset[index])


def shuffle_dataset(dataset: "TUDatasetExt") -> "TUDatasetExt":
    """Return a shuffled dataset with a concrete type for static checkers."""
    return cast(TUDatasetExt, dataset.shuffle())


def node_drop(data, aug_ratio):
    node_num, _ = data.x.size()
    device = data.x.device
    drop_num = int(node_num * aug_ratio)

    idx_nondrop = torch.randperm(node_num, device=device)[drop_num:].sort().values

    edge_index, _ = tg_utils.subgraph(
        idx_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num
    )
    data.x = data.x[idx_nondrop]
    data.edge_index = edge_index
    data.num_nodes, _ = data.x.shape
    return data


def weighted_drop_nodes(data, aug_ratio, npower):
    node_num, _ = data.x.size()
    device = data.x.device
    drop_num = int(node_num * aug_ratio)
    if drop_num <= 0:
        return data
    if drop_num >= node_num:
        drop_num = node_num - 1

    row = data.edge_index[0]
    deg = torch.zeros(node_num, device=device, dtype=torch.float)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=device))
    deg = deg.clamp(min=0.1).pow(npower)
    idx_drop = torch.multinomial(deg, drop_num, replacement=False)
    keep_mask = torch.ones(node_num, dtype=torch.bool, device=device)
    keep_mask[idx_drop] = False
    idx_nondrop = keep_mask.nonzero(as_tuple=False).view(-1)

    edge_index, _ = tg_utils.subgraph(
        idx_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num
    )
    data.x = data.x[idx_nondrop]
    data.edge_index = edge_index
    data.num_nodes, _ = data.x.shape
    return data


def edge_pert(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    pert_num = int(edge_num * aug_ratio)
    device = data.x.device

    if pert_num <= 0 or edge_num <= 0:
        return data

    keep_count = max(edge_num - pert_num, 0)
    if keep_count > 0:
        keep_idx = torch.randperm(edge_num, device=device)[:keep_count]
        edge_index = data.edge_index[:, keep_idx]
    else:
        edge_index = data.edge_index.new_empty((2, 0))

    idx_add = torch.randint(0, node_num, (2, pert_num), device=device)
    adj = torch.zeros((node_num, node_num), device=device)
    if edge_index.numel() > 0:
        adj[edge_index[0], edge_index[1]] = 1
    adj[idx_add[0], idx_add[1]] = 1
    adj.fill_diagonal_(0)
    data.edge_index = adj.nonzero(as_tuple=False).t()
    data.num_nodes, _ = data.x.shape
    return data


def subgraph(data, aug_ratio):
    node_num, _ = data.x.size()
    device = data.x.device
    sub_num = max(int(node_num * (1 - aug_ratio)), 1)

    adj = _cpu_adj_list(data.edge_index, node_num)
    idx_sub = [int(torch.randint(0, node_num, (1,), device=device).item())]
    idx_neigh = set(adj[idx_sub[-1]])

    while len(idx_sub) < sub_num:
        if len(idx_neigh) == 0:
            idx_unsub = list(set(range(node_num)) - set(idx_sub))
            idx_neigh = {int(np.random.choice(idx_unsub))}
        sample_node = int(np.random.choice(list(idx_neigh)))
        idx_sub.append(sample_node)
        idx_neigh = idx_neigh.union(set(adj[idx_sub[-1]])).difference(set(idx_sub))

    idx_nondrop = torch.tensor(sorted(idx_sub), device=device, dtype=torch.long)
    edge_index, _ = tg_utils.subgraph(
        idx_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num
    )
    data.x = data.x[idx_nondrop]
    data.edge_index = edge_index
    data.num_nodes, _ = data.x.shape
    return data


def attr_mask(data, aug_ratio):
    node_num, _ = data.x.size()
    device = data.x.device
    mask_num = int(node_num * aug_ratio)
    if mask_num <= 0:
        return data

    _x = data.x.clone()
    token = data.x.mean(dim=0)
    idx_mask = torch.randperm(node_num, device=device)[:mask_num]
    _x[idx_mask] = token
    data.x = _x
    data.num_nodes, _ = data.x.shape
    return data


def maskN_permE(data, aug_ratio):
    data = attr_mask(data, aug_ratio)
    return edge_pert(data, aug_ratio)


_AUGMENTATIONS = {
    "dropN": node_drop,
    "permE": edge_pert,
    "subgraph": subgraph,
    "maskN": attr_mask,
    "maskN_permE": maskN_permE,
}

_RANDOM_AUGMENTATIONS = {
    "random2": (node_drop, subgraph),
    "random3": (node_drop, subgraph, edge_pert),
    "random4": (node_drop, subgraph, edge_pert, attr_mask),
    "random_mask_permE": (attr_mask, edge_pert),
}


def apply_augmentation(data, aug, aug_ratio, npower=1.0):
    if aug == "none":
        return data
    if aug == "wdropN":
        return _ensure_num_nodes(weighted_drop_nodes(data, aug_ratio, npower))
    if aug in _AUGMENTATIONS:
        return _ensure_num_nodes(_AUGMENTATIONS[aug](data, aug_ratio))
    if aug in _RANDOM_AUGMENTATIONS:
        choices = _RANDOM_AUGMENTATIONS[aug]
        fn = choices[np.random.randint(len(choices))]
        if fn is weighted_drop_nodes:
            return _ensure_num_nodes(weighted_drop_nodes(data, aug_ratio, npower))
        return _ensure_num_nodes(fn(data, aug_ratio))
    raise ValueError(f"unknown augmentation: {aug}")


_VECTORIZED_AUGS = frozenset({
    "permE", "maskN", "maskN_permE", "random_mask_permE",
    "dropN", "subgraph", "random3",
})


def _node_ptr(batch) -> torch.Tensor:
    device = batch.x.device
    ptr = getattr(batch, "ptr", None)
    if ptr is not None:
        return ptr.to(device=device)
    counts = torch.bincount(batch.batch, minlength=int(batch.num_graphs))
    ptr = counts.new_zeros(int(batch.num_graphs) + 1)
    ptr[1:] = torch.cumsum(counts, dim=0)
    return ptr


def _random_k_mask(group_id: torch.Tensor, k_per_group: torch.Tensor, num_groups: int) -> torch.Tensor:
    """True on a uniform random subset of size ``k_per_group[g]`` within each group."""
    n = group_id.numel()
    device = group_id.device
    if n == 0:
        return group_id.new_zeros((0,), dtype=torch.bool)
    k_per_group = k_per_group.clamp(min=0)
    keys = torch.rand(n, device=device, dtype=torch.float64)
    composite = group_id.to(torch.float64) + keys * 0.5
    order = composite.argsort(stable=True)
    counts = torch.bincount(group_id, minlength=num_groups)
    starts = torch.cumsum(counts, dim=0) - counts
    rank = torch.arange(n, device=device) - starts[group_id[order]]
    selected_sorted = rank < k_per_group[group_id[order]]
    mask = torch.empty(n, dtype=torch.bool, device=device)
    mask[order] = selected_sorted
    return mask


def _sample_random_endpoints(ptr: torch.Tensor, count_per_graph: torch.Tensor):
    """Sample ``count_per_graph[g]`` directed pairs with local ids in ``[0, n_g)``."""
    device = ptr.device
    num_graphs = ptr.numel() - 1
    graph_id = torch.repeat_interleave(torch.arange(num_graphs, device=device), count_per_graph)
    if graph_id.numel() == 0:
        empty = ptr.new_empty((0,), dtype=torch.long)
        return empty, empty, empty
    n = (ptr[1:] - ptr[:-1])[graph_id].to(torch.float64)
    u = (torch.rand(graph_id.numel(), device=device, dtype=torch.float64) * n).to(torch.long)
    v = (torch.rand(graph_id.numel(), device=device, dtype=torch.float64) * n).to(torch.long)
    u = torch.minimum(u, (ptr[1:] - ptr[:-1])[graph_id] - 1)
    v = torch.minimum(v, (ptr[1:] - ptr[:-1])[graph_id] - 1)
    return ptr[graph_id] + u, ptr[graph_id] + v, graph_id


def _coalesce_directed(src: torch.Tensor, dst: torch.Tensor, num_nodes: int):
    """Drop self-loops and duplicate directed edges; return row-major unique pairs."""
    keep = src != dst
    src, dst = src[keep], dst[keep]
    if src.numel() == 0:
        return src, dst
    key = src * int(num_nodes) + dst
    uniq = torch.unique(key)
    return uniq // int(num_nodes), uniq % int(num_nodes)


def _assign_edge_index(batch, edge_index: torch.Tensor, edge_batch: torch.Tensor):
    batch.edge_index = edge_index
    slice_dict = getattr(batch, "_slice_dict", None)
    if slice_dict is None or "edge_index" not in slice_dict:
        return batch
    num_graphs = int(batch.num_graphs)
    counts = torch.bincount(edge_batch.detach().cpu(), minlength=num_graphs)
    ptr = torch.zeros(num_graphs + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(counts, dim=0)
    old = slice_dict["edge_index"]
    if torch.is_tensor(old):
        ptr = ptr.to(dtype=old.dtype, device=old.device)
    # Rebind a new dict so we never mutate a shared Batch._slice_dict.
    new_slices = dict(slice_dict)
    new_slices["edge_index"] = ptr
    batch._slice_dict = new_slices
    return batch


def _batch_attr_mask(batch, aug_ratio: float, graph_mask: torch.Tensor):
    x = batch.x
    node_batch = batch.batch
    num_graphs = int(batch.num_graphs)
    num_nodes = torch.bincount(node_batch, minlength=num_graphs)
    mask_num = torch.zeros(num_graphs, dtype=torch.long, device=x.device)
    apply = graph_mask & (num_nodes > 0)
    if apply.any():
        mask_num[apply] = (num_nodes[apply].to(torch.float64) * float(aug_ratio)).to(torch.long)
        apply = apply & (mask_num > 0)
        mask_num = torch.where(apply, mask_num, torch.zeros_like(mask_num))
    select = _random_k_mask(node_batch, mask_num, num_graphs)
    token_sum = x.new_zeros((num_graphs, x.size(1)))
    token_sum.scatter_add_(0, node_batch.unsqueeze(-1).expand_as(x), x)
    token = token_sum / num_nodes.unsqueeze(-1).clamp(min=1).to(dtype=x.dtype)
    x[select] = token[node_batch[select]]
    batch.x = x
    return batch


def _batch_edge_pert(batch, aug_ratio: float, graph_mask: torch.Tensor):
    src, dst = batch.edge_index[0], batch.edge_index[1]
    node_batch = batch.batch
    num_graphs = int(batch.num_graphs)
    num_nodes_total = int(batch.x.size(0))
    edge_g = node_batch[src] if src.numel() else src.new_empty((0,), dtype=torch.long)
    num_edges = torch.bincount(edge_g, minlength=num_graphs) if src.numel() else src.new_zeros(num_graphs)
    pert_num = torch.zeros(num_graphs, dtype=torch.long, device=src.device)
    apply = graph_mask & (num_edges > 0)
    if apply.any():
        pert_num[apply] = (num_edges[apply].to(torch.float64) * float(aug_ratio)).to(torch.long)
        apply = apply & (pert_num > 0)
    if not bool(apply.any()):
        return batch

    keep_count = num_edges.clone()
    keep_count[apply] = num_edges[apply] - pert_num[apply]
    keep_edge = _random_k_mask(edge_g, keep_count, num_graphs)
    kept_src, kept_dst, kept_g = src[keep_edge], dst[keep_edge], edge_g[keep_edge]
    apply_kept = apply[kept_g]
    add_src, add_dst, _add_g = _sample_random_endpoints(_node_ptr(batch), pert_num)
    pert_src = torch.cat([kept_src[apply_kept], add_src], dim=0)
    pert_dst = torch.cat([kept_dst[apply_kept], add_dst], dim=0)
    pert_src, pert_dst = _coalesce_directed(pert_src, pert_dst, num_nodes_total)
    pert_g = node_batch[pert_src] if pert_src.numel() else pert_src

    orig_keep = ~apply[edge_g] if src.numel() else src.new_zeros((0,), dtype=torch.bool)
    orig_src, orig_dst, orig_g = src[orig_keep], dst[orig_keep], edge_g[orig_keep]
    all_src = torch.cat([orig_src, pert_src], dim=0)
    all_dst = torch.cat([orig_dst, pert_dst], dim=0)
    all_g = torch.cat([orig_g, pert_g], dim=0)
    order = all_g.argsort(stable=True)
    new_index = torch.stack([all_src[order], all_dst[order]], dim=0)
    return _assign_edge_index(batch, new_index, all_g[order])


def _node_drop_keep_mask(node_batch, num_graphs: int, aug_ratio: float, graph_mask: torch.Tensor):
    """Nodes kept by ``node_drop`` inside ``graph_mask`` graphs. Others are false.

    ``keep = n - int(n * aug_ratio)`` matches ``randperm(n)[drop_num:].sort()``.
    """
    device = node_batch.device
    num_nodes = torch.bincount(node_batch, minlength=num_graphs)
    keep_count = torch.zeros(num_graphs, dtype=torch.long, device=device)
    apply = graph_mask.to(device=device) & (num_nodes > 0)
    if not bool(apply.any()):
        return torch.zeros(node_batch.numel(), dtype=torch.bool, device=device)
    dropped = (num_nodes[apply].to(torch.float64) * float(aug_ratio)).to(torch.long)
    keep_count[apply] = num_nodes[apply] - dropped
    return _random_k_mask(node_batch, keep_count, num_graphs)


def _subgraph_walk(adj, num_nodes: int, sub_num: int, start: int):
    """One graph's subgraph samples, including a duplicated start when drawn.

    ``start`` is sample 0. The first frontier is the out-neighborhood, so a
    self-loop lets the next draw repeat ``start``. Later draws never repeat:
    the frontier drops every node already in the sample set. Matches
    ``subgraph`` without building a Python set on every step.
    """
    selected = [start]
    if sub_num <= 1:
        return selected
    in_sel = [False] * num_nodes
    in_sel[start] = True
    in_front = [False] * num_nodes
    frontier = []
    for node in adj[start]:
        if not in_front[node]:
            in_front[node] = True
            frontier.append(node)
    while len(selected) < sub_num:
        if not frontier:
            candidates = [node for node in range(num_nodes) if not in_sel[node]]
            jump = candidates[random.randrange(len(candidates))]
            frontier = [jump]
            in_front = [False] * num_nodes
            in_front[jump] = True
        slot = random.randrange(len(frontier))
        node = frontier[slot]
        frontier[slot] = frontier[-1]
        frontier.pop()
        in_front[node] = False
        selected.append(node)
        in_sel[node] = True
        for neigh in adj[node]:
            if (not in_sel[neigh]) and (not in_front[neigh]):
                in_front[neigh] = True
                frontier.append(neigh)
        write = 0
        for cand in frontier:
            if in_sel[cand]:
                in_front[cand] = False
            else:
                frontier[write] = cand
                write += 1
        del frontier[write:]
    return selected


def _subgraph_sorted_ids(batch, aug_ratio: float, graph_mask: torch.Tensor):
    """Per-graph sorted sample multiset from ``subgraph``, shape ``[max_sub, G]``.

    ``sub_num`` is ``max(int(n * (1 - aug_ratio)), 1)``. The walk runs on CPU
    adjacency lists; a step-by-step GPU walk is slower than this for the graph
    sizes in the TU batches.
    """
    num_graphs = int(batch.num_graphs)
    device = batch.x.device
    num_nodes = torch.bincount(batch.batch, minlength=num_graphs)
    sub_num = torch.zeros(num_graphs, dtype=torch.long, device=device)
    apply = graph_mask.to(device=device) & (num_nodes > 0)
    if not bool(apply.any()):
        return None, sub_num
    raw = (num_nodes[apply].to(torch.float64) * (1.0 - float(aug_ratio))).to(torch.long)
    sub_num[apply] = raw.clamp(min=1)

    node_ptr = batch.ptr.detach().cpu().long().tolist()
    apply_list = apply.detach().cpu().tolist()
    sub_list = sub_num.detach().cpu().tolist()
    sizes = [node_ptr[g + 1] - node_ptr[g] for g in range(num_graphs)]
    adj = [[[] for _ in range(sizes[g])] for g in range(num_graphs)]
    if int(batch.edge_index.size(1)) > 0:
        src = batch.edge_index[0].detach().cpu().tolist()
        dst = batch.edge_index[1].detach().cpu().tolist()
        node_graph = [0] * node_ptr[-1]
        node_local = [0] * node_ptr[-1]
        for g in range(num_graphs):
            offset = node_ptr[g]
            for local, global_id in enumerate(range(offset, node_ptr[g + 1])):
                node_graph[global_id] = g
                node_local[global_id] = local
        for u, v in zip(src, dst):
            g = node_graph[u]
            if apply_list[g]:
                adj[g][node_local[u]].append(node_local[v])

    # One torch draw per graph keeps ``torch.manual_seed`` in the start choice.
    # Later frontier draws use ``random`` so the step loop does not sync.
    uniform = torch.rand(num_graphs, device=device).detach().cpu()
    max_sub = max(sub_list)
    ordered = torch.full((max_sub, num_graphs), node_ptr[-1], dtype=torch.long)
    for g in range(num_graphs):
        if not apply_list[g]:
            continue
        n = sizes[g]
        start = min(int(uniform[g].item() * n), n - 1)
        chosen = _subgraph_walk(adj[g], n, int(sub_list[g]), start)
        chosen.sort()
        offset = node_ptr[g]
        ordered[: len(chosen), g] = torch.tensor([node + offset for node in chosen], dtype=torch.long)
    return ordered.to(device=device), sub_num


def _within_graph_rank(selected: torch.Tensor, node_batch, ptr: torch.Tensor, num_graphs: int):
    """Rank of each selected node among selected nodes of its graph, in index order."""
    chosen = selected.nonzero(as_tuple=False).view(-1)
    if chosen.numel() == 0:
        empty = selected.new_empty((0,), dtype=torch.long)
        return empty, empty, empty
    prefix = torch.cumsum(selected.to(torch.long), dim=0)
    graph = node_batch[chosen]
    base = torch.zeros(num_graphs, dtype=torch.long, device=selected.device)
    has_prev = ptr[:-1] > 0
    prev = (ptr[:-1] - 1).clamp(min=0)
    base[has_prev] = prefix[prev[has_prev]]
    rank = prefix[chosen] - 1 - base[graph]
    return chosen, graph, rank


def _take_rows(values: torch.Tensor, valid: torch.Tensor):
    return values[valid]


def _sync_batch_layout(batch, node_counts: torch.Tensor, edge_batch: torch.Tensor):
    """Point ``ptr``, slices, and per-graph ``num_nodes`` at the packed batch."""
    num_graphs = int(batch.num_graphs)
    device = batch.x.device
    node_ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
    node_ptr[1:] = torch.cumsum(node_counts, dim=0)
    if edge_batch.numel() == 0:
        edge_counts = torch.zeros(num_graphs, dtype=torch.long, device=device)
    else:
        edge_counts = torch.bincount(edge_batch, minlength=num_graphs)
    edge_ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
    edge_ptr[1:] = torch.cumsum(edge_counts, dim=0)
    old_ptr = getattr(batch, "ptr", None)
    if torch.is_tensor(old_ptr):
        batch.ptr = node_ptr.to(dtype=old_ptr.dtype, device=old_ptr.device)
    else:
        batch.ptr = node_ptr

    slice_dict = getattr(batch, "_slice_dict", None)
    if slice_dict is not None:
        new_slices = dict(slice_dict)
        old_x = slice_dict.get("x")
        if torch.is_tensor(old_x):
            new_slices["x"] = node_ptr.detach().to(dtype=old_x.dtype, device=old_x.device)
        old_edge = slice_dict.get("edge_index")
        if torch.is_tensor(old_edge):
            new_slices["edge_index"] = edge_ptr.detach().to(
                dtype=old_edge.dtype, device=old_edge.device
            )
        batch._slice_dict = new_slices

    inc_dict = getattr(batch, "_inc_dict", None)
    if inc_dict is not None and torch.is_tensor(inc_dict.get("edge_index")):
        old_inc = inc_dict["edge_index"]
        new_inc = dict(inc_dict)
        new_inc["edge_index"] = node_ptr[:-1].detach().to(
            dtype=old_inc.dtype, device=old_inc.device
        )
        batch._inc_dict = new_inc

    counts_cpu = node_counts.detach().cpu().tolist()
    if hasattr(batch, "_num_nodes"):
        batch._num_nodes = [int(v) for v in counts_cpu]
    store = getattr(batch, "_store", None)
    if store is not None and "num_nodes" in store:
        store["num_nodes"] = int(sum(counts_cpu))
    return batch


def _rebuild_kept_nodes(batch, old_ids: torch.Tensor, graph_ids: torch.Tensor):
    """Pack ``x`` as ``x[old_ids]`` and remap ``edge_index`` onto that order.

    Duplicate ``old_ids`` keep one output row each. Endpoints map to the last
    row of that original node, matching ``map_index`` on a sorted multiset.
    """
    num_graphs = int(batch.num_graphs)
    device = batch.x.device
    num_out = int(old_ids.numel())
    orig_n = int(batch.x.size(0))
    if num_out == 0:
        batch.x = batch.x[:0]
        batch.batch = graph_ids
        batch.edge_index = batch.edge_index.new_empty((2, 0))
        node_counts = torch.zeros(num_graphs, dtype=torch.long, device=device)
        return _sync_batch_layout(batch, node_counts, graph_ids.new_empty((0,)))

    new_pos = torch.arange(num_out, device=device, dtype=torch.float64)
    max_pos, _ = scatter_max(new_pos, old_ids, dim=0, dim_size=orig_n)
    appeared = scatter_add(
        torch.ones(num_out, dtype=torch.long, device=device),
        old_ids,
        dim=0,
        dim_size=orig_n,
    ) > 0
    assoc = torch.full((orig_n,), -1, dtype=torch.long, device=device)
    assoc[appeared] = max_pos[appeared].to(torch.long)

    src, dst = batch.edge_index[0], batch.edge_index[1]
    if src.numel():
        new_src = assoc[src]
        new_dst = assoc[dst]
        keep = (new_src >= 0) & (new_dst >= 0)
        new_src, new_dst = new_src[keep], new_dst[keep]
        edge_g = batch.batch[src[keep]]
        order = edge_g.argsort(stable=True)
        edge_index = torch.stack([new_src[order], new_dst[order]], dim=0)
        edge_g = edge_g[order]
    else:
        edge_index = batch.edge_index.new_empty((2, 0))
        edge_g = src.new_empty((0,), dtype=torch.long)

    batch.x = batch.x[old_ids]
    batch.batch = graph_ids
    batch.edge_index = edge_index
    node_counts = torch.bincount(graph_ids, minlength=num_graphs)
    return _sync_batch_layout(batch, node_counts, edge_g)


def _pack_sources(sources, num_nodes_total: int):
    olds, graphs, ranks = [], [], []
    for old, graph, rank in sources:
        if old is None or old.numel() == 0:
            continue
        olds.append(old)
        graphs.append(graph)
        ranks.append(rank)
    if not olds:
        device = sources[0][0].device if sources else torch.device("cpu")
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    old = torch.cat(olds)
    graph = torch.cat(graphs)
    rank = torch.cat(ranks)
    key = graph * (int(num_nodes_total) + 1) + rank
    order = key.argsort(stable=True)
    return old[order], graph[order]


def _apply_node_subsets(batch, aug_ratio: float, drop_mask: torch.Tensor, sub_mask: torch.Tensor):
    """Drop nodes on ``drop_mask`` graphs and run the subgraph walk on ``sub_mask`` graphs.

    The two masks are disjoint. Every other graph keeps its current nodes, so a
    preceding ``permE`` pass is left intact.
    """
    device = batch.x.device
    drop_mask = drop_mask.to(device=device)
    sub_mask = sub_mask.to(device=device)
    if not (bool(drop_mask.any()) or bool(sub_mask.any())):
        return batch

    node_batch = batch.batch
    num_graphs = int(batch.num_graphs)
    num_nodes_total = int(batch.x.size(0))
    ptr = _node_ptr(batch)
    sources = []

    identity = ~(drop_mask | sub_mask)
    if bool(identity.any()):
        chosen = identity[node_batch]
        old = chosen.nonzero(as_tuple=False).view(-1)
        graph = node_batch[old]
        sources.append((old, graph, old - ptr[graph]))

    if bool(drop_mask.any()):
        kept = _node_drop_keep_mask(node_batch, num_graphs, aug_ratio, drop_mask)
        old, graph, rank = _within_graph_rank(kept, node_batch, ptr, num_graphs)
        sources.append((old, graph, rank))

    if bool(sub_mask.any()):
        ordered, sub_num = _subgraph_sorted_ids(batch, aug_ratio, sub_mask)
        if ordered is not None:
            max_sub = int(ordered.size(0))
            row = torch.arange(max_sub, device=device)[:, None]
            valid = (row < sub_num.unsqueeze(0)) & sub_mask.unsqueeze(0)
            grid_g = torch.arange(num_graphs, device=device).view(1, -1).expand(max_sub, -1)
            grid_r = row.expand(max_sub, num_graphs)
            sources.append((
                _take_rows(ordered, valid),
                _take_rows(grid_g, valid),
                _take_rows(grid_r, valid),
            ))

    old_ids, graph_ids = _pack_sources(sources, num_nodes_total)
    return _rebuild_kept_nodes(batch, old_ids, graph_ids)


def _can_vectorize_aug(batch, aug: str) -> bool:
    if aug not in _VECTORIZED_AUGS:
        return False
    edge_attr = getattr(batch, "edge_attr", None)
    if edge_attr is not None:
        return False
    return batch.x is not None and batch.edge_index is not None and getattr(batch, "batch", None) is not None


def _augment_batch_vectorized(batch, aug, aug_ratio, device):
    # Always clone: Data.to() returns self on the same device, and Batch.clone()
    # may still share _slice_dict with the DataLoader batch.
    out = batch.to(device).clone()
    num_graphs = int(out.num_graphs)
    dev = out.x.device
    all_graphs = torch.ones(num_graphs, dtype=torch.bool, device=dev)
    none = torch.zeros(num_graphs, dtype=torch.bool, device=dev)
    if aug == "maskN":
        _batch_attr_mask(out, aug_ratio, all_graphs)
    elif aug == "permE":
        _batch_edge_pert(out, aug_ratio, all_graphs)
    elif aug == "maskN_permE":
        _batch_attr_mask(out, aug_ratio, all_graphs)
        _batch_edge_pert(out, aug_ratio, all_graphs)
    elif aug == "random_mask_permE":
        choice = torch.randint(0, 2, (num_graphs,), device=dev)
        _batch_attr_mask(out, aug_ratio, choice == 0)
        _batch_edge_pert(out, aug_ratio, choice == 1)
    elif aug == "dropN":
        _apply_node_subsets(out, aug_ratio, all_graphs, none)
    elif aug == "subgraph":
        _apply_node_subsets(out, aug_ratio, none, all_graphs)
    elif aug == "random3":
        # Same menu as ``_RANDOM_AUGMENTATIONS['random3']``: dropN, subgraph, permE.
        choice = torch.randint(0, 3, (num_graphs,), device=dev)
        _batch_edge_pert(out, aug_ratio, choice == 2)
        _apply_node_subsets(out, aug_ratio, choice == 0, choice == 1)
    else:
        raise ValueError(f"vectorized augmentation does not support {aug!r}")
    return out


def _augment_batch_per_graph(batch, aug, aug_ratio, npower, device):
    from torch_geometric.data import Batch

    # deepcopy after .to(dev): apply_augmentation mutates in place; without a copy,
    # aug1/aug2 views or the DataLoader batch could share corrupted storage.
    # PyG Data.clone() only shallow-copies tensor fields; deepcopy remains safer here.
    aug_list = [
        apply_augmentation(copy.deepcopy(data.to(device)), aug, aug_ratio, npower)
        for data in batch.to_data_list()
    ]
    return Batch.from_data_list(aug_list)


def augment_batch(batch, aug, aug_ratio, npower=1.0, device=None):
    """Augment a PyG batch. maskN/permE/dropN/subgraph/random3 paths are batched.

    Vectorized ops match the original per-graph ops in distribution (same
    counts, uniform subsets, the subgraph walk, and a simple directed graph
    after permE) but do not replay the same RNG stream as the Python loop.
    """
    dev = _default_aug_device(device)
    if aug == "none":
        return batch.to(dev)
    if _can_vectorize_aug(batch, aug):
        return _augment_batch_vectorized(batch, aug, aug_ratio, dev)
    return _augment_batch_per_graph(batch, aug, aug_ratio, npower, dev)


class TUDatasetExt(TUDataset):
    """TU Dortmund graph kernel benchmarks with a per-graph cache."""

    url = "https://www.chrsmrrs.com/graphkerneldatasets"

    def __init__(
        self,
        root,
        name,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        use_node_attr=False,
        processed_filename="data.pt",
    ):
        self.processed_filename = processed_filename
        self._base_data_list: Optional[list[Optional[BaseData]]] = None
        super().__init__(
            root, name, transform, pre_transform, pre_filter,
            use_node_attr=use_node_attr,
        )
        # TUDataset.process() calls get() before pre_transform, which can cache
        # raw graphs. Drop that stale cache after init.
        self._base_data_list = None

    def process(self) -> None:
        super().process()
        self._base_data_list = None

    @property
    def processed_file_names(self):
        return self.processed_filename

    def _get_base_data(self, idx) -> BaseData:
        if self._base_data_list is None:
            self._base_data_list = cast(list[Optional[BaseData]], [None] * self.len())
        base_data_list = self._base_data_list
        if base_data_list[idx] is not None:
            return cast(BaseData, base_data_list[idx])

        data = cast(BaseData, separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        ))
        base_data_list[idx] = data
        return data

    def get(self, idx) -> BaseData:
        if self.len() == 1:
            if self._data is None:
                raise RuntimeError("dataset storage is not initialized")
            return cast(BaseData, copy.copy(self._data))
        return copy.copy(self._get_base_data(idx))
