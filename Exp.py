
import gc
import torch
import traceback
from typing import Any, cast
from datasets.tu_dataset import get_dataset, subset_dataset, shuffle_dataset, TUDatasetExt, augment_batch
import random
import os
import numpy as np
from torch_geometric.loader import DataLoader
from datasets.loaders import IterLoader
from models.model import HypSEE
from datasets.data_utils import (
    load_hypergraphs, hypergraph_to_dense_batch, precompute_hypergraphs,
    HypergraphDenseBatchCache,
)
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from copy import deepcopy


class Exp:
    def __init__(self, configs):
        self.configs = configs
        self._debug_ctx: dict[str, Any] = {}
        self.nonfinite_train_steps = 0

    def _set_debug_ctx(self, **kwargs):
        self._debug_ctx.update(kwargs)

    @staticmethod
    def _is_finite(*tensors: torch.Tensor) -> bool:
        return all(torch.isfinite(t).all() for t in tensors)

    @staticmethod
    def _is_oom_error(err: BaseException) -> bool:
        """True for CUDA OOM and generic RuntimeError wrappers (CPU / older PyTorch)."""
        oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(err, oom_type):
            return True
        return isinstance(err, RuntimeError) and "out of memory" in str(err).lower()

    @staticmethod
    def _raise_oom_without_traceback(err: BaseException):
        """Re-raise OOM without traceback frames that may retain large tensor locals."""
        msg = str(err)
        tb = err.__traceback__
        if tb is not None:
            traceback.clear_frames(tb)
        oom_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
        raise oom_type(msg) from None

    def _optimizer_step(self, model, optimizer, loss):
        if not self._is_finite(loss):
            print(f"[warn] skip step: non-finite loss ({loss.item()})")
            optimizer.zero_grad(set_to_none=True)
            return False

        try:
            loss.backward()
        except RuntimeError as err:
            if self._is_oom_error(err):
                optimizer.zero_grad(set_to_none=True)
                self._release_cuda_memory()
                self._raise_oom_without_traceback(err)
            raise
        grad_clip = self.configs.get("grad_clip", 5.0)
        if grad_clip and grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return True

    def fix_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def train(self, model, labeled_loader, unlabeled_loader, hypergraph_hie_aware_dict, optimizer, anchor_queue, epoch, device):
        model.train()

        labeled_loader.new_epoch()
        unlabeled_loader.new_epoch()

        assert len(anchor_queue) == self.configs["num_anchors"]

        warm = epoch <= self.configs["warm_epochs"]
        weight_consistency = self.configs["weight_consistency"]
        weight_hse = self.configs["weight_hse"]
        num_steps = len(unlabeled_loader)

        stat = torch.zeros((), device=device, dtype=torch.float64)
        total_loss = stat
        total_loss_sup = stat.clone()
        total_loss_con = stat.clone()
        total_loss_hse = stat.clone()
        total_loss_con_raw = stat.clone()
        total_loss_hse_raw = stat.clone()
        total_labeled_graphs = 0
        total_unlabeled_graphs = 0
        self.nonfinite_train_steps = 0

        for _ in range(num_steps):
            labeled_batch = None
            unlabeled_batch = None
            labeled_H_S = None
            unlabeled_H_S = None
            labeled_data_S = labeled_data_T = None
            unlabeled_data_S = unlabeled_data_T = None
            S_labeled = T_labeled = out_labeled = None
            S_unlabeled = T_unlabeled = None
            S_anchors = T_anchors = None
            loss_sup = loss_con = loss_hse = loss = None
            try:
                labeled_cpu = labeled_loader.next()
                unlabeled_cpu = unlabeled_loader.next()
                labeled_ids = self._cpu_graph_ids(labeled_cpu)
                unlabeled_ids = self._cpu_graph_ids(unlabeled_cpu)
                num_labeled_graphs = int(labeled_cpu.num_graphs)
                num_unlabeled_graphs = int(unlabeled_cpu.num_graphs)
                labeled_batch = self._batch_to_device(labeled_cpu, device)
                unlabeled_batch = self._batch_to_device(unlabeled_cpu, device)

                labeled_H_S = self._batch_hypergraph(
                    labeled_batch, hypergraph_hie_aware_dict, device, labeled_ids)
                labeled_data_S, labeled_data_T = self._augment_views(labeled_batch, device)
                optimizer.zero_grad()

                S_labeled, T_labeled, out_labeled, loss_hse_S_labeled, loss_hse_T_labeled = self._forward_model(
                    model, labeled_data_S, labeled_data_T, labeled_H_S)
                loss_sup = F.cross_entropy(out_labeled, labeled_batch.y)

                unlabeled_H_S = self._batch_hypergraph(
                    unlabeled_batch, hypergraph_hie_aware_dict, device, unlabeled_ids)
                unlabeled_data_S, unlabeled_data_T = self._augment_views(unlabeled_batch, device)

                if warm:
                    with torch.no_grad():
                        S_unlabeled, T_unlabeled, _, loss_hse_S_unlabeled, loss_hse_T_unlabeled = self._forward_model(
                            model, unlabeled_data_S, unlabeled_data_T, unlabeled_H_S)
                        S_anchors = torch.stack([x[0] for x in anchor_queue])
                        T_anchors = torch.stack([x[1] for x in anchor_queue])
                        loss_con = model.loss_con(S_unlabeled, S_anchors, T_unlabeled, T_anchors)
                        hse_labeled_scale = num_labeled_graphs / max(num_unlabeled_graphs, 1)
                        loss_hse = (loss_hse_S_unlabeled + loss_hse_T_unlabeled
                                    + hse_labeled_scale * (loss_hse_S_labeled + loss_hse_T_labeled))
                    loss = loss_sup
                else:
                    S_unlabeled, T_unlabeled, _, loss_hse_S_unlabeled, loss_hse_T_unlabeled = self._forward_model(
                        model, unlabeled_data_S, unlabeled_data_T, unlabeled_H_S)
                    S_anchors = torch.stack([x[0] for x in anchor_queue])
                    T_anchors = torch.stack([x[1] for x in anchor_queue])
                    loss_con = model.loss_con(S_unlabeled, S_anchors, T_unlabeled, T_anchors)
                    hse_labeled_scale = num_labeled_graphs / max(num_unlabeled_graphs, 1)
                    loss_hse = (loss_hse_S_unlabeled + loss_hse_T_unlabeled
                                + hse_labeled_scale * (loss_hse_S_labeled + loss_hse_T_labeled))
                    loss = loss_sup + weight_consistency * loss_con + weight_hse * loss_hse

                total_loss_con = total_loss_con + (weight_consistency * loss_con).detach().double() * num_unlabeled_graphs
                total_loss_hse = total_loss_hse + (weight_hse * loss_hse).detach().double() * num_unlabeled_graphs
                total_loss_con_raw = total_loss_con_raw + loss_con.detach().double() * num_unlabeled_graphs
                total_loss_hse_raw = total_loss_hse_raw + loss_hse.detach().double() * num_unlabeled_graphs
                total_unlabeled_graphs += num_unlabeled_graphs

                if not self._optimizer_step(model, optimizer, loss):
                    self.nonfinite_train_steps += 1
                    continue

                total_loss = total_loss + loss.detach().double()
                total_loss_sup = total_loss_sup + loss_sup.detach().double() * num_labeled_graphs
                total_labeled_graphs += num_labeled_graphs

                for i in range(num_labeled_graphs):
                    anchor_queue.append((S_labeled[i].detach(), T_labeled[i].detach()))
                    if len(anchor_queue) > self.configs["num_anchors"]:
                        anchor_queue.pop(0)
            finally:
                labeled_batch = None
                unlabeled_batch = None
                labeled_H_S = None
                unlabeled_H_S = None
                labeled_data_S = labeled_data_T = None
                unlabeled_data_S = unlabeled_data_T = None
                S_labeled = T_labeled = out_labeled = None
                S_unlabeled = T_unlabeled = None
                S_anchors = T_anchors = None
                loss_sup = loss_con = loss_hse = loss = None

        if num_steps == 0:
            return 0.0, 0.0, 0.0, 0.0
        summed = torch.stack([
            total_loss,
            total_loss_sup,
            total_loss_con,
            total_loss_hse,
            total_loss_con_raw,
            total_loss_hse_raw,
        ]).detach().cpu().tolist()
        mean_sup = summed[1] / max(total_labeled_graphs, 1)
        mean_unlabeled = max(total_unlabeled_graphs, 1)
        return (
            summed[0] / num_steps,
            mean_sup,
            summed[2] / mean_unlabeled,
            summed[3] / mean_unlabeled,
        )

    @staticmethod
    def _cpu_graph_ids(batch):
        """Graph ids from a CPU batch. Reading them after ``.to(cuda)`` syncs."""
        graph_id = batch.graph_id.detach()
        if graph_id.is_cuda:
            graph_id = graph_id.cpu()
        return [int(gid) for gid in graph_id.view(-1).tolist()]

    def _batch_hypergraph(self, batch, hypergraph_hie_aware_dict, device, graph_id_list=None):
        if graph_id_list is None:
            graph_id_list = self._cpu_graph_ids(batch)
        cache = getattr(self, "_hg_dense_cache", None)
        return hypergraph_to_dense_batch(
            hypergraph_dict=hypergraph_hie_aware_dict,
            graph_id_list=graph_id_list,
            num_edges=self.configs['num_edges1'],
            device=device,
            dtype=batch.x.dtype,
            cache=cache,
        )

    def _augment_views(self, batch, device, augment=True):
        if not augment:
            batch = batch.to(device)
            return batch, batch

        cfg = self.configs
        aug1 = cfg.get("aug1", "none")
        aug_ratio1 = cfg.get("aug_ratio1", 0.2)
        aug2 = cfg.get("aug2", "none")
        aug_ratio2 = cfg.get("aug_ratio2", 0.2)
        npower = cfg.get("npower", 1.0)

        data_S = cast(Any, augment_batch(batch, aug1, aug_ratio1, npower=npower, device=device))
        data_T = cast(Any, augment_batch(batch, aug2, aug_ratio2, npower=npower, device=device))
        return data_S, data_T

    @torch.no_grad()
    def evaluate(self, model, loader, hypergraph_hie_aware_dict, device, anchor_queue=None):
        model.eval()

        stat = torch.zeros((), device=device, dtype=torch.float64)
        total_correct = torch.zeros((), device=device, dtype=torch.long)
        total_loss_sup = stat
        total_loss_hse = stat.clone()
        total_loss_con = stat.clone()
        num_samples = len(loader.loader.dataset)
        num_batches = len(loader)

        compute_loss_con = (
            anchor_queue is not None
            and len(anchor_queue) == self.configs["num_anchors"]
        )
        anchors = anchor_queue if anchor_queue is not None else []
        S_anchors = None
        T_anchors = None
        if compute_loss_con:
            S_anchors = torch.stack([x[0] for x in anchors])
            T_anchors = torch.stack([x[1] for x in anchors])

        loader.new_epoch()
        for _ in range(num_batches):
            cpu_batch = loader.next()
            graph_ids = self._cpu_graph_ids(cpu_batch)
            num_graphs = int(cpu_batch.num_graphs)
            batch = self._batch_to_device(cpu_batch, device)
            H_S = self._batch_hypergraph(batch, hypergraph_hie_aware_dict, device, graph_ids)
            data_S, data_T = self._augment_views(batch, device, augment=False)
            S, T, out, loss_hse_S, loss_hse_T = self._forward_model(
                model, data_S, data_T, H_S)

            total_loss_sup = total_loss_sup + F.cross_entropy(out, batch.y).detach().double() * num_graphs
            total_loss_hse = total_loss_hse + (
                self.configs["weight_hse"] * (loss_hse_S + loss_hse_T)
            ).detach().double() * num_graphs
            if compute_loss_con and S_anchors is not None and T_anchors is not None:
                loss_con = model.loss_con(S, S_anchors, T, T_anchors)
                total_loss_con = total_loss_con + (
                    self.configs["weight_consistency"] * loss_con
                ).detach().double() * num_graphs
            pred = out.softmax(dim=-1).argmax(dim=-1)
            total_correct = total_correct + (pred == batch.y).sum()

        if num_samples == 0 or num_batches == 0:
            return 0.0, 0.0, 0.0, 0.0
        summed = torch.stack([
            total_loss_sup,
            total_loss_hse,
            total_loss_con,
            total_correct.double(),
        ]).detach().cpu().tolist()
        return (
            summed[3] / num_samples,
            summed[0] / num_samples,
            summed[1] / num_samples,
            summed[2] / num_samples if compute_loss_con else 0.0,
        )

    def _create_model(self, dataset: TUDatasetExt, avg_num_nodes: int, device: torch.device) -> HypSEE:
        cfg = self.configs
        model = HypSEE(
            in_channels=dataset.num_features,
            hidden_channels_gnn=cfg["dim_embedding_gnn"],
            hidden_channels=cfg["dim_embedding"],
            out_channels=dataset.num_classes,
            num_layers_gnn=cfg["num_layers_gnn"],
            num_edges2=cfg["num_edges2"],
            avg_num_nodes=avg_num_nodes,
            height=cfg["height"],
            decay_rate=cfg["decay_rate"],
            hgsl_constraint=cfg.get("hgsl_constraint", "sigmoid"),
            dropout=cfg.get("dropout", 0.5),
            cluster_assign=cfg.get("cluster_assign", "em"),
            drop_zero_incidence=bool(cfg.get("drop_zero_incidence", 0)),
        )
        model.reset_parameters()
        model = model.to(device)
        total_params = sum(param.numel() for param in model.parameters())
        trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f"param_count: total={total_params} trainable={trainable_params}")
        return model

    def _load_hypergraphs(self, dataset, seed=0, touch_cache=True):
        try:
            hypergraph_dict = load_hypergraphs(
                data_name=self.configs['data_name'],
                data_root=self.configs['data_root'],
                hyperedge_length_list=self.configs['hypergraph_length_list'],
                num_edges=self.configs['num_edges1'],
                num_graphs=len(dataset),
                seed=seed,
            )
        except FileNotFoundError:
            print("[info] precomputed hypergraphs not found; generating now...")
            for hyperedge_length in self.configs['hypergraph_length_list']:
                precompute_hypergraphs(
                    data_name=self.configs['data_name'],
                    data_root=self.configs['data_root'],
                    hyperedge_length=hyperedge_length,
                    num_edges=self.configs['num_edges1'],
                    seed=seed,
                )
            hypergraph_dict = load_hypergraphs(
                data_name=self.configs['data_name'],
                data_root=self.configs['data_root'],
                hyperedge_length_list=self.configs['hypergraph_length_list'],
                num_edges=self.configs['num_edges1'],
                num_graphs=len(dataset),
                seed=seed,
            )
        if touch_cache:
            cache = getattr(self, "_hg_dense_cache", None)
            if cache is not None:
                cache.set_version(int(seed))
        return hypergraph_dict

    def _forward_model(self, model, data_S, data_T, H_S):
        return model(data_S, data_T, H_S)

    def _init_anchor_queue(self, model, labeled_loader, hypergraph_hie_aware_dict, device):
        num_anchors = self.configs["num_anchors"]
        with torch.no_grad():
            model.eval()
            labeled_loader.new_epoch()
            anchor_queue = []
            while len(anchor_queue) < num_anchors:
                labeled_cpu = labeled_loader.next()
                labeled_ids = self._cpu_graph_ids(labeled_cpu)
                labeled_batch = self._batch_to_device(labeled_cpu, device)
                labeled_H_S = self._batch_hypergraph(
                    labeled_batch, hypergraph_hie_aware_dict, device, labeled_ids)
                labeled_data_S, labeled_data_T = self._augment_views(labeled_batch, device)
                S_labeled, T_labeled, _, _, _ = self._forward_model(
                    model, labeled_data_S, labeled_data_T, labeled_H_S)
                need = num_anchors - len(anchor_queue)
                for i in range(min(S_labeled.shape[0], need)):
                    anchor_queue.append((S_labeled[i].detach(), T_labeled[i].detach()))
            labeled_loader.new_epoch()
        assert len(anchor_queue) == num_anchors
        model.train()
        return anchor_queue

    @staticmethod
    def _batch_to_device(batch, device):
        if getattr(device, "type", None) == "cuda":
            return batch.to(device, non_blocking=True)
        return batch.to(device)

    def _dataloader_kwargs(self):
        num_workers = int(self.configs.get("num_workers", 0) or 0)
        pin_memory = bool(int(self.configs.get("pin_memory", 1))) and torch.cuda.is_available()
        kwargs = {
            "shuffle": False,
            "num_workers": max(num_workers, 0),
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            prefetch = int(self.configs.get("loader_prefetch", 2) or 2)
            kwargs["prefetch_factor"] = max(prefetch, 2)
            kwargs["persistent_workers"] = bool(int(self.configs.get("persistent_workers", 1)))
            kwargs["in_order"] = True
            kwargs["multiprocessing_context"] = "spawn"
        return kwargs

    def _iter_loader(self, dataset, batch_size):
        num_workers = int(self.configs.get("num_workers", 0) or 0)
        prefetch = int(self.configs.get("loader_prefetch", 2) or 0)
        if num_workers > 0:
            prefetch = 0
        return IterLoader(
            DataLoader(dataset, batch_size=batch_size, **self._dataloader_kwargs()),
            prefetch=prefetch,
        )

    def _build_data_loaders(self, dataset: TUDatasetExt):
        batch_size = self.configs["batch_size"]
        dataset = shuffle_dataset(dataset)
        labeled_ds = subset_dataset(dataset, slice(None, 0.1))
        unlabeled_ds = subset_dataset(dataset, slice(0.2, 0.7))
        val_ds = subset_dataset(dataset, slice(0.7, 0.8))
        test_ds = subset_dataset(dataset, slice(0.8, 1.0))

        labeled_loader = self._iter_loader(labeled_ds, int(np.ceil(batch_size / 5)))
        unlabeled_loader = self._iter_loader(unlabeled_ds, batch_size)
        val_loader = self._iter_loader(val_ds, batch_size)
        test_loader = self._iter_loader(test_ds, batch_size)
        return labeled_loader, unlabeled_loader, val_loader, test_loader, dataset

    def _checkpoint_configs(self):
        """Plain dict of configs suitable for torch.save / later reload."""
        cfg = self.configs
        if isinstance(cfg, dict):
            return {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
        return dict(vars(cfg))

    def _save_run_checkpoint(
        self,
        *,
        best_state,
        test_acc,
        best_epoch,
        seed,
        best_hypergraph_seed,
        exp_iter,
    ):
        data_name = self.configs["data_name"]
        ckpt_root = self.configs.get("checkpoint_dir") or "checkpoints"
        ckpt_dir = os.path.join(ckpt_root, data_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"run{exp_iter}_seed{seed}.pt")
        payload = {
            "model_state_dict": {k: v.detach().cpu() for k, v in best_state.items()},
            "test_acc": float(test_acc),
            "best_epoch": best_epoch,
            "seed": int(seed),
            "best_hypergraph_seed": (
                None if best_hypergraph_seed is None else int(best_hypergraph_seed)
            ),
            "exp_iter": int(exp_iter),
            "epoch_select": self.configs.get("epoch_select"),
            "checkpoint_dir": ckpt_root,
            "configs": self._checkpoint_configs(),
        }
        torch.save(payload, path)
        print(f"[checkpoint] saved {path} (test_acc={test_acc:.6f}, best_epoch={best_epoch})")
        return path

    def run(self, seed, device, exp_iter=0):
        self.fix_seed(seed)
        self._set_debug_ctx(seed=seed, epoch=None, phase="setup", model=None)

        self._hg_dense_cache = HypergraphDenseBatchCache()

        dataset = None
        model = None
        optimizer = None
        lr_scheduler = None
        hypergraph_hie_aware_dict = None
        anchor_queue = []
        labeled_loader = unlabeled_loader = val_loader = test_loader = None
        best_state = None
        result = None

        try:
            dataset = get_dataset(
                name=self.configs["data_name"],
                root=self.configs["data_root"],
                feat_str=self.configs["feat_str"],
            )

            data = dataset._data
            if data is None or data.x is None:
                raise RuntimeError("dataset node features are not initialized")
            avg_num_nodes = int(data.x.size(0) / len(dataset))

            labeled_loader, unlabeled_loader, val_loader, test_loader, dataset = (
                self._build_data_loaders(dataset))

            model = self._create_model(dataset, avg_num_nodes, device)
            self._set_debug_ctx(model=model)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.configs["lr"],
                weight_decay=self.configs["weight_decay"],
            )
            lr_decay = self.configs.get("lr_decay", 1.0)
            if lr_decay < 1.0:
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.configs["epochs"],
                    eta_min=self.configs["lr"] * lr_decay,
                )

            best_epoch = None
            best_val_loss_sup = 1e10
            best_val_loss_sup_se = 1e10
            best_val_acc = 0.0
            current_hypergraph_seed = None
            best_hypergraph_seed = None

            for epoch in range(1, self.configs["epochs"] + 1):
                self._set_debug_ctx(epoch=epoch)
                if epoch == 1:
                    self._set_debug_ctx(phase="load_hypergraphs")
                    current_hypergraph_seed = seed + epoch
                    hypergraph_hie_aware_dict = self._load_hypergraphs(
                        dataset, seed=current_hypergraph_seed)
                    assert hypergraph_hie_aware_dict is not None
                    self._set_debug_ctx(phase="init_anchor_queue")
                    anchor_queue = self._init_anchor_queue(
                        model, labeled_loader, hypergraph_hie_aware_dict, device)

                assert hypergraph_hie_aware_dict is not None
                self._set_debug_ctx(phase="train")
                train_loss, train_loss_sup, train_loss_con, train_loss_se = self.train(model, labeled_loader,
                                                                                  unlabeled_loader, hypergraph_hie_aware_dict,
                                                                                  optimizer, anchor_queue, epoch, device)
                self._set_debug_ctx(phase="evaluate")
                val_acc, val_loss_sup, val_loss_se, val_loss_con = self.evaluate(
                    model, val_loader, hypergraph_hie_aware_dict, device,
                    anchor_queue=anchor_queue)
                test_acc, test_loss_sup, test_loss_hse, test_loss_con = self.evaluate(
                    model, test_loader, hypergraph_hie_aware_dict, device,
                    anchor_queue=anchor_queue)
                print(
                    "epoch{}:\ttrain loss: {}\ttrain_sup loss: {}\ttrain_con loss: {}\ttrain_hse loss: {}\t"
                    "val_acc: {}\ttest_acc: {}"
                    .format(epoch, train_loss, train_loss_sup, train_loss_con, train_loss_se, val_acc, test_acc))
                if epoch > self.configs["warm_epochs"]:
                    epoch_select = self.configs["epoch_select"]
                    improved = False
                    score = 0.0
                    if epoch_select == 'val_loss_sup':
                        if best_state is None or val_loss_sup < best_val_loss_sup:
                            best_val_loss_sup = val_loss_sup
                            improved = True
                            score = val_loss_sup
                    elif epoch_select == 'val_acc':
                        if best_state is None or val_acc > best_val_acc:
                            best_val_acc = val_acc
                            improved = True
                            score = val_acc
                    elif epoch_select == 'val_acc_eq':
                        if best_state is None or val_acc >= best_val_acc:
                            best_val_acc = val_acc
                            improved = True
                            score = val_acc
                    elif epoch_select == 'val_loss_sup_hse':
                        val_loss_sup_hse = val_loss_sup + val_loss_se
                        if best_state is None or val_loss_sup_hse < best_val_loss_sup_se:
                            best_val_loss_sup_se = val_loss_sup_hse
                            improved = True
                            score = val_loss_sup_hse
                    else:
                        raise NotImplementedError(epoch_select)

                    if improved:
                        best_epoch = epoch
                        best_state = deepcopy(model.state_dict())
                        best_hypergraph_seed = current_hypergraph_seed
                        print("epoch{}:\t{}={}".format(best_epoch, epoch_select, score))

                if lr_scheduler is not None:
                    lr_scheduler.step()

            if best_state is not None:
                model.load_state_dict(best_state)
                if best_hypergraph_seed is not None:
                    hypergraph_hie_aware_dict = self._load_hypergraphs(
                        dataset, seed=best_hypergraph_seed)
                    anchor_queue = self._init_anchor_queue(
                        model, labeled_loader, hypergraph_hie_aware_dict, device)
            test_acc, test_loss_sup, test_loss_hse, test_loss_con = self.evaluate(
                model, test_loader, hypergraph_hie_aware_dict, device,
                anchor_queue=anchor_queue)
            print(
                "best_epoch:{}\ttest_acc:{}\ttest_loss_sup:{}\ttest_loss_con:{}\ttest_loss_hse:{}"
                .format(best_epoch, test_acc, test_loss_sup, test_loss_con, test_loss_hse))
            if self.configs.get("save_checkpoint") and best_state is not None:
                self._save_run_checkpoint(
                    best_state=best_state,
                    test_acc=test_acc,
                    best_epoch=best_epoch,
                    seed=seed,
                    best_hypergraph_seed=best_hypergraph_seed,
                    exp_iter=exp_iter,
                )
            result = test_acc
        except RuntimeError as err:
            if self._is_oom_error(err):
                self._oom_cleanup(optimizer)
                self._raise_oom_without_traceback(err)
            raise
        finally:
            self._run_finalize(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                anchor_queue=anchor_queue,
                best_state=best_state,
                hypergraph_hie_aware_dict=hypergraph_hie_aware_dict,
                dataset=dataset,
                labeled_loader=labeled_loader,
                unlabeled_loader=unlabeled_loader,
                val_loader=val_loader,
                test_loader=test_loader,
            )

        if result is None:
            raise RuntimeError("run completed without producing test metrics")
        return result

    def _oom_cleanup(self, optimizer=None):
        """Best-effort release after CUDA OOM before the exception propagates."""
        self._set_debug_ctx(phase="oom_cleanup")
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        self._release_cuda_memory()

    @staticmethod
    def _clear_hyper_encoder_caches(model):
        if model is None:
            return
        grl = getattr(model, "hyper_hierarchical_GRL", None)
        if grl is not None:
            grl.clu_mat.clear()
            grl.vol_dict.clear()
            grl.last_assignments = None
            grl.last_node_mask = None

    def _run_finalize(
        self,
        *,
        model,
        optimizer,
        lr_scheduler,
        anchor_queue,
        best_state,
        hypergraph_hie_aware_dict,
        dataset,
        labeled_loader,
        unlabeled_loader,
        val_loader,
        test_loader,
    ):
        """Release GPU-heavy references on every exit path (success, OOM, or other error)."""
        self._set_debug_ctx(model=None, phase="done")
        cache = getattr(self, "_hg_dense_cache", None)
        if cache is not None:
            cache.clear()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if isinstance(anchor_queue, list):
            anchor_queue.clear()
        self._clear_hyper_encoder_caches(model)
        del best_state, hypergraph_hie_aware_dict, dataset
        del model, optimizer, lr_scheduler
        del labeled_loader, unlabeled_loader, val_loader, test_loader
        self._release_cuda_memory()

    @staticmethod
    def _release_cuda_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()

    def cleanup(self):
        """Drop references held on the experiment and free GPU memory.

        Called by ``main`` in a ``finally`` block; safe to invoke after a failed run.
        """
        self._debug_ctx.clear()
        cache = getattr(self, "_hg_dense_cache", None)
        if cache is not None:
            cache.clear()
        self._release_cuda_memory()



    def exp(self):
        print(self.configs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.configs["data_name"], self.configs["data_root"])

        if self.configs["data_name"] == "COLLAB" and self.configs["feat_str"] == "":
            self.configs["feat_str"] = "deg+odeg100"

        test_acc_list = []
        seed = int(self.configs.get("seed", 0))
        exp_iter = 0
        runs = int(self.configs["runs"])
        while len(test_acc_list) < runs:
            try:
                test_acc = self.run(seed, device, exp_iter=exp_iter)
            except RuntimeError as err:
                if self._is_oom_error(err):
                    self._release_cuda_memory()
                    self._raise_oom_without_traceback(err)
                raise
            print(f"run: {len(test_acc_list)}\tacc: {test_acc}")
            test_acc_list.append(test_acc)
            exp_iter += 1
            seed += 1

        mean_acc = float(np.mean(test_acc_list))
        std_acc = float(np.std(test_acc_list))
        print(f"test_acc_mean: {mean_acc}\tstd: {std_acc}")
        return mean_acc
