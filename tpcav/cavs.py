#!/usr/bin/env python3
"""
CAV training and attribution utilities built on TPCAV.
"""

import logging
import os
import gc
from pathlib import Path
from typing import List, Optional, Tuple, Union
import time

from concurrent.futures import ProcessPoolExecutor
from Bio import motifs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from copy import deepcopy
from scipy import stats
import uuid
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from packaging.version import Version
import logomaker
import multiprocessing as mp

from . import helper, utils, report
from .concepts import ConceptBuilder
from .tpcav_model import TPCAV
from matplotlib import gridspec

logger = logging.getLogger(__name__)


class _TorchLinear(torch.nn.Module):
    """Torch linear layer classifier"""

    def __init__(self, input_dim, num_class=1, device='cuda:0'):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, num_class)
        self.device = device

    def forward(self, avs):
        return self.linear(avs).squeeze(-1)

    def fit(self, train_loader, val_loader,
            patience=10, lr=1e-2, weight_decay=1e-2, max_epochs=1000):

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        best_loss = None; best_state_dict = None
        epoch = 0; t = 0

        while True:
            epoch += 1
            if epoch > max_epochs: break

            self.train()
            for avs, l in train_loader:
                optimizer.zero_grad()
                l_hinge = (l.float() * 2 - 1).to(self.device)  # 0→-1, 1→1
                y_hat = self(avs.to(self.device))
                loss = torch.mean(torch.clamp(1 - l_hinge * y_hat, min=0))
                loss.backward()
                optimizer.step()

            logger.debug(f"Training loss at epoch {epoch}: {loss}")

            self.eval()
            val_losses = []
            with torch.no_grad():
                for avs, l in val_loader:
                    l_hinge = (l.float() * 2 - 1).to(self.device)
                    y_hat = self(avs.to(self.device))
                    val_losses.append(
                        torch.mean(torch.clamp(1 - l_hinge * y_hat, min=0)).item()
                    )

            val_loss_mean = np.mean(val_losses)
            if (best_loss is None) or (val_loss_mean < best_loss):
                best_loss = val_loss_mean
                best_state_dict = deepcopy(self.state_dict())
            else:
                t += 1
                if t >= patience: break

        return best_state_dict, best_loss

class _TorchLinearWrapper:
    def __init__(self, input_dim, num_class=1, lr=1e-2, weight_decay_search = [1e-2, 1e-4, 1e-6], device="cuda:0"):
        super().__init__()
        self.input_dim = input_dim
        self.num_class = num_class
        self.lr = lr
        self.weight_decay_search = weight_decay_search
        self.device = device

    def fit(self, train_loader, val_loader):
        best_state_dict = None; best_loss = None
        for w in self.weight_decay_search:
            model = _TorchLinear(self.input_dim, self.num_class, device=self.device).to(self.device)
            state_dict, loss = model.fit(train_loader, val_loader, lr=self.lr, weight_decay=w)
            if (best_loss is None) or (loss < best_loss):
                best_loss = loss; best_state_dict = state_dict
            del model; gc.collect(); torch.cuda.empty_cache()

        self.best_model = _TorchLinear(self.input_dim, self.num_class)
        self.best_model.load_state_dict(best_state_dict)
        self.best_model.to(self.device)

    def predict(self, avs: np.ndarray, batch_size=128):
        self.best_model.eval()
        y_hats = []
        with torch.no_grad():
            for i in range(0, len(avs), batch_size):
                batch = torch.from_numpy(avs[i:i+batch_size]).to(self.device)
                y_hats.append(self.best_model(batch).cpu())
        y_hats = torch.cat(y_hats, dim=0)
        return (y_hats >= 0).long().numpy()
    
    @property
    def weights(self):
        linear_weight = self.best_model.linear.weight.detach().cpu()[0]
        return torch.stack([-1 * linear_weight, linear_weight])

    @property
    def classes_(self):
        return self.num_class

class _SGDWrapper:
    """Lightweight SGD concept classifier."""

    def __init__(self, penalty: str = "l2", n_jobs: int = -1):
        self.lm = SGDClassifier(
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            learning_rate="optimal",
            n_iter_no_change=10,
            n_jobs=n_jobs,
            penalty=penalty,
            average=True,
        )
        if penalty == "l2":
            params = {"alpha": [1e-2, 1e-4, 1e-6, 1e-8]}
        elif penalty == "l1":
            params = {"alpha": [1e-1, 1]}
        else:
            raise ValueError(f"Unexpected penalty type {penalty}")
        self.search = GridSearchCV(self.lm, params)

    def fit(self, train_avs: np.ndarray, train_ls: np.ndarray):
        self.search.fit(train_avs, train_ls)
        self.lm = self.search.best_estimator_
        logger.info(
            "Best Params: %s | Iterations: %s",
            self.search.best_params_,
            self.lm.n_iter_,
        )

    @property
    def weights(self) -> torch.Tensor:
        if len(self.lm.coef_) == 1:
            return torch.tensor(np.array([-1 * self.lm.coef_[0], self.lm.coef_[0]]))
        return torch.tensor(self.lm.coef_)

    @property
    def classes_(self):
        return self.lm.classes_

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.lm.predict(x)

def prepare_split(concept_path, control_path, seed=42, test_size=0.1, val_size=0.1):
    """Compute train/val/test index splits without materializing embeddings.

    Indices address a virtual array of [concept_rows..., control_rows...].
    Returns (train_idx, val_idx, test_idx, n_concept).
    """
    n_concept = len(np.load(str(concept_path), mmap_mode="r"))
    n_control = len(np.load(str(control_path), mmap_mode="r"))

    y = np.concatenate([np.zeros(n_concept), np.ones(n_control)]).astype(int)
    all_idx = np.arange(n_concept + n_control)

    train_val_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, stratify=y, random_state=seed
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (1 - test_size),
        stratify=y[train_val_idx],
        random_state=seed,
    )
    return train_idx, val_idx, test_idx, n_concept


def _load_subset(concept_path, control_path, idx, n_concept):
    """Materialize a subset of embeddings as a numpy array.

    concept embeddings → label 0, control embeddings → label 1.
    """
    concept = np.load(str(concept_path), mmap_mode="r")
    control = np.load(str(control_path), mmap_mode="r")
    feat_dim = int(np.prod(concept.shape[1:]))
    concept = concept.reshape(len(concept), feat_dim)
    control = control.reshape(len(control), feat_dim)

    mask = idx < n_concept
    X = np.empty((len(idx), feat_dim), dtype=concept.dtype)
    if mask.any():
        X[mask] = concept[idx[mask]]
    if (~mask).any():
        X[~mask] = control[idx[~mask] - n_concept]

    y = np.where(mask, 0, 1).astype(np.int64)
    return X, y


class EmbeddingDataset(torch.utils.data.Dataset):
    """Lazy dataset backed by two memmap .npy files.

    concept embeddings → label 0, control embeddings → label 1.
    Only the requested rows are read from disk per __getitem__ call.
    """

    def __init__(self, concept_path: str, control_path: str, idx: np.ndarray, n_concept: int):
        concept = np.load(str(concept_path), mmap_mode="r")
        control = np.load(str(control_path), mmap_mode="r")
        feat_dim = int(np.prod(concept.shape[1:]))
        self._concept   = concept.reshape(len(concept), feat_dim)
        self._control   = control.reshape(len(control), feat_dim)
        self._idx       = idx
        self._n_concept = n_concept

    def __len__(self):
        return len(self._idx)

    def __getitem__(self, i):
        global_idx = self._idx[i]
        if global_idx < self._n_concept:
            x = self._concept[global_idx].copy()
            y = 0
        else:
            x = self._control[global_idx - self._n_concept].copy()
            y = 1
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)

def _train(
    concept_embeddings: str,
    control_embeddings: str,
    output_dir: str,
    penalty: str = "l2",
    backend: str = "sklearn",
    device=None,
    name=None,
) -> Tuple[float, torch.Tensor]:
    """Train a binary CAV classifier for a concept vs cached control embeddings."""
    assert backend in ["sklearn", "torch"], "Backend has to be either sklearn or torch!"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx, test_idx, n_concept = prepare_split(concept_embeddings, control_embeddings)

    if backend == "sklearn":
        X_train, y_train = _load_subset(concept_embeddings, control_embeddings, train_idx, n_concept)
        X_test,  y_test  = _load_subset(concept_embeddings, control_embeddings, test_idx,  n_concept)
        clf = _SGDWrapper(penalty=penalty)
        clf.fit(X_train, y_train)
    else:
        concept_mmap = np.load(str(concept_embeddings), mmap_mode="r")
        feat_dim = int(np.prod(concept_mmap.shape[1:]))
        del concept_mmap

        device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        pin = torch.cuda.is_available()

        train_ds = EmbeddingDataset(concept_embeddings, control_embeddings, train_idx, n_concept)
        val_ds   = EmbeddingDataset(concept_embeddings, control_embeddings, val_idx,   n_concept)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True,  pin_memory=pin, num_workers=0)
        val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=32, shuffle=False, pin_memory=pin, num_workers=0)

        clf = _TorchLinearWrapper(input_dim=feat_dim, device=device)
        clf.fit(train_loader, val_loader)

        X_train, y_train = _load_subset(concept_embeddings, control_embeddings, train_idx, n_concept)
        X_test,  y_test  = _load_subset(concept_embeddings, control_embeddings, test_idx,  n_concept)

    def _eval(X, y, split_name: str):
        y_preds = clf.predict(X)
        acc = (y_preds == y).sum() / len(y)
        _, _, fscore, _ = precision_recall_fscore_support(
            y, y_preds, average="binary", pos_label=1
        )
        (output_dir / f"classifier_perform_on_{split_name}.txt").write_text(
            f"Accuracy: {acc}\n"
        )
        return fscore

    train_fscore = _eval(X_train, y_train, "train")
    test_fscore  = _eval(X_test,  y_test,  "test")
    logger.info("Concept %s: [train] F-score: %.4f, [test] F-score: %.4f", name, train_fscore, test_fscore)

    weights = clf.weights
    assert len(weights.shape) == 2 and weights.shape[0] == 2
    torch.save(weights, output_dir / "classifier_weights.pt")

    if backend == 'torch':
        del clf.best_model
        gc.collect()
        torch.cuda.empty_cache()

    return test_fscore, weights[0]


class CavTrainer:
    """Train CAVs and compute attribution-driven TCAV scores."""

    def __init__(self, tpcav: TPCAV, penalty: str = "l2") -> None:
        self.tpcav = tpcav
        self.penalty = penalty
        self.cav_fscores = {}
        self.cav_weights = {}
        self.control_embeddings: Optional[torch.Tensor] = None
        self.cavs_list: List[torch.Tensor] = []

    def save_state(self, output_path: str = "cav_trainer_state.pt"):
        """
        Save CavTrainer state to a file.
        """
        state = {
            "penalty": self.penalty,
            "cav_fscores": self.cav_fscores,
            "cav_weights": self.cav_weights,
            "control_embeddings": self.control_embeddings,
            "cavs_list": self.cavs_list,
        }
        torch.save(state, output_path)
    
    @staticmethod
    def load_state(tpcav_model: TPCAV, state_path: str = "cav_trainer_state.pt"):
        """
        Restore CavTrainer state from a file.
        """
        state = torch.load(state_path, map_location="cpu")
        cav_trainer = CavTrainer(tpcav_model, penalty=state["penalty"])

        cav_trainer.cav_fscores = state["cav_fscores"]
        cav_trainer.cav_weights = state["cav_weights"]
        cav_trainer.control_embeddings = state["control_embeddings"]
        cav_trainer.cavs_list = state["cavs_list"]

        logger.info("Successfully restored cav trainer states!")
        return cav_trainer

    def set_control(self, control_concept, num_samples: int) -> torch.Tensor:
        """
        Set and cache control embeddings to avoid recomputation across CAV trainings.
        """
        self.control_embeddings = self.tpcav.concept_embeddings(
            control_concept, num_samples=num_samples
        )
        return self.control_embeddings

    @staticmethod
    def _save_tensor_npy(path: Path, tensor: torch.Tensor) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, tensor.detach().cpu().numpy())
        return str(path)

    @staticmethod
    def _cleanup_paths(paths: list[str]) -> None:
        for p in paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass

    @classmethod
    def _reap_done_futures(cls, futures: list, results: list):
        pending = []
        for name, fut, paths in futures:
            if fut.done():
                results.append((name, fut.result()))  # raises if worker failed
                cls._cleanup_paths(paths)
            else:
                pending.append((name, fut, paths))
        return pending

    @classmethod
    def _wait_for_capacity(
        cls,
        futures: list,
        results: list,
        capacity: int,
        sleep_s: int = 5,
    ):
        while True:
            futures = cls._reap_done_futures(futures, results)
            if len(futures) < capacity:
                return futures
            time.sleep(sleep_s)

    def train_concepts(
        self,
        concept_list,
        num_samples: int,
        output_dir: str,
        num_processes: int = 1,
        max_pending: int = 8,
        backend='sklearn',
        device=None
    ):
        "Train concepts with a fixed control set by self.set_control()"
        if self.control_embeddings is None:
            raise ValueError(
                "Call set_control(control_concept, num_samples=...) before training CAVs."
            )
        else:
            self.control_embeddings = self.control_embeddings.cpu()

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        control_memmap_path = output_dir_path / f"_control_embeddings_{uuid.uuid4().hex}.npy"
        self._save_tensor_npy(control_memmap_path, self.control_embeddings)

        if num_processes == 1:
            for c in concept_list:
                concept_embeddings = self.tpcav.concept_embeddings(
                    c, num_samples=num_samples
                )
                concept_dir = output_dir_path / c.name
                concept_dir.mkdir(parents=True, exist_ok=True)
                concept_memmap_path = concept_dir / "concept_embeddings.npy"
                self._save_tensor_npy(concept_memmap_path, concept_embeddings)
                fscore, weight = _train(
                    str(concept_memmap_path),
                    str(control_memmap_path),
                    concept_dir,
                    self.penalty,
                    backend=backend,
                    device=device,
                    name=c.name,
                )
                self.cav_fscores[c.name] = fscore
                self.cav_weights[c.name] = weight
                self.cavs_list.append(weight)

                self._cleanup_paths([str(concept_memmap_path)])
        else:
            futures = []; results = []
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(mp_context=ctx, max_workers=num_processes) as executor:
                for c in concept_list:
                    concept_embeddings = self.tpcav.concept_embeddings(
                        c, num_samples=num_samples
                    )

                    concept_dir = output_dir_path / c.name
                    concept_dir.mkdir(parents=True, exist_ok=True)
                    concept_memmap_path = concept_dir / "concept_embeddings.npy"
                    self._save_tensor_npy(concept_memmap_path, concept_embeddings)

                    # block the process to avoid too long queue
                    futures = self._wait_for_capacity(
                        futures, results, capacity=(max_pending + num_processes), sleep_s=5
                    )

                    future = executor.submit(
                        _train,
                        str(concept_memmap_path),
                        str(control_memmap_path),
                        concept_dir,
                        self.penalty,
                        backend=backend,
                        device=device,
                        name=c.name,
                    )
                    logger.info("Submitted CAV training for concept %s", c.name)
                    futures.append((c.name, future, [str(concept_memmap_path)]))

                for name, fut, paths in futures:
                    results.append((name, fut.result()))
                    self._cleanup_paths(paths)
            for name, (fscore, weight) in results:
                self.cav_fscores[name] = fscore
                self.cav_weights[name] = weight
                self.cavs_list.append(weight)

        self._cleanup_paths([str(control_memmap_path)])

    def train_concepts_pairs(self,
                             concept_pair_list,
                             num_samples: int,
                             output_dir: str,
                             num_processes: int = 1,
                             max_pending: int = 8,
                             backend='sklearn',
                             device=None):
        """Train concept pairs (test concept, control concept)

        Note: It would compute embeddings on every control concept, use self.train_concepts if control concept is fixed
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        if num_processes == 1:
            for c_test, c_control in concept_pair_list:
                concept_embeddings = self.tpcav.concept_embeddings(
                    c_test, num_samples=num_samples
                )
                control_embeddings = self.tpcav.concept_embeddings(
                    c_control, num_samples=num_samples
                )

                concept_dir = output_dir_path / c_test.name
                concept_dir.mkdir(parents=True, exist_ok=True)
                concept_memmap_path = concept_dir / "concept_embeddings.npy"
                control_memmap_path = concept_dir / "control_embeddings.npy"
                self._save_tensor_npy(concept_memmap_path, concept_embeddings)
                self._save_tensor_npy(control_memmap_path, control_embeddings)

                fscore, weight = _train(
                    str(concept_memmap_path),
                    str(control_memmap_path),
                    concept_dir,
                    self.penalty,
                    backend=backend,
                    device=device,
                    name=c_test.name,
                )
                self.cav_fscores[c_test.name] = fscore
                self.cav_weights[c_test.name] = weight
                self.cavs_list.append(weight)

                self._cleanup_paths([str(concept_memmap_path), str(control_memmap_path)])
        else:
            futures = []; results = []
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                for c_test, c_control in concept_pair_list:
                    concept_embeddings = self.tpcav.concept_embeddings(
                        c_test, num_samples=num_samples
                    )
                    control_embeddings = self.tpcav.concept_embeddings(
                        c_control, num_samples=num_samples
                    )

                    concept_dir = output_dir_path / c_test.name
                    concept_dir.mkdir(parents=True, exist_ok=True)
                    concept_memmap_path = concept_dir / "concept_embeddings.npy"
                    control_memmap_path = concept_dir / "control_embeddings.npy"
                    self._save_tensor_npy(concept_memmap_path, concept_embeddings)
                    self._save_tensor_npy(control_memmap_path, control_embeddings)

                    # block the process to avoid too long queue
                    futures = self._wait_for_capacity(
                        futures, results, capacity=(max_pending + num_processes), sleep_s=5
                    )

                    future = executor.submit(
                        _train,
                        str(concept_memmap_path),
                        str(control_memmap_path),
                        concept_dir,
                        self.penalty,
                        backend=backend,
                        device=device,
                        name=c_test.name,
                    )
                    logger.info("Submitted CAV training for concept %s", c_test.name)
                    futures.append(
                        (
                            c_test.name,
                            future,
                            [str(concept_memmap_path), str(control_memmap_path)],
                        )
                    )

                for name, fut, paths in futures:
                    results.append((name, fut.result()))
                    self._cleanup_paths(paths)
            for name, (fscore, weight) in results:
                self.cav_fscores[name] = fscore
                self.cav_weights[name] = weight
                self.cavs_list.append(weight)


    def tpcav_score(
        self, concept_name: str, attributions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute a simple TCAV score: mean directional attribution along the concept CAV.
        """
        if concept_name not in self.cav_weights:
            raise ValueError(f"No CAV weights stored for concept {concept_name}")
        weights = self.cav_weights[concept_name]
        flat_attr = attributions.flatten(start_dim=1)
        scores = torch.matmul(flat_attr, weights.to(flat_attr.device).unsqueeze(-1))

        return scores

    def tpcav_score_all_concepts(self, attributions: torch.Tensor) -> dict:
        """
        Compute TCAV scores for all trained concepts.
        """
        scores_dict = {}
        for concept_name in self.cav_weights.keys():
            scores = self.tpcav_score(concept_name, attributions)
            scores_dict[concept_name] = scores
        return scores_dict

    def tpcav_score_binary_log_ratio(
        self, concept_name: str, attributions: torch.Tensor, pseudocount: float = 1.0
    ) -> float:
        """
        Compute TCAV log ratio score: log2 of ratio of positive to negative directional attributions.
        """
        scores = self.tpcav_score(concept_name, attributions)

        pos_count = (scores > 0).sum().item()
        neg_count = (scores < 0).sum().item()

        return np.log((pos_count + pseudocount) / (neg_count + pseudocount))

    def tpcav_score_all_concepts_log_ratio(
        self, attributions: torch.Tensor, pseudocount: float = 1.0
    ) -> dict:
        """
        Compute TCAV log ratio scores for all trained concepts.
        """
        log_ratio_dict = {}
        for concept_name in self.cav_weights.keys():
            log_ratio = self.tpcav_score_binary_log_ratio(
                concept_name, attributions, pseudocount
            )
            log_ratio_dict[concept_name] = log_ratio
        return log_ratio_dict

    def plot_cavs_similaritiy_heatmap(
        self,
        attributions: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        concept_list: Optional[List[str]] = None,
        fscore_thresh=0.8,
        motif_meme_file: Optional[str] = None,
        output_path: str = "cavs_similarity_heatmap.png",
    ):
        if concept_list is None:
            cavs_names = list(self.cav_weights.keys())
        else:
            cavs_names = concept_list
        cavs_pass = []
        cavs_names_pass = []
        for cname in cavs_names:
            if self.cav_fscores[cname] >= fscore_thresh:
                cavs_pass.append(self.cav_weights[cname].cpu().numpy())
                cavs_names_pass.append(cname)
            else:
                logger.info(
                    "Skipping CAV %s with F-score %.3f below threshold %.3f",
                    cname,
                    self.cav_fscores[cname],
                    fscore_thresh,
                )
        if len(cavs_pass) == 0:
            logger.warning(f"No CAVs passed the F-score threshold {fscore_thresh:.3f}.")
            return

        # compute similarity matrix
        matrix_similarity = cosine_similarity(cavs_pass)

        # plot
        cm = sns.clustermap(
            matrix_similarity,
            xticklabels=False,
            yticklabels=False if attributions is not None else cavs_names_pass,
            cmap="bwr",
            vmin=-1,
            vmax=1,
        )
        cm.gs.update(left=0, right=1)
        cm.ax_cbar.set_position([0.01, 0.9, 0.05, 0.05])

        cavs_names_sorted = [
            cavs_names_pass[i] for i in cm.dendrogram_col.reordered_ind
        ]

        heatmap_bbox = cm.ax_heatmap.get_position()
        ax_logs = []
        log_ratios_by_attr = None
        if attributions is not None:
            log_ratios_by_attr = {}
            attributions = (
                attributions
                if isinstance(attributions, (list, tuple))
                else [
                    attributions,
                ]
            )
            for i, attrs in enumerate(attributions):
                offset =  1 + i*0.2
                ## plot log ratio plot
                ax_log = cm.figure.add_subplot()
                ax_log.set_position([offset, heatmap_bbox.y0, 0.2, heatmap_bbox.height])

                log_ratios_reordered = [
                    self.tpcav_score_binary_log_ratio(cname, attrs)
                    for cname in cavs_names_sorted
                ]
                log_ratios_by_attr[i] = log_ratios_reordered
                sns.barplot(y=cavs_names_sorted, x=log_ratios_reordered, orient="y", ax=ax_log)
                # set color of bar by value
                for idx in range(len((ax_log.containers[0]))):
                    if ax_log.containers[0].datavalues[idx] > 0:
                        ax_log.containers[0][idx].set_color("red")
                    else:
                        ax_log.containers[0][idx].set_color("blue")

                ax_log.set_xlim(left=-5, right=5)
                ax_log.yaxis.tick_right()
                ax_log.set_title(f"TCAV log ratio {i}")
                ax_log.get_yaxis().set_visible(False)
                ax_logs.append(ax_log)
            ax_logs[-1].get_yaxis().set_visible(True) # only show yaxis labels in the last log ratio plot
        
        # plot motif logo if provided meme file, try to look for pwm for every concept in the file
        if motif_meme_file is not None:
            if attributions is not None:
                ax_logs[-1].tick_params(
                   axis="y", which="major", pad=cm.figure.get_size_inches()[0] * 0.3 * 72 # leave space for motif logos
                )
            else:
                cm.ax_heatmap.tick_params(
                   axis="y", which="major", pad=cm.figure.get_size_inches()[0] * 0.3 * 72 # leave space for motif logos
                )
            gs_logo = gridspec.GridSpec(len(cavs_names_pass), 1)

            logo_height = heatmap_bbox.height/len(cavs_names_pass)
            for i, (cav_key, g) in enumerate(zip(cavs_names_sorted[::-1], gs_logo)):
                ax_logo = plt.subplot(g)
                ax_logo.set_position([1+len(ax_logs)*0.2+0.01, heatmap_bbox.y0+i*logo_height, 0.3+0.01, logo_height])
                if cav_key is not None:
                    seq_logo(cav_key, motif_meme_file=motif_meme_file, ax=ax_logo)
                else:
                    ax_logo.axis('off')

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        row_reordered_ind = list(cm.dendrogram_row.reordered_ind)
        col_reordered_ind = list(cm.dendrogram_col.reordered_ind)
        matrix_similarity_sorted = matrix_similarity[np.ix_(row_reordered_ind, col_reordered_ind)]
        return {
            "concept_names": cavs_names_pass,
            "matrix_similarity": matrix_similarity,
            "row_reordered_ind": row_reordered_ind,
            "col_reordered_ind": col_reordered_ind,
            "concept_names_sorted_rows": cavs_names_sorted,
            "concept_names_sorted_cols": cavs_names_sorted,
            "matrix_similarity_sorted": matrix_similarity_sorted,
            "log_ratios_by_attr": log_ratios_by_attr,
            "output_path": str(output_path),
        }

def seq_logo(key, motif_meme_file, ax, max_len=20):
    "plot a pwm logo"
    with open(motif_meme_file) as handle:
        motif_pwms = motifs.parse(handle, fmt='MINIMAL')
        motif_pwms = {utils.clean_motif_name(m.name): m.pwm for m in motif_pwms} 
        pwm = motif_pwms.get(key)

        if pwm is not None:
            pwm_df = pd.DataFrame(pwm)
            motif_len = len(pwm_df)
            
            # Logomaker expects columns as A,C,G,T, so ensure correct order
            pwm_df = pwm_df[['A', 'C', 'G', 'T']]
        
            # Compute information content at each position (2 - entropy)
            def compute_ic(pwm_row):
                entropy = -sum([p * np.log2(p) if p > 0 else 0 for p in pwm_row])
                return 2 - entropy
            
            # Compute IC matrix: IC_letter = p * IC_total
            ic_df = pwm_df.copy()
            for i in range(len(pwm_df)):
                ic_total = compute_ic(pwm_df.iloc[i])
                ic_df.iloc[i] = pwm_df.iloc[i] * ic_total
        
            
            # Plot with logomaker
            x0, y0, width, height = ax.get_position().bounds
            ax.set_position([x0, y0, width * min(1., motif_len/max_len), height])
            logo = logomaker.Logo(ic_df, color_scheme={'A': 'red', 'C': 'blue',
                                                      'G': 'orange', 'T': 'green'},
                                  ax=ax)
            logo.ax.axis('off')
        else:
            logger.info(f'PWM not found for {key}')
            ax.axis('off')

def load_motifs_from_meme(motif_meme_file):
    return {utils.clean_motif_name(m.name): m for m in motifs.parse(open(motif_meme_file), fmt="MINIMAL")}

def load_motifs_from_custom_motif(motif_file):
    df = pd.read_table(motif_file, names=["motif_name", "consensus_seq"])
    motifs_dict = {}
    for motif_name in np.unique(df.motif_name):
        motif_name = utils.clean_motif_name(motif_name)
        consensus = [s.upper() for s in df.loc[df.motif_name == motif_name, "consensus_seq"].tolist()]
        motifs_dict[motif_name] = consensus

    return motifs_dict

def plot_reg(data, x, y, ax=None):
    ax = sns.regplot(data=data, x=x, y=y, ax=ax)
    res = stats.linregress(data[x], data[y])
    ax.text(0.05, 0.9, f"R^2: {res.rvalue**2:.4f}\nP value:  {res.pvalue}", transform=ax.transAxes)
    return res

def compute_motif_auc_fscore(num_motif_insertions: List[int], cav_trainers: List[CavTrainer], motif_file: Optional[str] = None,
                             motif_file_fmt: str = 'meme', output_path: Optional[str]=None):
    
    assert motif_file_fmt in ['meme', 'consensus']

    cavs_fscores_df = pd.DataFrame({f"fscore_{nm}_insertions": cav_trainer.cav_fscores for nm, cav_trainer in zip(num_motif_insertions, cav_trainers)})
    cavs_fscores_df['concept'] = list(cav_trainers[0].cav_fscores.keys())

    def compute_auc_fscore(row):
        y = [row[f"fscore_{nm}_insertions"] for nm in num_motif_insertions]
        if Version(np.__version__) < Version("2.0.0"): 
            return np.trapz(y, num_motif_insertions) / (
                num_motif_insertions[-1] - num_motif_insertions[0]
            )
        else:
            return np.trapezoid(y, num_motif_insertions) / (
                num_motif_insertions[-1] - num_motif_insertions[0]
            )

    cavs_fscores_df["AUC_fscores"] = cavs_fscores_df.apply(compute_auc_fscore, axis=1)

    # if motif instances are provided, fit linear regression curve to remove the dependency of f-scores on either information_content_GC or motif length and motif gc
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15, 4))
    if motif_file is not None:
        if motif_file_fmt == 'meme':
            motifs_dict = load_motifs_from_meme(motif_file)
            def load_meme_motif_info(key):
                m = motifs_dict[key]
                return (len(m.consensus), m.relative_entropy.sum(), np.dot(np.array(m.pwm['G']) + np.array(m.pwm['C']), m.relative_entropy))
            cavs_fscores_df[['motif_len', 'information_content', 'information_content_GC']] = cavs_fscores_df.apply(lambda x: load_meme_motif_info(x['concept']), axis=1, result_type='expand')
            
            model = LinearRegression()
            model.fit(cavs_fscores_df[['information_content_GC',]].to_numpy(), cavs_fscores_df['AUC_fscores'].to_numpy()[:, np.newaxis])
            
            y_pred = model.predict(cavs_fscores_df[['information_content_GC',]].to_numpy())
            residuals = cavs_fscores_df['AUC_fscores'].to_numpy() - y_pred.flatten()
            cavs_fscores_df['Motif_concept_sensitivity_score (AUC_fscores_residual)'] = residuals
            plot_reg(data=cavs_fscores_df, x='information_content_GC', y='AUC_fscores', ax=axes[0])
            plot_reg(data=cavs_fscores_df, x='information_content', y='AUC_fscores', ax=axes[1])
            plot_reg(data=cavs_fscores_df, x='motif_len', y='AUC_fscores', ax=axes[2])
        else:
            motifs_dict = load_motifs_from_custom_motif(motif_file)
            def load_custom_motif_info(key):
                consensus_seqs = motifs_dict[key]
                avg_len = np.mean([len(s) for s in consensus_seqs])
                avg_gc = np.mean([s.count('G') + s.count('C') for s in consensus_seqs])
                return (avg_len, avg_gc)

            cavs_fscores_df[['avg_len', 'avg_gc']] = cavs_fscores_df.apply(lambda x: load_custom_motif_info(x['concept']), axis=1, result_type='expand')
            
            model = LinearRegression()
            model.fit(cavs_fscores_df[['avg_gc',]].to_numpy(), cavs_fscores_df['AUC_fscores'].to_numpy()[:, np.newaxis])
            
            y_pred = model.predict(cavs_fscores_df[['avg_gc',]].to_numpy())
            residuals = cavs_fscores_df['AUC_fscores'].to_numpy() - y_pred.flatten()
            cavs_fscores_df['Motif_concept_sensitivity_score (AUC_fscores_residual)'] = residuals
            plot_reg(data=cavs_fscores_df, x='avg_len', y='AUC_fscores', ax=axes[0])
            plot_reg(data=cavs_fscores_df, x='avg_gc', y='AUC_fscores', ax=axes[1])
            axes[2].axis('off')

        cavs_fscores_df.sort_values("Motif_concept_sensitivity_score (AUC_fscores_residual)", ascending=False, inplace=True)
    else:
        cavs_fscores_df.sort_values("AUC_fscores", ascending=False, inplace=True)
    
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return cavs_fscores_df

def run_tpcav(
    model,
    motif_file: Union[str, List[str]],
    genome_fasta: str,
    motif_file_fmt: str = 'meme',
    num_motif_insertions: List[int] = [4, 8, 16],
    motif_control_type="random",
    bed_seq_file: Optional[str] = None,
    bed_chrom_file: Optional[str] = None,
    synthetic_gc_concept_step: Optional[float] = None,
    layer_name: Optional[str]=None,
    layer=None,
    output_dir: str = "tpcav/",
    num_samples_for_pca=10,
    num_samples_for_cav=1000,
    input_window_length=1024,
    batch_size=8,
    num_workers=0,
    bws=None,
    input_transform_func=helper.fasta_chrom_to_one_hot_seq,
    num_pc: Union[str,int]='full',
    p=1, 
    max_pending_jobs=4,
    save_cav_trainer=True,
    generate_html_report=True,
    html_report_fscore_thresh=0.9,
    seed=1001,
    backend='sklearn',
    device=None,
):
    """
    One-stop function to compute CAVs on motif concepts and bed concepts, compute AUC of motif concept f-scores after correction
    """
    assert motif_control_type in ["random", "permute"], "motif_control_type has to be one of [random, permute]!"
    assert motif_file_fmt in ['meme', 'consensus'], "motif_file_fmt has to be one of [meme, consensus]!"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # merge motif files if there are multiple
    if isinstance(motif_file, list):
        if motif_file_fmt == 'meme':
            motif_file = utils.merge_meme_files(motif_file)
        else:
            motif_file = utils.merge_consensus_motif_files(motif_file)

    output_path = Path(output_dir)
    # create concept builder to generate concepts
    ## motif concepts
    motif_concepts_pairs = {}
    motif_concept_builders = {}
    num_motif_insertions.sort()
    for nm in num_motif_insertions:
        builder = ConceptBuilder(
            genome_fasta=genome_fasta,
            input_window_length=input_window_length,
            bws=bws,
            num_motifs=nm,
            include_reverse_complement=True,
            min_samples=num_samples_for_cav,
            batch_size=batch_size,
            num_workers=num_workers,
            rng_seed = seed,
        )
        # use random regions as control
        builder.build_control()

        if motif_file_fmt == 'meme':
            # use meme motif PWMs to build motif concepts, one concept per motif
            # each pair of concepts include the motif concept and the permuted motif concept
            concepts_pairs = builder.add_meme_motif_concepts(str(motif_file))
        else:
            concepts_pairs = builder.add_custom_motif_concepts(str(motif_file))

        # apply transform to convert fasta sequences to one-hot encoded sequences
        builder.apply_transform(input_transform_func)

        motif_concepts_pairs[nm] = concepts_pairs
        motif_concept_builders[nm] = builder

    ## bed concepts (optional)
    if bed_seq_file is not None or bed_chrom_file is not None or synthetic_gc_concept_step is not None:
        non_motif_concept_builder = ConceptBuilder(
            genome_fasta=genome_fasta,
            input_window_length=input_window_length,
            bws=bws,
            num_motifs=0,
            include_reverse_complement=True,
            min_samples=num_samples_for_cav,
            batch_size=batch_size,
            rng_seed = seed,
        )
        # use random regions as control
        non_motif_concept_builder.build_control()
        if bed_seq_file is not None:
            # build concepts from fasta sequences in bed file
            non_motif_concept_builder.add_bed_sequence_concepts(bed_seq_file)
        if bed_chrom_file is not None:
            # build concepts from chromatin tracks in bed file
            non_motif_concept_builder.add_bed_chrom_concepts(bed_chrom_file)
        if synthetic_gc_concept_step is not None:
            # build synthetic gc content concepts
            non_motif_concept_builder.add_synthetic_gc_content_concepts(synthetic_gc_concept_step)
        # apply transform to convert fasta sequences to one-hot encoded sequences
        non_motif_concept_builder.apply_transform(input_transform_func)
    else:
        non_motif_concept_builder = None

    # create TPCAV model on top of the given model
    tpcav_model = TPCAV(model, layer_name=layer_name, layer=layer)
    # fit PCA on sampled all concept activations of the last builder (should have the most motifs)
    tpcav_model.fit_pca(
        concepts=motif_concept_builders[num_motif_insertions[-1]].concepts_for_pca() + non_motif_concept_builder.concepts_for_pca() if non_motif_concept_builder is not None else motif_concept_builders[num_motif_insertions[-1]].concepts_for_pca(),
        num_samples_per_concept=num_samples_for_pca,
        num_pc=num_pc,
    )
    #torch.save(tpcav_model, output_path / "tpcav_model.pt")

    # create trainer for computing CAVs
    motif_cav_trainers = {}
    for nm in num_motif_insertions:
        cav_trainer = CavTrainer(tpcav_model, penalty="l2")
        if motif_control_type == 'permute':
            cav_trainer.train_concepts_pairs(motif_concepts_pairs[nm], 
                                             num_samples_for_cav, 
                                             output_dir=str(output_path / f"cavs_{nm}_motifs/"),
                                             num_processes=p, max_pending=max_pending_jobs, backend=backend, device=device)
        else:
            cav_trainer.set_control(motif_concept_builders[nm].control_concepts[0], num_samples=num_samples_for_cav)
            cav_trainer.train_concepts([c for c, _ in motif_concepts_pairs[nm]],
                                        num_samples_for_cav,
                                        output_dir=str(output_path / f"cavs_{nm}_motifs/"),
                                        num_processes=p, max_pending=max_pending_jobs, backend=backend, device=device)
        if save_cav_trainer:
            torch.save(cav_trainer, str(output_path / f"cavs_{nm}_motifs/cav_trainer.pt"))
        motif_cav_trainers[nm] = cav_trainer

    if non_motif_concept_builder is not None:
        bed_cav_trainer = CavTrainer(tpcav_model, penalty="l2")
        bed_cav_trainer.set_control(
            non_motif_concept_builder.control_concepts[0], num_samples=num_samples_for_cav
        )
        bed_cav_trainer.train_concepts(
            non_motif_concept_builder.concepts,
            num_samples_for_cav,
            output_dir=str(output_path / f"cavs_bed_concepts/"),
            num_processes=p,
            backend=backend,
            device=device
        )
        if save_cav_trainer:
            torch.save(bed_cav_trainer, str(output_path / f"cavs_bed_concepts/cav_trainer.pt"))
    else:
        bed_cav_trainer = None

    if len(num_motif_insertions) > 1:
        cavs_fscores_df = compute_motif_auc_fscore(num_motif_insertions, list(motif_cav_trainers.values()), motif_file=motif_file, motif_file_fmt=motif_file_fmt)
    else:
        cavs_fscores_df = None

    if generate_html_report:
        report.generate_tcav_html_report(str(output_path / f"report.html"), motif_cav_trainers,
                                         extra_cav_trainers = {'bed concepts': bed_cav_trainer},
                                         motif_file=motif_file, motif_file_fmt=motif_file_fmt, fscore_thresh=html_report_fscore_thresh)

    return cavs_fscores_df, motif_cav_trainers, bed_cav_trainer
