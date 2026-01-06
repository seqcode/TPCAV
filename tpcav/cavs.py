#!/usr/bin/env python3
"""
CAV training and attribution utilities built on TPCAV.
"""

import logging
import multiprocessing
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader, TensorDataset, random_split

from tpcav.tpcav_model import TPCAV

logger = logging.getLogger(__name__)


def _load_all_tensors_to_numpy(dataloaders: Iterable[DataLoader]):
    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]
    avs, ls = [], []
    for dataloader in dataloaders:
        for av, l in dataloader:
            avs.append(av.cpu().numpy())
            ls.append(l.cpu().numpy())
    return np.concatenate(avs), np.concatenate(ls)


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
        )
        if penalty == "l2":
            params = {"alpha": [1e-2, 1e-4, 1e-6]}
        elif penalty == "l1":
            params = {"alpha": [1e-1, 1]}
        else:
            raise ValueError(f"Unexpected penalty type {penalty}")
        self.search = GridSearchCV(self.lm, params)

    def fit(self, train_dl: DataLoader, val_dl: DataLoader):
        train_avs, train_ls = _load_all_tensors_to_numpy([train_dl, val_dl])
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


def _train(
    concept_embeddings: torch.Tensor,
    control_embeddings: torch.Tensor,
    output_dir: str,
    penalty: str = "l2",
) -> Tuple[float, torch.Tensor]:
    """
    Train a binary CAV classifier for a concept vs cached control embeddings.

    Requires set_control to have been called beforehand.
    """
    output_dir = Path(output_dir)

    avd = TensorDataset(
        concept_embeddings, torch.full((concept_embeddings.shape[0],), 0)
    )
    cvd = TensorDataset(
        control_embeddings, torch.full((control_embeddings.shape[0],), 1)
    )
    train_ds, val_ds, test_ds = random_split(avd, [0.8, 0.1, 0.1])
    c_train, c_val, c_test = random_split(cvd, [0.8, 0.1, 0.1])

    train_dl = DataLoader(train_ds + c_train, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds + c_val, batch_size=32)
    test_dl = DataLoader(test_ds + c_test, batch_size=32)

    clf = _SGDWrapper(penalty=penalty)
    clf.fit(train_dl, val_dl)

    def _eval(split_dl: DataLoader, name: str):
        y_preds, y_trues = [], []
        for x, y in split_dl:
            y_pred = clf.predict(x.cpu().numpy())
            y_preds.append(y_pred)
            y_trues.append(y.cpu().numpy())
        y_preds = np.concatenate(y_preds)
        y_trues = np.concatenate(y_trues)
        acc = (y_preds == y_trues).sum() / len(y_trues)
        precision, recall, fscore, support = precision_recall_fscore_support(
            y_trues, y_preds, average="binary", pos_label=1
        )
        logger.info("[%s] Accuracy: %.4f", name, acc)
        (output_dir / f"classifier_perform_on_{name}.txt").write_text(
            f"Accuracy: {acc}\n"
        )
        return fscore

    output_dir.mkdir(parents=True, exist_ok=True)
    _eval(train_dl, "train")
    _eval(val_dl, "val")
    test_fscore = _eval(test_dl, "test")

    weights = clf.weights
    assert len(weights.shape) == 2 and weights.shape[0] == 2
    torch.save(weights, output_dir / "classifier_weights.pt")

    return test_fscore, weights[0]


class CavTrainer:
    """Train CAVs and compute attribution-driven TCAV scores."""

    def __init__(self, tpcav: TPCAV, penalty: str = "l2") -> None:
        self.tpcav = tpcav
        self.penalty = penalty
        self.cavs_fscores = {}
        self.cav_weights = {}
        self.control_embeddings: Optional[torch.Tensor] = None
        self.cavs_list: List[torch.Tensor] = []

    def save_state(self, output_path: str = "cav_trainer_state.pt"):
        """
        Save CavTrainer state to a file.
        """
        state = {
            "cavs_fscores": self.cavs_fscores,
            "cav_weights": self.cav_weights,
            "control_embeddings": self.control_embeddings,
            "cavs_list": self.cavs_list,
        }
        torch.save(state, output_path)

    def restore_state(self, input_path: str = "cav_trainer_state.pt"):
        """
        Restore CavTrainer state from a file.
        """
        state = torch.load(input_path, map_location="cpu")
        self.cavs_fscores = state["cavs_fscores"]
        self.cav_weights = state["cav_weights"]
        self.control_embeddings = state["control_embeddings"]
        self.cavs_list = state["cavs_list"]

    def set_control(self, control_concept, num_samples: int) -> torch.Tensor:
        """
        Set and cache control embeddings to avoid recomputation across CAV trainings.
        """
        self.control_embeddings = self.tpcav.concept_embeddings(
            control_concept, num_samples=num_samples
        )
        return self.control_embeddings

    def train_concepts(
        self,
        concept_list,
        num_samples: int,
        output_dir: str,
        num_processes: int = 1,
    ):
        if self.control_embeddings is None:
            raise ValueError(
                "Call set_control(control_concept, num_samples=...) before training CAVs."
            )

        if num_processes == 1:
            for c in concept_list:
                concept_embeddings = self.tpcav.concept_embeddings(
                    c, num_samples=num_samples
                )
                fscore, weight = _train(
                    concept_embeddings,
                    self.control_embeddings,
                    Path(output_dir) / c.name,
                    self.penalty,
                )
                self.cavs_fscores[c.name] = fscore
                self.cav_weights[c.name] = weight
                self.cavs_list.append(weight)
        else:
            pool = multiprocessing.Pool(processes=num_processes)
            results = []
            for c in concept_list:
                concept_embeddings = self.tpcav.concept_embeddings(
                    c, num_samples=num_samples
                )
                res = pool.apply_async(
                    _train,
                    args=(
                        concept_embeddings,
                        self.control_embeddings,
                        Path(output_dir) / c.name,
                        self.penalty,
                    ),
                )
                logger.info("Submitted CAV training for concept %s", c.name)
                results.append((c.name, res))
            pool.close()
            pool.join()
            results = [(name, res.get()) for name, res in results]
            for name, (fscore, weight) in results:
                self.cavs_fscores[name] = fscore
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

    def plot_cavs_similaritiy_heatmap(
        self,
        attributions: torch.Tensor,
        concept_list: List[str] | None = None,
        fscore_thresh=0.8,
        output_path: str = "cavs_similarity_heatmap.png",
    ):
        if concept_list is None:
            cavs_names = list(self.cav_weights.keys())
        else:
            cavs_names = concept_list
        cavs_pass = []
        cavs_names_pass = []
        for cname in cavs_names:
            if self.cavs_fscores[cname] >= fscore_thresh:
                cavs_pass.append(self.cav_weights[cname])
                cavs_names_pass.append(cname)
            else:
                logger.info(
                    "Skipping CAV %s with F-score %.3f below threshold %.3f",
                    cname,
                    self.cavs_fscores[cname],
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
            yticklabels=False,
            cmap="bwr",
            vmin=-1,
            vmax=1,
        )
        cm.gs.update(left=0.05, right=0.5)
        cm.ax_cbar.set_position([0.01, 0.9, 0.05, 0.05])

        cavs_names_sorted = [
            cavs_names_pass[i] for i in cm.dendrogram_col.reordered_ind
        ]

        ## plot log ratio plot
        ax_log = cm.figure.add_subplot()
        heatmap_bbox = cm.ax_heatmap.get_position()
        ax_log.set_position([0.5, heatmap_bbox.y0, 0.2, heatmap_bbox.height])
        # used to leave space for motif logos
        # ax_log.tick_params(
        #    axis="y", which="major", pad=cm.figure.get_size_inches()[0] * 0.2 * 72
        # )

        log_ratios_reordered = [
            self.tpcav_score_binary_log_ratio(cname, attributions)
            for cname in cavs_names_sorted
        ]
        sns.barplot(y=cavs_names_sorted, x=log_ratios_reordered, orient="y", ax=ax_log)
        # set color of bar by value
        for idx in range(len((ax_log.containers[0]))):
            if ax_log.containers[0].datavalues[idx] > 0:
                ax_log.containers[0][idx].set_color("red")
            else:
                ax_log.containers[0][idx].set_color("blue")

        ax_log.set_xlim(left=-5, right=5)
        ax_log.yaxis.tick_right()
        ax_log.set_title("TCAV log ratio")

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
