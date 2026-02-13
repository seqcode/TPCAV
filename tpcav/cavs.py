#!/usr/bin/env python3
"""
CAV training and attribution utilities built on TPCAV.
"""

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union
import time

from concurrent.futures import ProcessPoolExecutor
from Bio import motifs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.linear_model import LinearRegression
import logomaker

from . import helper, utils
from .concepts import ConceptBuilder
from .tpcav_model import TPCAV
from matplotlib import gridspec

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
            average=True,
        )
        if penalty == "l2":
            params = {"alpha": [1e-2, 1e-4, 1e-6, 1e-8]}
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
            "penalty": self.penalty,
            "cavs_fscores": self.cavs_fscores,
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

        cav_trainer.cavs_fscores = state["cavs_fscores"]
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

    def train_concepts(
        self,
        concept_list,
        num_samples: int,
        output_dir: str,
        num_processes: int = 1,
        max_pending: int = 8
    ):
        "Train concepts with a fixed control set by self.set_control()"
        if self.control_embeddings is None:
            raise ValueError(
                "Call set_control(control_concept, num_samples=...) before training CAVs."
            )
        else:
            self.control_embeddings = self.control_embeddings.cpu()

        if num_processes == 1:
            for c in concept_list:
                concept_embeddings = self.tpcav.concept_embeddings(
                    c, num_samples=num_samples
                )
                fscore, weight = _train(
                    concept_embeddings.cpu(),
                    self.control_embeddings.cpu(),
                    Path(output_dir) / c.name,
                    self.penalty,
                )
                self.cavs_fscores[c.name] = fscore
                self.cav_weights[c.name] = weight
                self.cavs_list.append(weight)
        else:
            futures = []
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                for c in concept_list:
                    concept_embeddings = self.tpcav.concept_embeddings(
                        c, num_samples=num_samples
                    )

                    # block the process to avoid too long queue
                    while True:
                        done = [f for (_, f) in futures if f.done()]
                        for f in done:
                            f.result()  # raises if worker failed

                        pending = [f for (_, f) in futures if not f.done()]
                        if len(pending) < (max_pending + num_processes):
                            break

                        time.sleep(5)

                    future = executor.submit(
                        _train,
                        concept_embeddings.cpu(),
                        self.control_embeddings,
                        Path(output_dir) / c.name,
                        self.penalty,
                    )
                    logger.info("Submitted CAV training for concept %s", c.name)
                    futures.append((c.name, future))

                results = [(name, f.result()) for name, f in futures]
            for name, (fscore, weight) in results:
                self.cavs_fscores[name] = fscore
                self.cav_weights[name] = weight
                self.cavs_list.append(weight)

    def train_concepts_pairs(self,
                             concept_pair_list,
                             num_samples: int,
                             output_dir: str,
                             num_processes: int = 1,
                             max_pending: int = 8):
        """Train concept pairs (test concept, control concept)

        Note: It would compute embeddings on every control concept, use self.train_concepts if control concept is fixed
        """
        if num_processes == 1:
            for c_test, c_control in concept_pair_list:
                concept_embeddings = self.tpcav.concept_embeddings(
                    c_test, num_samples=num_samples
                )
                control_embeddings = self.tpcav.concept_embeddings(
                    c_control, num_samples=num_samples
                )

                fscore, weight = _train(
                    concept_embeddings.cpu(),
                    control_embeddings.cpu(),
                    Path(output_dir) / c_test.name,
                    self.penalty,
                )
                self.cavs_fscores[c_test.name] = fscore
                self.cav_weights[c_test.name] = weight
                self.cavs_list.append(weight)
        else:
            futures = []
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                for c_test, c_control in concept_pair_list:
                    concept_embeddings = self.tpcav.concept_embeddings(
                        c_test, num_samples=num_samples
                    )
                    control_embeddings = self.tpcav.concept_embeddings(
                        c_control, num_samples=num_samples
                    )

                    # block the process to avoid too long queue
                    while True:
                        done = [f for (_, f) in futures if f.done()]
                        for f in done:
                            f.result()  # raises if worker failed

                        pending = [f for (_, f) in futures if not f.done()]
                        if len(pending) < (max_pending + num_processes):
                            break

                        time.sleep(5)

                    future = executor.submit(
                        _train,
                        concept_embeddings.cpu(),
                        control_embeddings.cpu(),
                        Path(output_dir) / c_test.name,
                        self.penalty,
                    )
                    logger.info("Submitted CAV training for concept %s", c_test.name)
                    futures.append((c_test.name, future))

                results = [(name, f.result()) for name, f in futures]
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
        attributions: Optional[List[torch.Tensor]] = None,
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
            if self.cavs_fscores[cname] >= fscore_thresh:
                cavs_pass.append(self.cav_weights[cname].cpu().numpy())
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
        cm.gs.update(left=0, right=1)
        cm.ax_cbar.set_position([0.01, 0.9, 0.05, 0.05])

        cavs_names_sorted = [
            cavs_names_pass[i] for i in cm.dendrogram_col.reordered_ind
        ]

        heatmap_bbox = cm.ax_heatmap.get_position()
        ax_logs = []
        if attributions is not None:
            for i, attrs in enumerate(attributions):
                offset =  1 + i*0.2
                ## plot log ratio plot
                ax_log = cm.figure.add_subplot()
                ax_log.set_position([offset, heatmap_bbox.y0, 0.2, heatmap_bbox.height])

                log_ratios_reordered = [
                    self.tpcav_score_binary_log_ratio(cname, attrs)
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
                ax_log.set_title(f"TCAV log ratio {i}")
                ax_log.get_yaxis().set_visible(False)
                ax_logs.append(ax_log)
            ax_logs[-1].get_yaxis().set_visible(True) # only show yaxis labels in the last log ratio plot
        
        # plot motif logo if provided meme file, try to look for pwm for every concept in the file
        if motif_meme_file is not None:
            ax_logs[-1].tick_params(
               axis="y", which="major", pad=cm.figure.get_size_inches()[0] * 0.2 * 72 # leave space for motif logos
            )
            gs_logo = gridspec.GridSpec(len(cavs_names), 1)

            logo_height = heatmap_bbox.height/len(cavs_names)
            for i, (cav_key, g) in enumerate(zip(cavs_names_sorted[::-1], gs_logo)):
                ax_logo = plt.subplot(g)
                ax_logo.set_position([1+len(ax_logs)*0.2+0.01, heatmap_bbox.y0+i*logo_height, 0.2+0.01, logo_height])
                if cav_key is not None:
                    seq_logo(cav_key, motif_meme_file=motif_meme_file, ax=ax_logo)
                else:
                    ax_logo.axis('off')

        plt.savefig(output_path, dpi=300, bbox_inches="tight")

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

def compute_motif_auc_fscore(num_motif_insertions: List[int], cav_trainers: List[CavTrainer], meme_motif_file: Optional[str] = None):
    cavs_fscores_df = pd.DataFrame({nm: cav_trainer.cavs_fscores for nm, cav_trainer in zip(num_motif_insertions, cav_trainers)})
    cavs_fscores_df['concept'] = list(cav_trainers[0].cavs_fscores.keys())

    def compute_auc_fscore(row):
        y = [row[nm] for nm in num_motif_insertions]
        return np.trapz(y, num_motif_insertions) / (
            num_motif_insertions[-1] - num_motif_insertions[0]
        )

    cavs_fscores_df["AUC_fscores"] = cavs_fscores_df.apply(compute_auc_fscore, axis=1)

    # if motif instances are provided, fit linear regression curve to remove the dependency of f-scores on information content and motif lengthj
    if meme_motif_file is not None:
        motifs_dict = load_motifs_from_meme(meme_motif_file)
        cavs_fscores_df['information_content'] = cavs_fscores_df.apply(lambda x: motifs_dict[x['concept']].relative_entropy.sum(), axis=1)
        cavs_fscores_df['motif_len'] = cavs_fscores_df.apply(lambda x: len(motifs_dict[x['concept']].consensus), axis=1)
        
        model = LinearRegression()
        model.fit(cavs_fscores_df[['information_content', 'motif_len']].to_numpy(), cavs_fscores_df['AUC_fscores'].to_numpy()[:, np.newaxis])
        
        y_pred = model.predict(cavs_fscores_df[['information_content', 'motif_len']].to_numpy())
        residuals = cavs_fscores_df['AUC_fscores'].to_numpy() - y_pred.flatten()
        cavs_fscores_df['AUC_fscores_residual'] = residuals

        cavs_fscores_df.sort_values("AUC_fscores_residual", ascending=False, inplace=True)
    else:
        cavs_fscores_df.sort_values("AUC_fscores", ascending=False, inplace=True)

    return cavs_fscores_df

def run_tpcav(
    model,
    meme_motif_file: str,
    genome_fasta: str,
    num_motif_insertions: List[int] = [4, 8, 16],
    bed_seq_file: Optional[str] = None,
    bed_chrom_file: Optional[str] = None,
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
    p=1, max_pending_jobs=4,
):
    """
    One-stop function to compute CAVs on motif concepts and bed concepts, compute AUC of motif concept f-scores after correction
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = Path(output_dir)
    # create concept builder to generate concepts
    ## motif concepts
    motif_concepts_pairs = {}
    motif_concept_builders = []
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
        )
        # use random regions as control
        builder.build_control()
        # use meme motif PWMs to build motif concepts, one concept per motif
        concepts_pairs = builder.add_meme_motif_concepts(str(meme_motif_file))

        # apply transform to convert fasta sequences to one-hot encoded sequences
        builder.apply_transform(input_transform_func)

        motif_concepts_pairs[nm] = concepts_pairs
        motif_concept_builders.append(builder)

    ## bed concepts (optional)
    if bed_seq_file is not None or bed_chrom_file is not None:
        bed_builder = ConceptBuilder(
            genome_fasta=genome_fasta,
            input_window_length=input_window_length,
            bws=bws,
            num_motifs=0,
            include_reverse_complement=True,
            min_samples=num_samples_for_cav,
            batch_size=batch_size,
        )
        # use random regions as control
        bed_builder.build_control()
        if bed_seq_file is not None:
            # build concepts from fasta sequences in bed file
            bed_builder.add_bed_sequence_concepts(bed_seq_file)
        if bed_chrom_file is not None:
            # build concepts from chromatin tracks in bed file
            bed_builder.add_bed_chrom_concepts(bed_chrom_file)
        # apply transform to convert fasta sequences to one-hot encoded sequences
        bed_builder.apply_transform(input_transform_func)
    else:
        bed_builder = None

    # create TPCAV model on top of the given model
    tpcav_model = TPCAV(model, layer_name=layer_name, layer=layer)
    # fit PCA on sampled all concept activations of the last builder (should have the most motifs)
    tpcav_model.fit_pca(
        concepts=motif_concept_builders[-1].concepts_for_pca() + bed_builder.concepts_for_pca() if  bed_builder is not None else motif_concept_builders[-1].concepts_for_pca(),
        num_samples_per_concept=num_samples_for_pca,
        num_pc=num_pc,
    )
    #torch.save(tpcav_model, output_path / "tpcav_model.pt")

    # create trainer for computing CAVs
    motif_cav_trainers = {}
    for nm in num_motif_insertions:
        cav_trainer = CavTrainer(tpcav_model, penalty="l2")
        cav_trainer.train_concepts_pairs(motif_concepts_pairs[nm], 
                                         num_samples_for_cav, 
                                         output_dir=str(output_path / f"cavs_{nm}_motifs/"),
                                         num_processes=p, max_pending=max_pending_jobs)
        motif_cav_trainers[nm] = cav_trainer
    if bed_builder is not None:
        bed_cav_trainer = CavTrainer(tpcav_model, penalty="l2")
        bed_cav_trainer.set_control(
            bed_builder.control_concepts[0], num_samples=num_samples_for_cav
        )
        bed_cav_trainer.train_concepts(
            bed_builder.concepts,
            num_samples_for_cav,
            output_dir=str(output_path / f"cavs_bed_concepts/"),
            num_processes=p,
        )
    else:
        bed_cav_trainer = None

    if len(num_motif_insertions) > 1:
        cavs_fscores_df = compute_motif_auc_fscore(num_motif_insertions, list(motif_cav_trainers.values()), meme_motif_file=meme_motif_file)
    else:
        cavs_fscores_df = None

    return cavs_fscores_df, motif_cav_trainers, bed_cav_trainer
