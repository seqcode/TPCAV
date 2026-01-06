import logging
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import seqchromloader as scl
import webdataset as wds
from Bio import motifs as Bio_motifs
from captum.concept import Concept
from torch.utils.data import DataLoader

from . import helper, utils

logger = logging.getLogger(__name__)


class _PairedLoader:
    """Allow repeated iteration over paired dataloaders."""

    def __init__(self, seq_dl: Iterable, chrom_dl: Iterable) -> None:
        self.seq_dl = seq_dl
        self.chrom_dl = chrom_dl
        self.apply_func = None

    def apply(self, apply_func):
        self.apply_func = apply_func

    def __iter__(self):
        for inputs in zip(self.seq_dl, self.chrom_dl):
            if self.apply_func:
                inputs = self.apply_func(*inputs)
            yield inputs


def _construct_motif_concept_dataloader_from_control(
    control_seq_df: pd.DataFrame,
    genome_fasta: str,
    motifs: Sequence,
    num_motifs: int,
    motif_mode: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Mirror the motif-based dataloader logic used in the TCAV script."""
    datasets = []
    for motif in motifs:
        ds = utils.IterateSeqDataFrame(
            control_seq_df,
            genome_fasta,
            motif=motif,
            motif_mode=motif_mode,
            num_motifs=num_motifs,
            start_buffer=0,
            end_buffer=0,
            print_warning=False,
            infinite=False,
        )
        datasets.append(ds)

    mixed_dl = DataLoader(
        wds.RandomMix(datasets),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return mixed_dl


class ConceptBuilder:
    """Build and store concepts/control concepts in a reusable, programmatic way."""

    def __init__(
        self,
        genome_fasta: str,
        genome_size_file: str,
        input_window_length: int = 1024,
        bws: Optional[List[str]] = None,
        batch_size: int = 8,
        num_workers: int = 0,
        num_motifs: int = 12,
        include_reverse_complement: bool = False,
        min_samples: int = 5000,
        rng_seed: int = 1001,
    ) -> None:
        self.genome_fasta = genome_fasta
        self.genome_size_file = genome_size_file
        self.input_window_length = input_window_length
        self.bws = bws or []
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_motifs = num_motifs
        self.include_reverse_complement = include_reverse_complement
        self.min_samples = min_samples
        self.rng_seed = rng_seed

        self.control_regions: pd.DataFrame | None = None
        self.control_concepts: List[Concept] = []
        self.concepts: List[Concept] = []
        self.metadata: Dict[str, object] = {}
        self._next_concept_id = 0
        self._control_seq_loader: Optional[Iterable] = None
        self._control_chrom_loader: Optional[Iterable] = None

    def build_control(self, name: str = "random_regions") -> Concept:
        """Create the background/control concept."""
        control_regions = scl.random_coords(
            gs=self.genome_size_file,
            l=self.input_window_length,
            n=self.min_samples,
        )
        control_regions["label"] = -1
        control_regions["strand"] = "+"
        self.control_regions = control_regions

        concept = Concept(
            id=self._reserve_id(is_control=True),
            name=name,
            data_iter=_PairedLoader(self._control_seq_dl(), self._control_chrom_dl()),
        )
        self.control_concepts = [concept]
        self.metadata["control_regions"] = control_regions
        return concept

    def _control_seq_dl(self):
        if self.control_regions is None:
            raise ValueError("Call build_control before creating control regions.")
        seq_fasta_iter = helper.DataFrame2FastaIterator(
            self.control_regions, self.genome_fasta, batch_size=self.batch_size
        )
        return seq_fasta_iter

    def _control_chrom_dl(self):
        if self.control_regions is None:
            raise ValueError("Call build_control before creating control regions.")
        chrom_iter = helper.DataFrame2ChromTracksIterator(
            self.control_regions,
            self.bws,
            batch_size=self.batch_size,
        )
        return chrom_iter

    def add_custom_motif_concepts(
        self, motif_table: str, control_regions: Optional[pd.DataFrame] = None
    ) -> List[Concept]:
        """Add concepts from a tab-delimited motif table: motif_name<TAB>consensus."""
        if control_regions is None:
            if not self.control_concepts:
                raise ValueError("Call build_control or pass control_regions first.")
            control_regions = self.metadata.get("control_regions")
        assert control_regions is not None
        df = pd.read_table(motif_table, names=["motif_name", "consensus_seq"])
        added: List[Concept] = []
        for motif_name in np.unique(df.motif_name):
            consensus = df.loc[df.motif_name == motif_name, "consensus_seq"].tolist()
            motifs = []
            for idx, cons in enumerate(consensus):
                motif = utils.CustomMotif(f"{motif_name}_{idx}", cons)
                motifs.append(motif)
                if self.include_reverse_complement:
                    motifs.append(motif.reverse_complement())
            seq_dl = _construct_motif_concept_dataloader_from_control(
                control_regions,
                self.genome_fasta,
                motifs=motifs,
                num_motifs=self.num_motifs,
                motif_mode="consensus",
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
            concept = Concept(
                id=self._reserve_id(),
                name=motif_name,
                data_iter=_PairedLoader(seq_dl, self._control_chrom_dl()),
            )
            self.concepts.append(concept)
            added.append(concept)
        return added

    def add_meme_motif_concepts(
        self, meme_file: str, control_regions: Optional[pd.DataFrame] = None
    ) -> List[Concept]:
        """Add concepts from a MEME minimal-format motif file."""
        if control_regions is None:
            if not self.control_concepts:
                raise ValueError("Call build_control or pass control_regions first.")
            control_regions = self.metadata.get("control_regions")
        assert control_regions is not None

        added: List[Concept] = []
        with open(meme_file) as handle:
            for motif in Bio_motifs.parse(handle, fmt="MINIMAL"):
                motifs = [motif]
                if self.include_reverse_complement:
                    motifs.append(motif.reverse_complement())
                motif_name = motif.name.replace("/", "-")
                seq_dl = _construct_motif_concept_dataloader_from_control(
                    control_regions,
                    self.genome_fasta,
                    motifs=motifs,
                    num_motifs=self.num_motifs,
                    motif_mode="pwm",
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                )
                concept = Concept(
                    id=self._reserve_id(),
                    name=motif_name,
                    data_iter=_PairedLoader(seq_dl, self._control_chrom_dl()),
                )
                self.concepts.append(concept)
                added.append(concept)
        return added

    def add_bed_sequence_concepts(self, bed_paths: Iterable[str]) -> List[Concept]:
        """Add concepts backed by BED sequences with concept_name in column 5."""
        added: List[Concept] = []
        for bed in bed_paths:
            bed_df = pd.read_table(
                bed,
                header=None,
                usecols=[0, 1, 2, 3, 4],
                names=["chrom", "start", "end", "strand", "concept_name"],
            )
            for concept_name in bed_df.concept_name.unique():
                concept_df = bed_df.loc[bed_df.concept_name == concept_name]
                if len(concept_df) < self.min_samples:
                    logger.warning(
                        "Concept %s has %s samples, fewer than min_samples=%s; skipping",
                        concept_name,
                        len(concept_df),
                        self.min_samples,
                    )
                    continue
                seq_fasta_iter = helper.dataframe_to_fasta_iter(
                    concept_df.sample(n=self.min_samples, random_state=self.rng_seed),
                    self.genome_fasta,
                    batch_size=self.batch_size,
                )
                concept = Concept(
                    id=self._reserve_id(),
                    name=concept_name,
                    data_iter=_PairedLoader(seq_fasta_iter, self._control_chrom_dl()),
                )
                self.concepts.append(concept)
                added.append(concept)
        return added

    def add_bed_chrom_concepts(self, bed_paths: Iterable[str]) -> List[Concept]:
        """Add concepts backed by chromatin signal bigwigs and BED coordinates."""
        added: List[Concept] = []
        for bed in bed_paths:
            bed_df = pd.read_table(
                bed,
                header=None,
                usecols=[0, 1, 2, 3, 4],
                names=["chrom", "start", "end", "strand", "concept_name"],
            )
            for concept_name in bed_df.concept_name.unique():
                concept_df = bed_df.loc[bed_df.concept_name == concept_name]
                if len(concept_df) < self.min_samples:
                    logger.warning(
                        "Concept %s has %s samples, fewer than min_samples=%s; skipping",
                        concept_name,
                        len(concept_df),
                        self.min_samples,
                    )
                    continue
                chrom_dl = helper.dataframe_to_chrom_tracks_iter(
                    concept_df.sample(n=self.min_samples, random_state=self.rng_seed),
                    self.genome_fasta,
                    self.bws,
                    batch_size=self.batch_size,
                )
                concept = Concept(
                    id=self._reserve_id(),
                    name=concept_name,
                    data_iter=_PairedLoader(self._control_seq_dl(), chrom_dl),
                )
                self.concepts.append(concept)
                added.append(concept)
        return added

    def all_concepts(self) -> List[Concept]:
        """Return test + control concepts."""
        return [*self.concepts, *self.control_concepts]

    def summary(self) -> Dict[str, object]:
        """Lightweight run metadata."""
        return {
            "num_concepts": len(self.concepts),
            "num_control": len(self.control_concepts),
            "concept_names": [c.name for c in self.concepts],
            "control_names": [c.name for c in self.control_concepts],
            "input_window_length": self.input_window_length,
            "num_motifs": self.num_motifs,
            "bigwigs": self.bws,
        }

    def _reserve_id(self, is_control: bool = False) -> int:
        cid = -self._next_concept_id if is_control else self._next_concept_id
        self._next_concept_id += 1
        return cid

    def apply_transform(self, transform):
        """Apply a transform function to all concepts"""
        for c in self.all_concepts():
            c.data_iter.apply(transform)
