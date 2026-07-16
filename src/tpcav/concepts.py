import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyfaidx
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
        self.apply_func_list = []

    def apply(self, apply_func):
        self.apply_func_list.append(apply_func)

    def __iter__(self):
        for inputs in zip(self.seq_dl, self.chrom_dl):
            for apply_func in self.apply_func_list:
                inputs = apply_func(*inputs)
            yield inputs


class _SyntheticGCSeqIterator:
    def __init__(self, seq_len: int, n: int, batch_size: int, gc: float, seed: int):
        self.seq_len = int(seq_len)
        self.n = int(n)
        self.batch_size = int(batch_size)
        self.gc = float(gc)
        self.seed = int(seed)

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        bases = np.array(["A", "C", "G", "T"], dtype="<U1")
        p_at = (1.0 - self.gc) / 2.0
        p_gc = self.gc / 2.0
        p = [p_at, p_gc, p_gc, p_at]
        for start in range(0, self.n, self.batch_size):
            bs = min(self.batch_size, self.n - start)
            arr = rng.choice(4, size=(bs, self.seq_len), p=p)
            seqs = ["".join(bases[row]) for row in arr]
            yield seqs

def _construct_motif_concept_dataloader_from_control(
    control_seq_df: pd.DataFrame,
    genome_fasta: str,
    motifs: Sequence,
    num_motifs: int,
    batch_size: int,
    num_workers: int,
    start_buffer=0,
    end_buffer=0
) -> DataLoader:
    """Mirror the motif-based dataloader logic used in the TCAV script."""
    datasets = []
    for motif in motifs:
        ds = utils.IterateSeqDataFrame(
            control_seq_df,
            genome_fasta,
            motif=motif,
            num_motifs=num_motifs,
            start_buffer=start_buffer,
            end_buffer=end_buffer,
            print_warning=False,
            infinite=False,
        )
        datasets.append(ds)

    mixed_dl = DataLoader(
        wds.RandomMix(datasets),
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    return mixed_dl


class ConceptBuilder:
    """Build and store concepts/control concepts in a reusable, programmatic way."""

    def __init__(
        self,
        genome_fasta: str,
        input_window_length: int = 1024,
        bws: Optional[List[str]] = None,
        batch_size: int = 8,
        num_workers: int = 0,
        num_motifs: int = 12,
        include_reverse_complement: bool = False,
        min_samples: int = 5000,
        rng_seed: int = 1001,
        concept_name_suffix: str = "",
    ) -> None:
        self.genome_fasta = genome_fasta
        pyfaidx.Fasta(
            genome_fasta, build_index=True
        )  # validate genome fasta file and build index if needed
        self.genome_size_file = self.genome_fasta + ".fai"
        self.input_window_length = input_window_length
        self.bws = bws or []
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_motifs = num_motifs
        self.include_reverse_complement = include_reverse_complement
        self.min_samples = min_samples
        self.rng_seed = rng_seed
        self.concept_name_suffix = concept_name_suffix

        self.control_regions = None
        self.control_concepts: List[Concept] = []
        self.motif_permute_concepts: List[Concept] = []
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
            name=name + self.concept_name_suffix,
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

    def add_synthetic_gc_content_concepts(self, gc_content_step=0.1):
        """
        Add a list of GC content concepts,
        according to gc_content_step, example 0.1, GC content of each concept increases from 0.0 to 1.0 by the step
        concept iter is basically the same as other add concept function, batch the generated input
        """
        step = float(gc_content_step)
        if step <= 0 or step > 1:
            raise ValueError("gc_content_step must be in (0, 1].")

        # Include 1.0 endpoint (within floating tolerance).
        gc_values = np.arange(0.0, 1.0 + 1e-9, step, dtype=float)

        added: List[Concept] = []
        for gc in gc_values:
            gc = float(np.clip(gc, 0.0, 1.0))
            concept_name = f"synthetic_gc_{gc:.2f}" + self.concept_name_suffix
            seed = self.rng_seed + int(round(gc * 10000))
            seq_iter = _SyntheticGCSeqIterator(
                seq_len=self.input_window_length,
                n=self.min_samples,
                batch_size=self.batch_size,
                gc=gc,
                seed=seed,
            )
            concept = Concept(
                id=self._reserve_id(),
                name=concept_name,
                data_iter=_PairedLoader(seq_iter, self._control_chrom_dl()),
            )
            self.concepts.append(concept)
            added.append(concept)

        self.metadata["synthetic_gc_content_step"] = step
        self.metadata["synthetic_gc_content_values"] = gc_values.tolist()
        return added

    def add_custom_motif_concepts(
        self, motif_table: str, control_regions: Optional[pd.DataFrame] = None, build_permute_control=True
    ) -> Union[List[Concept], List[Tuple[Concept]]]:
        """Add concepts from a tab-delimited motif table: motif_name<TAB>consensus."""
        df = pd.read_table(motif_table, names=["motif_name", "consensus_seq"])
        added = []
        for motif_name in np.unique(df.motif_name):
            motif_name = utils.clean_motif_name(motif_name)
            consensus = df.loc[df.motif_name == motif_name, "consensus_seq"].tolist()
            motifs = []
            for idx, cons in enumerate(consensus):
                motif = utils.ConsensusMotif(f"{motif_name}_{idx}", cons)
                motifs.append(motif)
            concept = self.build_motif_concept(motifs, motif_name, control_regions=control_regions)
            self.concepts.append(concept)
            # build permute control if specified
            if build_permute_control:
                motifs_permuted = [utils.PermutedConsensusMotif(m.name + "_perm", m.consensus) for m in motifs]
                concept_permuted = self.build_motif_concept(motifs_permuted, motif_name + "_perm", control_regions=control_regions)
                self.motif_permute_concepts.append(concept_permuted)
                added.append((concept, concept_permuted))
            else:
                added.append(concept)
        return added

    def add_meme_motif_concepts(
        self, meme_file: str, control_regions: Optional[pd.DataFrame] = None, build_permute_control=True) -> Union[List[Concept], List[Tuple[Concept]]]:
        """Add concepts from a MEME minimal-format motif file."""

        added = []
        with open(meme_file) as handle:
            for motif in Bio_motifs.parse(handle, fmt="MINIMAL"):
                motif_name = utils.clean_motif_name(motif.name)
                concept = self.build_motif_concept([utils.BioMotifWrapped(motif),], motif_name, control_regions=control_regions)
                self.concepts.append(concept)
                # build permute control if specified
                if build_permute_control:
                    motif_permuted = utils.PermutedPWMMotif(motif)
                    concept_permuted = self.build_motif_concept([motif_permuted,], motif_name + "_perm", control_regions=control_regions)
                    self.motif_permute_concepts.append(concept_permuted)
                    added.append((concept, concept_permuted))
                else:
                    added.append(concept)
        return added

    def build_motif_concept(self, motifs, concept_name, control_regions=None, start_buffer=0, end_buffer=0):
        if control_regions is None:
            if not self.control_concepts:
                raise ValueError("Call build_control or pass control_regions first.")
            control_regions = self.metadata.get("control_regions")
        assert control_regions is not None

        if self.include_reverse_complement:
            motifs.extend([m.reverse_complement() for m in motifs])
        seq_dl = _construct_motif_concept_dataloader_from_control(
            control_regions,
            self.genome_fasta,
            motifs=motifs,
            num_motifs=self.num_motifs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            start_buffer=start_buffer,
            end_buffer=end_buffer
        )
        concept = Concept(
            id=self._reserve_id(),
            name=concept_name + self.concept_name_suffix,
            data_iter=_PairedLoader(seq_dl, self._control_chrom_dl()),
        )
        return concept

    def add_bed_sequence_concepts(self, bed_path: str) -> List[Concept]:
        """Add concepts backed by BED sequences with concept_name in column 5."""
        added: List[Concept] = []
        bed_df = pd.read_table(
            bed_path,
            header=None,
            usecols=[0, 1, 2, 3, 4],
            names=["chrom", "start", "end", "strand", "concept_name"],
        )
        added.extend(self.add_dataframe_sequence_concepts(bed_df))
        return added

    def add_dataframe_sequence_concepts(self, dataframe: pd.DataFrame) -> List[Concept]:
        """Add concepts backed by BED sequences with concept_name in column 5."""
        dataframe = helper.center_dataframe_regions(dataframe, self.input_window_length)
        added: List[Concept] = []
        for concept_name in dataframe.concept_name.unique():
            concept_df = dataframe.loc[dataframe.concept_name == concept_name]
            if len(concept_df) < self.min_samples:
                logger.warning(
                    "Concept %s has %s samples, fewer than min_samples=%s; skipping",
                    concept_name,
                    len(concept_df),
                    self.min_samples,
                )
                continue
            seq_fasta_iter = helper.DataFrame2FastaIterator(
                #concept_df.sample(n=self.min_samples, random_state=self.rng_seed),
                concept_df,
                self.genome_fasta,
                batch_size=self.batch_size,
            )
            concept = Concept(
                id=self._reserve_id(),
                name=concept_name + self.concept_name_suffix,
                data_iter=_PairedLoader(seq_fasta_iter, self._control_chrom_dl()),
            )
            self.concepts.append(concept)
            added.append(concept)
        return added

    def add_bed_chrom_concepts(self, bed_path: str) -> List[Concept]:
        """Add concepts backed by chromatin signal bigwigs and BED coordinates."""
        added: List[Concept] = []
        bed_df = pd.read_table(
            bed_path,
            header=None,
            usecols=[0, 1, 2, 3, 4],
            names=["chrom", "start", "end", "strand", "concept_name"],
        )
        added.extend(self.add_dataframe_chrom_concepts(bed_df))
        return added

    def add_dataframe_chrom_concepts(self, dataframe) -> List[Concept]:
        """Add concepts backed by chromatin signal bigwigs and BED coordinates."""
        dataframe = helper.center_dataframe_regions(dataframe, self.input_window_length)
        added: List[Concept] = []
        for concept_name in dataframe.concept_name.unique():
            concept_df = dataframe.loc[dataframe.concept_name == concept_name]
            if len(concept_df) < self.min_samples:
                logger.warning(
                    "Concept %s has %s samples, fewer than min_samples=%s; skipping",
                    concept_name,
                    len(concept_df),
                    self.min_samples,
                )
                continue
            chrom_dl = helper.DataFrame2ChromTracksIterator(
                #concept_df.sample(n=self.min_samples, random_state=self.rng_seed),
                concept_df,
                self.bws,
                batch_size=self.batch_size,
            )
            concept = Concept(
                id=self._reserve_id(),
                name=concept_name + self.concept_name_suffix,
                data_iter=_PairedLoader(self._control_seq_dl(), chrom_dl),
            )
            self.concepts.append(concept)
            added.append(concept)
        return added

    def all_concepts(self) -> List[Concept]:
        """Return test + control + permute concepts."""
        return [*self.concepts, *self.control_concepts, *self.motif_permute_concepts]

    def concepts_for_pca(self) -> List[Concept]:
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
