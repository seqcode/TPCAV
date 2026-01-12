#!/usr/bin/env python3
"""
Lightweight data loading helpers for sequences and chromatin tracks.
"""

from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import seqchromloader as scl
import torch
from deeplift.dinuc_shuffle import dinuc_shuffle
from pyfaidx import Fasta
from seqchromloader.utils import dna2OneHot, extract_bw


def load_bed_and_center(bed_file: str, window: int) -> pd.DataFrame:
    """
    Load a BED file and center the regions to a fixed window size.
    """
    bed_df = pd.read_table(bed_file, usecols=[0, 1, 2], names=["chrom", "start", "end"])
    return center_dataframe_regions(bed_df, window)


def center_dataframe_regions(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Center the regions in a DataFrame to a fixed window size, keep other columns. Put chrom, start, end as the first 3 columns.
    """
    df = df.copy()
    df["center"] = ((df["start"] + df["end"]) // 2).astype(int)
    df["start"] = df["center"] - (window // 2)
    df["end"] = df["start"] + window
    df = df.drop(columns=["center"])
    cols = ["chrom", "start", "end"] + [
        col for col in df.columns if col not in ["chrom", "start", "end"]
    ]
    df = df[cols]
    return df


def bed_to_fasta_iter(
    bed_file: str, genome_fasta: str, batch_size: int
) -> Iterable[List[str]]:
    """
    Yield sequences from a BED file as fasta strings.
    """
    bed_df = pd.read_table(bed_file, usecols=[0, 1, 2], names=["chrom", "start", "end"])
    yield from dataframe_to_fasta_iter(bed_df, genome_fasta, batch_size)


def dataframe_to_fasta_iter(
    df: pd.DataFrame, genome_fasta: str, batch_size: int
) -> Iterable[List[str]]:
    """
    Yield sequences from a DataFrame with columns [chrom, start, end].
    """
    genome = Fasta(genome_fasta)
    fasta_seqs = []
    for row in df.itertuples(index=False):
        seq = str(genome[row.chrom][row.start : row.end]).upper()
        if len(seq) != (row.end - row.start):
            raise ValueError(
                f"Extract Fasta sequence length mismatch with region coordinate length {row.chrom}:{row.start}-{row.end}"
            )
        fasta_seqs.append(seq)
        if len(fasta_seqs) == batch_size:
            yield fasta_seqs
            fasta_seqs = []
    if fasta_seqs:
        yield fasta_seqs


class DataFrame2FastaIterator:
    """
    Iterator class to yield sequences from a DataFrame with columns [chrom, start, end].
    """

    def __init__(self, df: pd.DataFrame, genome_fasta: str, batch_size: int):
        self.genome_fasta = genome_fasta
        self.df = df
        self.batch_size = batch_size

    def __iter__(self):
        return dataframe_to_fasta_iter(
            self.df, genome_fasta=self.genome_fasta, batch_size=self.batch_size
        )


def bed_to_chrom_tracks_iter(
    bed_file: str, genome_fasta: str, bigwigs: List[str]
) -> Iterable[torch.Tensor]:
    """
    Yield chromatin tracks for BED regions using bigwig files.
    Each batch is shaped [batch, num_tracks, window].
    """
    bed_df = pd.read_table(bed_file, usecols=[0, 1, 2], names=["chrom", "start", "end"])
    yield from dataframe_to_chrom_tracks_iter(bed_df, genome_fasta, bigwigs)


def dataframe_to_chrom_tracks_iter(
    df: pd.DataFrame,
    bigwigs: List[str] | None,
    batch_size: int = 1,
) -> Iterable[torch.Tensor | None]:
    """
    Yield chromatin tracks for regions from a DataFrame using bigwig files.
    """
    if bigwigs is not None and len(bigwigs) > 0:
        chrom_arrs = []
        for row in df.itertuples(index=False):
            chrom = extract_bw(
                row.chrom, row.start, row.end, getattr(row, "strand", "+"), bigwigs
            )
            chrom_arrs.append(chrom)
            if batch_size is not None and len(chrom_arrs) == batch_size:
                yield torch.tensor(np.stack(chrom_arrs))
                chrom_arrs = []
        if chrom_arrs:
            yield torch.tensor(np.stack(chrom_arrs))
    else:
        while True:
            yield torch.full((batch_size, 1), torch.nan)


class DataFrame2ChromTracksIterator:
    """
    Iterator class to yield chromatin tracks from a DataFrame with columns [chrom, start, end].
    """

    def __init__(
        self,
        df: pd.DataFrame,
        bigwigs: List[str] | None,
        batch_size: int = 1,
    ):
        self.bigwigs = bigwigs
        self.df = df
        self.batch_size = batch_size

    def __iter__(self):
        return dataframe_to_chrom_tracks_iter(
            self.df,
            bigwigs=self.bigwigs,
            batch_size=self.batch_size,
        )


def fasta_to_one_hot_sequences(seqs: List[str]) -> torch.Tensor:
    """
    Return one-hot encoded numpy arrays [4, L] for list of fasta sequences.
    """
    return torch.tensor(np.stack([dna2OneHot(seq) for seq in seqs]))


def random_regions_dataframe(
    genome_size_file: str, window: int, n: int, seed: int = 1
) -> pd.DataFrame:
    """
    Generate random regions as a DataFrame with columns [chrom, start, end].
    """
    return scl.random_coords(gs=genome_size_file, l=window, n=n, seed=seed)[
        ["chrom", "start", "end"]
    ]


def dinuc_shuffle_sequences(
    seqs: Iterable[str], num_shuffles: int = 10, seed: int = 1
) -> List[List[str]]:
    """
    For each fasta sequence, yield a list of dinucleotide-shuffled sequences.
    """
    rng = np.random.RandomState(seed)
    results = []
    for seq in seqs:
        shuffles = dinuc_shuffle(
            seq,
            num_shufs=num_shuffles,
            rng=rng,
        )
        results.append(shuffles)
    return results


def fasta_chrom_to_one_hot_seq(seq, chrom):
    return (fasta_to_one_hot_sequences(seq),)
