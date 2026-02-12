#!/usr/bin/env python3
"""
Lightweight data loading helpers for sequences and chromatin tracks.
"""

from typing import Iterable, List, Optional

import itertools
import logging
import pyBigWig
import re
import sys
import numpy as np
import pandas as pd
import seqchromloader as scl
import torch
from deeplift.dinuc_shuffle import dinuc_shuffle
from pyfaidx import Fasta
from seqchromloader.utils import dna2OneHot, extract_bw

logger = logging.getLogger(__name__)


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
    bigwigs: Optional[List[str]],
    batch_size: int = 1,
) -> Iterable[torch.Tensor]:
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
        bigwigs: Optional[List[str]],
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

def write_attrs_to_bw(arrs, regions, genome_info, bigwig_fn, smooth=False):
    """
    write the attributions into bigwig files
    shape of arrs should be (# samples, length)
    Note: If regions overlap with each other, only base pairs not covered by previous regions would be assigned current region's attribution score
    """
    # write header into bigwig
    bw = pyBigWig.open(bigwig_fn, "w")
    heads = []
    with open(genome_info, "r") as f:
        for line in f:
            chrom, length = line.strip().split("\t")[:2]
            heads.append((chrom, int(length)))
    heads = sorted(heads, key=lambda x: x[0])
    bw.addHeader(heads)

    # sort regions and arrs
    assert len(regions) == len(arrs)

    def get_key(x):
        chrom, start, end = re.split("[:-]", regions[x])
        start = int(start)
        return chrom, start

    idx_sort = sorted(range(len(regions)), key=get_key)
    regions = [regions[i] for i in idx_sort]
    arrs = arrs[idx_sort]
    # construct iterables
    it = zip(arrs, regions)
    it = itertools.chain(
        it, zip([np.array([-1000])], ["chrNone:10-100"])
    )  # add pseudo region to make sure the last entry will be added to bw file
    arr, lastRegion = next(it)
    lastChrom, start, end = re.split(r"[:-]", lastRegion)

    start = int(start)
    end = int(end)
    # extend coordinates if attribution arr is larger than interval length
    if end - start < len(arr):
        logger.warning(
            "Interval length is smaller than attribution array length, expand it!"
        )
        diff = len(arr) - (end - start)
        if diff % 2 != 0:
            raise Exception(
                "The difference between attribution array length and interval length is not even! Can't do symmetric extension in this case, exiting..."
            )
        start -= int(diff / 2)
        end += int(diff / 2)
    elif end - start == len(arr):
        diff = 0
    else:
        raise Exception(
            "Interval length is larger than attribution array length, this is not expected situation, exiting..."
        )
    arr_store_tmp = arr
    for arr, region in it:
        rchrom, rstart, rend = re.split(r"[:-]", region)
        rstart = int(rstart)
        rend = int(rend)
        # extend coordinates if attribution arr is larger than interval length
        rstart -= int(diff / 2)
        rend += int(diff / 2)
        if rstart < 0:
            break
        if end <= rstart or rchrom != lastChrom:
            arr_store_tmp = (
                np.convolve(arr_store_tmp, np.ones(10) / 10, mode="same")
                if smooth
                else arr_store_tmp
            )
            try:
                bw.addEntries(
                    lastChrom,
                    np.arange(start, end, dtype=np.int64),
                    values=arr_store_tmp.astype(np.float64),
                    span=1,
                )
            except:
                print(lastChrom)
                print(start)
                print(end)
                print(arr_store_tmp.shape, arr_store_tmp.dtype)
                print(rchrom)
                print(rstart)
                print(rend)
                raise Exception(
                    "Runtime error when adding entries to bigwig file, see above messages for region info"
                )
            lastChrom = rchrom
            start = rstart
            end = rend
            arr_store_tmp = arr
        # get uncovered interval (defined by start coordinate `start` and relative start coordinate `start_idx`)
        else:
            assert (
                end > rstart and rchrom == lastChrom
            )  # just double check make sure two intervals are overlapped
            start_idx = end - rstart
            end = rend
            try:
                arr_store_tmp = np.concatenate([arr_store_tmp, arr[start_idx:]])
            except TypeError:
                print(start_idx)
                print(rstart, rend, rchrom, start, end, lastChrom)
                print(arr_store_tmp.shape, print(arr.shape))
                sys.exit(1)
    bw.close()

