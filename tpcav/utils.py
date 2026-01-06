#!/usr/bin/env python3
"""
Copied utility helpers from scripts/utils.py so the package can run standalone.
"""

import logging
from copy import deepcopy

import Bio
import numpy as np
import pandas as pd
import pyfaidx
import seqchromloader as scl
import torch
from Bio import SeqIO
from pyfaidx import Fasta
from torch.utils.data import default_collate, get_worker_info

logger = logging.getLogger(__name__)


def center_windows(df, window_len=1024):
    "Get center window_len bp region of the given coordinate dataframe."
    halfR = int(window_len / 2)
    df = df.assign(mid=lambda x: ((x["start"] + x["end"]) / 2).astype(int)).assign(
        start=lambda x: x["mid"] - halfR, end=lambda x: x["mid"] + halfR
    )
    if "strand" in df.columns:
        return df[["chrom", "start", "end", "strand"]]
    else:
        return df[["chrom", "start", "end"]]


def collate_seq(batch):
    seq, chrom, target, label = default_collate(batch)
    return seq


def collate_chrom(batch):
    seq, chrom, target, label = default_collate(batch)
    return chrom


def collate_seqchrom(batch):
    seq, chrom, target, label = default_collate(batch)
    return seq, chrom


def seq_dataloader_from_bed(
    seq_bed, genome_fasta, window_len=1024, batch_size=8, num_workers=0
):
    seq_df = pd.read_table(
        seq_bed,
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
    )
    return seq_dataloader_from_dataframe(
        seq_df, genome_fasta, window_len, batch_size, num_workers
    )


def seq_dataloader_from_dataframe(
    seq_df, genome_fasta, window_len=1024, batch_size=8, num_workers=0
):
    seq_df = center_windows(seq_df, window_len=window_len)
    seq_df["label"] = -1
    if "strand" not in seq_df.columns:
        seq_df["strand"] = "+"
    seq_df = scl.filter_chromosomes(seq_df, to_keep=Fasta(genome_fasta).keys())
    dl = scl.SeqChromDatasetByDataFrame(
        seq_df,
        genome_fasta=genome_fasta,
        dataloader_kws={
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": collate_seq,
            "drop_last": True,
        },
    )
    return dl


def seq_dataloader_from_fa(seq_fa, input_window_length=1024, batch_size=8):
    with open(seq_fa) as handle:
        dnaSeqs = []
        for record in SeqIO.parse(handle, "fasta"):
            if len(record.seq) != input_window_length:
                raise Exception(
                    f"Sequence length {len(record.seq)} != input_window_length {input_window_length}"
                )
            dnaSeqs.append(torch.tensor(scl.dna2OneHot(record.seq)))
            if len(dnaSeqs) >= batch_size:
                yield torch.stack(dnaSeqs)
                dnaSeqs = []

        if len(dnaSeqs) > 0:
            yield torch.stack(dnaSeqs)


def chrom_dataloader_from_bed(
    chrom_bed, genome_fasta, input_window_length=1024, bigwigs=None, batch_size=8
):
    chrom_df = pd.read_table(
        chrom_bed,
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
    )
    return chrom_dataloader_from_dataframe(
        chrom_df, genome_fasta, input_window_length, bigwigs or [], batch_size
    )


def chrom_dataloader_from_dataframe(
    chrom_df,
    genome_fasta,
    input_window_length=1024,
    bigwigs=None,
    batch_size=8,
    num_workers=0,
):
    bigwigs = bigwigs or []
    chrom_df = center_windows(chrom_df, window_len=input_window_length)
    chrom_df["label"] = -1
    if "strand" not in chrom_df.columns:
        chrom_df["strand"] = "+"
    chrom_df = scl.filter_chromosomes(chrom_df, to_keep=Fasta(genome_fasta).keys())
    dl = scl.SeqChromDatasetByDataFrame(
        chrom_df,
        genome_fasta=genome_fasta,
        bigwig_filelist=bigwigs,
        dataloader_kws={
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": collate_chrom,
            "drop_last": True,
        },
    )
    return dl


class IterateSeqDataFrame(torch.utils.data.IterableDataset):
    def __init__(
        self,
        seq_df,
        genome_fasta,
        motif=None,
        motif_mode="pwm",
        num_motifs=128,
        start_buffer=0,
        end_buffer=0,
        regions_insert=None,
        return_region=False,
        print_warning=True,
        infinite=False,
    ):
        self.seq_df = seq_df
        self.genome_fasta = genome_fasta
        self.motif = motif
        self.motif_mode = motif_mode
        self.num_motifs = num_motifs
        self.start_buffer = start_buffer
        self.end_buffer = end_buffer
        self.regions_insert = regions_insert
        self.return_region = return_region
        self.print_warning = print_warning
        self.infinite = infinite

    def __iter__(self):
        worker_info = get_worker_info()
        self.genome = pyfaidx.Fasta(self.genome_fasta)
        rng = np.random.default_rng(worker_info.id if worker_info is not None else 1)

        if self.infinite:
            while True:
                chunk = self.seq_df.sample(frac=1.0).reset_index(drop=True)
                try:
                    yield from iterate_seq_df_chunk(
                        chunk,
                        genome=self.genome,
                        motif=self.motif,
                        motif_mode=self.motif_mode,
                        num_motifs=self.num_motifs,
                        start_buffer=self.start_buffer,
                        end_buffer=self.end_buffer,
                        regions_insert=self.regions_insert,
                        batch_size=None,
                        return_region=self.return_region,
                        print_warning=self.print_warning,
                        rng=rng,
                    )
                except StopIteration:
                    continue
        else:
            if worker_info is None:
                chunk = self.seq_df
            else:
                chunk = np.array_split(self.seq_df, worker_info.num_workers)[
                    worker_info.id
                ]
            yield from iterate_seq_df_chunk(
                chunk,
                genome=self.genome,
                motif=self.motif,
                motif_mode=self.motif_mode,
                num_motifs=self.num_motifs,
                start_buffer=self.start_buffer,
                end_buffer=self.end_buffer,
                regions_insert=self.regions_insert,
                batch_size=None,
                return_region=self.return_region,
                print_warning=self.print_warning,
                rng=rng,
            )


def iterate_seq_df_chunk(
    chunk,
    genome,
    motif=None,
    motif_mode="pwm",
    num_motifs=128,
    start_buffer=0,
    end_buffer=0,
    regions_insert=None,
    return_region=False,
    batch_size=None,
    print_warning=True,
    rng=np.random.default_rng(1),
):
    seqs = []
    regions = []
    for item in chunk.itertuples():
        try:
            seq = str(genome[item.chrom][item.start : item.end]).upper()
        except KeyError:
            if print_warning:
                print(f"catch KeyError in region {item.chrom}:{item.start}-{item.end}")
            continue
        except pyfaidx.FetchError:
            if print_warning:
                print(
                    f"catch FetchError in region {item.chrom}:{item.start}-{item.end}, probably start coordinate negative"
                )
            continue
        unique_chars = np.unique(list(seq))
        if "N" in seq:
            if print_warning:
                print(f"Skip {item.chrom}:{item.start}-{item.end} due to containing N")
            continue
        elif len(unique_chars) == 0:
            if print_warning:
                print(
                    f"Skip region {item.chrom}:{item.start}-{item.end} due to no sequences available"
                )
            continue

        if motif is not None:
            seq = insert_motif_into_seq(
                seq,
                motif,
                num_motifs=num_motifs,
                start_buffer=start_buffer,
                end_buffer=end_buffer,
                mode=motif_mode,
                rng=rng,
            )
        elif regions_insert is not None:
            seq = insert_region_into_seq(seq, regions_insert, genome, rng=rng)

        if batch_size is None:
            if return_region:
                yield f"{item.chrom}:{item.start}-{item.end}", seq
            else:
                yield seq
        else:
            seqs.append(seq)
            regions.append(f"{item.chrom}:{item.start}-{item.end}")
            if len(seqs) >= batch_size:
                if return_region:
                    yield regions, seqs
                else:
                    yield seqs
                seqs = []
                regions = []
    if (len(seqs) > 0) and batch_size is not None:
        if return_region:
            yield regions, seqs
        else:
            yield seqs


def iterate_seq_df(
    seq_df,
    genome_fasta,
    motif=None,
    num_motifs=128,
    motif_mode="pwm",
    start_buffer=0,
    end_buffer=0,
    regions_insert=None,
    batch_size=32,
    return_region=False,
    print_warning=True,
    rng=np.random.default_rng(1),
):
    genome = Fasta(genome_fasta)
    yield from iterate_seq_df_chunk(
        chunk=seq_df,
        genome=genome,
        motif=motif,
        num_motifs=num_motifs,
        motif_mode=motif_mode,
        start_buffer=start_buffer,
        end_buffer=end_buffer,
        regions_insert=regions_insert,
        return_region=return_region,
        batch_size=batch_size,
        print_warning=print_warning,
        rng=rng,
    )


def iterate_seq_bed(
    seq_bed,
    genome_fasta,
    motif=None,
    num_motifs=128,
    motif_mode="pwm",
    start_buffer=0,
    end_buffer=0,
    regions_insert=None,
    batch_size=32,
    print_warning=True,
    rng=np.random.default_rng(1),
):
    seq_df = pd.read_table(seq_bed, usecols=range(3), names=["chrom", "start", "end"])
    yield from iterate_seq_df(
        seq_df,
        genome_fasta,
        motif=motif,
        motif_mode=motif_mode,
        num_motifs=num_motifs,
        start_buffer=start_buffer,
        end_buffer=end_buffer,
        regions_insert=regions_insert,
        batch_size=batch_size,
        print_warning=print_warning,
        rng=rng,
    )


class SeqChromConcept:
    "Sequence + Chromatin concept given the bed files of sequences and chromatin regions."

    def __init__(
        self,
        seq_bed,
        seq_fa,
        chrom_bed,
        genome_fasta,
        bws: list,
        transforms=None,
        window_len=1024,
        batch_size=8,
    ):
        self.seq_bed = seq_bed
        self.seq_fa = seq_fa
        self.chrom_bed = chrom_bed
        self.seq_dl = None
        self.chrom_dl = None

        self.genome_fasta = genome_fasta
        self.bws = bws
        self.transforms = transforms
        self.window_len = window_len
        self.batch_size = batch_size

    def seq_dataloader(self):
        if self.seq_bed is not None:
            seq_df = pd.read_table(
                self.seq_bed,
                header=None,
                usecols=[0, 1, 2],
                names=["chrom", "start", "end"],
            )
            seq_df = center_windows(seq_df, window_len=self.window_len)
            seq_df["label"] = -1
            if "strand" not in seq_df.columns:
                seq_df["strand"] = "+"
            seq_df = scl.filter_chromosomes(
                seq_df, to_keep=Fasta(self.genome_fasta).keys()
            )
            dl = scl.SeqChromDatasetByDataFrame(
                seq_df,
                genome_fasta=self.genome_fasta,
                dataloader_kws={
                    "batch_size": self.batch_size,
                    "num_workers": 0,
                    "collate_fn": collate_seq,
                    "drop_last": True,
                },
            )
            for seq in dl:
                if isinstance(seq, list):
                    assert len(seq) == 1
                    seq = seq[0]
                yield seq
        else:
            with open(self.seq_fa) as handle:
                dnaSeqs = []
                for record in SeqIO.parse(handle, "fasta"):
                    dnaSeqs.append(torch.tensor(scl.dna2OneHot(record.seq)))
                    if len(dnaSeqs) >= self.batch_size:
                        yield torch.stack(dnaSeqs)
                        dnaSeqs = []
                if len(dnaSeqs) > 0:
                    yield torch.stack(dnaSeqs)

    def chrom_dataloader(self):
        chrom_df = pd.read_table(
            self.chrom_bed,
            header=None,
            usecols=[0, 1, 2],
            names=["chrom", "start", "end"],
        )
        chrom_df = center_windows(chrom_df, window_len=self.window_len)
        chrom_df["label"] = -1
        if "strand" not in chrom_df.columns:
            chrom_df["strand"] = "+"
        chrom_df = scl.filter_chromosomes(
            chrom_df, to_keep=Fasta(self.genome_fasta).keys()
        )
        dl = scl.SeqChromDatasetByDataFrame(
            chrom_df,
            genome_fasta=self.genome_fasta,
            bigwig_filelist=self.bws,
            transforms=self.transforms,
            dataloader_kws={
                "batch_size": self.batch_size,
                "num_workers": 0,
                "collate_fn": collate_chrom,
                "drop_last": True,
            },
        )
        yield from dl

    def dataloader(self):
        if (
            self.seq_bed is not None or self.seq_fa is not None
        ) and self.seq_dl is None:
            self.seq_dl = self.seq_dataloader()
        if (self.chrom_bed is not None) and self.chrom_dl is None:
            self.chrom_dl = self.chrom_dataloader()
        yield from zip(self.seq_dl, self.chrom_dl)


def sample_from_pwm(motif, n_seqs=1, rng=None):
    """
    Draw `n_seqs` independent sequences from a Bio.motifs.Motif PWM.

    Parameters
    ----------
    motif  : Bio.motifs.Motif
        Motif whose `.pwm` is used for sampling.
    n_seqs : int, default 1
        How many sequences to generate.
    rng    : numpy.random.Generator or None
        Leave None for np.random.default_rng().

    Returns
    -------
    str | list[str]
        A single string if n_seqs==1, otherwise a list of strings.
    """
    rng = rng or np.random.default_rng()

    # ---- 1. Build a (L, A) probability matrix --------------------------------
    alphabet = list(motif.alphabet)  # e.g. ['A', 'C', 'G', 'T']
    L = motif.length
    pwm_dict = motif.pwm  # dict base → list(float)

    # shape (L, A) with rows = positions, cols = alphabet order
    prob_mat = np.column_stack([pwm_dict[b] for b in alphabet])

    # ---- 2. Vectorised multinomial sampling -----------------------------------
    # Draw U(0,1) numbers of shape (n_seqs, L)
    u = rng.random((n_seqs, L))

    # cumulative probabilities along alphabet axis
    cum = np.cumsum(prob_mat, axis=1)  # still (L, A)

    # Broadcast cum to (n_seqs, L, A) and pick first index where cum > u
    idx = (u[..., None] < cum).argmax(axis=2)  # (n_seqs, L) int indices

    # ---- 3. Convert indices back to letters -----------------------------------
    letters = np.array(alphabet, dtype="U1")
    seq_arr = letters[idx]  # (n_seqs, L) array of chars

    # Join per sequence
    seqs = ["".join(row) for row in seq_arr]
    return seqs[0] if n_seqs == 1 else seqs


def insert_motif_into_seq(
    seq,
    motif,
    num_motifs=3,
    start_buffer=50,
    end_buffer=50,
    rng=np.random.default_rng(1),
    mode="consensus",
):
    assert mode in ["consensus", "pwm"]

    seq_ins = list(deepcopy(seq))
    pos_motif_overlap = np.ones(len(seq))
    pos_motif_overlap[:start_buffer] = 0
    pos_motif_overlap[(-end_buffer - len(motif)) :] = 0
    num_insert_motifs = 0
    for i in range(num_motifs):
        try:
            motif_start = rng.choice(
                np.where(pos_motif_overlap > 0)[0]
            ).item()  # randomly pick insert location
        except ValueError:
            # print(
            #    f"No samples can be taken for motif {motif.name}, skip inserting the rest of motifs"
            # )
            break
        if isinstance(motif, PairedMotif):
            seq_ins[motif_start : (motif_start + len(motif.motif1))] = (
                list(motif.motif1.consensus)
                if mode == "consensus"
                else sample_from_pwm(motif.motif1)
            )
            seq_ins[
                (motif_start + len(motif.motif1) + motif.spacing) : (
                    motif_start + len(motif)
                )
            ] = (
                list(motif.motif2.consensus)
                if mode == "consensus"
                else sample_from_pwm(motif.motif2)
            )
        else:
            seq_ins[motif_start : (motif_start + len(motif))] = (
                list(motif.consensus) if mode == "consensus" else sample_from_pwm(motif)
            )

        pos_motif_overlap[(motif_start - len(motif)) : (motif_start + len(motif))] = 0
        num_insert_motifs += 1
    if num_insert_motifs < num_motifs:
        logger.warning(
            f"Only inserted {num_insert_motifs} out of {num_motifs} motifs for motif {motif.name}"
        )
    return "".join(seq_ins)


class CustomMotif:
    def __init__(self, name, consensus):
        self.name = name
        self.matrix_id = "custom"
        self.consensus = consensus.upper()
        self.rc = False

    def __len__(self):
        return len(self.consensus)

    def reverse_complement(self):
        self.consensus = Bio.Seq.reverse_complement(self.consensus)
        self.name = self.name + "_rc"
        return self


class PairedMotif:
    def __init__(self, motif1, motif2, spacing=0):
        self.motif1 = motif1
        self.motif2 = motif2
        self.rc = False
        self.spacing = spacing
        self.pname = f"{self.motif1.name}_and_{self.motif2.name}"
        self.pmatrix_id = f"{self.motif1.matrix_id}_and_{self.motif2.matrix_id}"

    def reverse_complement(self):
        self.rc = True
        self.motif1 = self.motif1.reverse_complement()
        self.motif2 = self.motif2.reverse_complement()
        return self

    @property
    def name(self):
        return self.pname

    @property
    def matrix_id(self):
        return self.pmatrix_id

    def __len__(self):
        return len(self.motif1) + len(self.motif2) + self.spacing
