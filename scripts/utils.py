#!/usr/bin/env python3

"""
Utility functions of loading model, dataset, etc.
"""

from itertools import cycle

import numpy as np
import pandas as pd
import pyfaidx
import seqchromloader as scl
import torch
from Bio import SeqIO
from deeplift.dinuc_shuffle import dinuc_shuffle
from pybedtools import BedTool
from pyfaidx import Fasta
from seq_utils import insert_motif_into_seq, insert_region_into_seq
from seqchromloader.utils import dna2OneHot
from torch.utils.data import TensorDataset, default_collate, get_worker_info


def seq_transform_fn(seq_one_hot):
    "TODO: modify this function if your model input is not one hot coded, seq_one_hot is one hot coded DNA by the order of [A, C, G, T], its shape is [batch_size, 4, len]"

    return seq_one_hot


def load_model():
    "TODO: Please load your model here"

    return model


def expand_dataframe(df, target_len=196_608):
    "expand interval dataframe to target length"
    halfR = int(target_len / 2)
    df = df.assign(mid=lambda x: ((x["start"] + x["end"]) / 2).astype(int)).assign(
        start=lambda x: x["mid"] - halfR, end=lambda x: x["mid"] + halfR
    )  # center 2*halfR window
    return df[["chrom", "start", "end"]]


def split_dataframe(df, chunk_size=10000):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size : (i + 1) * chunk_size])
    return chunks


def iterate_seq_df_chunk(
    chunk,
    genome,
    motif=None,
    motif_mode="pwm",
    regions_insert=None,
    num_motifs=128,
    start_buffer=0,
    end_buffer=0,
    return_region=False,
    batch_size=None,
    print_warning=True,
    rng=np.random.default_rng(1),
):
    "Core function to generate motif concepts given a motif instance and a dataframe of regions"

    seq_one_hots = []
    regions = []
    for item in chunk.itertuples():
        try:
            seq = str(genome[item.chrom][item.start : item.end]).upper()
        except KeyError as e:
            if print_warning:
                print(f"catch KeyError in region {item.chrom}:{item.start}-{item.end}")
            continue
        except pyfaidx.FetchError as e:
            if print_warning:
                print(
                    f"catch FetchError in region {item.chrom}:{item.start}-{item.end}, probably it's due to the start coordinate is negative"
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
                    f"Skip region {item.chrom}:{item.start}-{item.end} due to no sequences avaiable"
                )
            continue

        # if motif is not None, insert motif sequence
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
            seq_one_hot = dna2OneHot(seq)
            if return_region:
                yield f"{item.chrom}:{item.start}-{item.end}", seq_one_hot
            else:
                yield seq_one_hot
        else:
            seq_one_hots.append(dna2OneHot(seq))
            regions.append(f"{item.chrom}:{item.start}-{item.end}")
            if len(seq_one_hots) >= batch_size:
                seq_one_hots = torch.stack(seq_one_hots)
                if return_region:
                    yield regions, seq_one_hots
                else:
                    yield seq_one_hots
                seq_one_hots = []
                regions = []
    if (len(seq_one_hots) > 0) and batch_size is not None:
        seq_one_hots = torch.stack(seq_one_hots)
        if return_region:
            yield regions, seq_one_hots
        else:
            yield seq_one_hots


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
                chunk = self.seq_df.sample(frac=1.0).reset_index(
                    drop=True
                )  # shuffle the dataframe every time
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


def center_windows(df, window_len=1024):
    "Get center window_len bp region of the given coordinate dataframe, input dataframe must have start and end colums"
    halfR = int(window_len / 2)
    return df.assign(mid=lambda x: ((x["start"] + x["end"]) / 2).astype(int)).assign(
        start=lambda x: x["mid"] - halfR, end=lambda x: x["mid"] + halfR
    )


def collate_seq(batch):
    seq, chrom, target, label = default_collate(batch)
    return seq


def collate_chrom(batch):
    seq, chrom, target, label = default_collate(batch)
    return chrom


def collate_seqchrom(batch):
    seq, chrom, target, label = default_collate(batch)
    return seq, chrom


def seqs_to_tensords(seqs):
    "Turn a list of DNA sequences into one-hot coded tensordataset"
    onehots = []
    for s in seqs:
        onehots.append(scl.dna2OneHot(s))
    return TensorDataset(torch.tensor(np.stack(onehots)))


def preprocess(x, y, device="cpu"):
    if isinstance(x, list):
        x = x[0]
    if isinstance(y, list):
        y = y[0]
    return x.to(device), y.to(device)


class WrappedDataLoader:
    def __init__(self, dl1, dl2, func):
        self.dl1 = dl1
        self.dl2 = dl2
        self.func = func

    def __len__(self):
        return len(self.dl1.dataset.dataframe)

    def __iter__(self):
        if len(self.dl1) > len(self.dl2):
            dl1 = self.dl1
            dl2 = cycle(self.dl2)
        else:
            dl1 = cycle(self.dl1)
            dl2 = self.dl2
        for b1, b2 in zip(dl1, dl2):
            yield (self.func(b1, b2))


class SeqChromConcept:
    "Sequence + Chromatin concept given the bed files of sequences and chromatin regions"

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
            seq_df["strand"] = "+"
            print(f"Filtering out concept samples that don't exist in the genome...")
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
                    if len(record.seq) != 1024:
                        print(len(record.seq))
                    if len(dnaSeqs) >= self.batch_size:
                        yield torch.stack(dnaSeqs)
                        dnaSeqs = []

    def chrom_dataloader(self):
        chrom_df = pd.read_table(
            self.chrom_bed,
            header=None,
            usecols=[0, 1, 2],
            names=["chrom", "start", "end"],
        )
        chrom_df = center_windows(chrom_df, window_len=self.window_len)
        chrom_df["label"] = -1
        chrom_df["strand"] = "+"
        # print(f"Filtering out concept samples that don't exist in the genome...")
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


def dinuc_shuffle_several_times_seqchrom(seqchrom, times: int = 10, seed: int = 1):
    seq, chrom = seqchrom
    seq_to_return = dinuc_shuffle_several_times_seq(seq, times, seed)
    chrom_to_return = torch.stack(
        [torch.zeros_like(i) for j in range(times) for i in chrom]
    )
    return (seq_to_return, chrom_to_return)


def dinuc_shuffle_several_times_seq(seq, times: int = 10, seed: int = 1):
    assert len(seq.shape) == 3  # dim: N x D x L
    onehot_seq = torch.permute(seq, (0, 2, 1))  # dim: N x L x D
    assert onehot_seq.shape[-1] == 4
    device = onehot_seq.device
    to_returns = []
    for s in onehot_seq:
        # reset RandomState every loop to make it the same as v2 behavior on each example
        rng = np.random.RandomState(seed)
        to_return = torch.tensor(
            np.array(
                [
                    dinuc_shuffle(s.detach().cpu().numpy(), rng=rng).T
                    for i in range(times)
                ]
            )
        ).to(device)
        to_returns.append(to_return)
    return torch.cat(to_returns)


def create_dataloader_from_bed(
    bed, data_config, window=1024, batch_size=8, transforms=None
):
    halfR = int(window / 2)

    target_df = BedTool(bed).to_dataframe()
    target_df = target_df.assign(
        mid=lambda x: ((x["start"] + x["end"]) / 2).astype(int)
    ).assign(start=lambda x: x["mid"] - halfR, end=lambda x: x["mid"] + halfR)
    target_df["label"] = -1
    target_df["strand"] = "+"
    target_dl = scl.SeqChromDatasetByDataFrame(
        target_df,
        genome_fasta=data_config["genome_fasta_file"],
        bigwig_filelist=data_config["pre_bws"],
        transforms=transforms,
        return_region=True,
        dataloader_kws={"batch_size": batch_size, "drop_last": False},
    )

    return target_dl
