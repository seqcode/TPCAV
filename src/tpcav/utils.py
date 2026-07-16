#!/usr/bin/env python3
"""
Copied utility helpers from scripts/utils.py so the package can run standalone.
"""

import logging
import tempfile
from copy import deepcopy

import Bio
import numpy as np
import pandas as pd
import pyfaidx
import seqchromloader as scl
import torch
from Bio import SeqIO, motifs
from pyfaidx import Fasta
from torch.utils.data import default_collate, get_worker_info
logger = logging.getLogger(__name__)
try:
    from tpcav import rust_optim as _rust_optim
    _RUST_AVAILABLE = True
    logger.info("RUST backend available")
except ImportError:
    _rust_optim = None
    _RUST_AVAILABLE = False


def clean_motif_name(motif_name):
    return motif_name.replace("/", "-")

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
        num_motifs=128,
        start_buffer=0,
        end_buffer=0,
        return_region=False,
        print_warning=True,
        infinite=False,
    ):
        self.seq_df = seq_df
        self.genome_fasta = genome_fasta
        self.motif = motif
        self.num_motifs = num_motifs
        self.start_buffer = start_buffer
        self.end_buffer = end_buffer
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
                        num_motifs=self.num_motifs,
                        start_buffer=self.start_buffer,
                        end_buffer=self.end_buffer,
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
                chunk = [self.seq_df.iloc[idx] for idx in np.array_split(np.arange(len(self.seq_df)), worker_info.num_workers)][worker_info.id]
            yield from iterate_seq_df_chunk(
                chunk,
                genome=self.genome,
                motif=self.motif,
                num_motifs=self.num_motifs,
                start_buffer=self.start_buffer,
                end_buffer=self.end_buffer,
                batch_size=None,
                return_region=self.return_region,
                print_warning=self.print_warning,
                rng=rng,
            )


def iterate_seq_df_chunk(
    chunk,
    genome,
    motif=None,
    num_motifs=128,
    start_buffer=0,
    end_buffer=0,
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
                rng=rng,
            )

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
    start_buffer=0,
    end_buffer=0,
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
        start_buffer=start_buffer,
        end_buffer=end_buffer,
        return_region=return_region,
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


def _sample_from_pwm_numpy(pwm_prob_mat, n_seqs=1, rng=None, alphabet=['A', 'C', 'G', 'T']):
    """Pure numpy implementation of sample_from_pwm."""
    rng = rng or np.random.default_rng()
    L = pwm_prob_mat.shape[0]
    u = rng.random((n_seqs, L))
    cum = np.cumsum(pwm_prob_mat, axis=1)
    idx = (u[..., None] < cum).argmax(axis=2)
    letters = np.array(alphabet, dtype="U1")
    seq_arr = letters[idx]
    seqs = ["".join(row) for row in seq_arr]
    return seqs[0] if n_seqs == 1 else seqs

def _sample_from_pwm_rust(pwm_prob_mat, n_seqs=1, rng=None):
    seed = int(rng.integers(2**63)) if rng is not None else None
    seqs = _rust_optim.sample_from_pwm(
        np.ascontiguousarray(pwm_prob_mat, dtype=np.float64),
        n_seqs=n_seqs,
        seed=seed,
    )
    return seqs[0] if n_seqs == 1 else seqs

def sample_from_pwm(pwm_prob_mat, n_seqs=1, rng=None, alphabet=['A', 'C', 'G', 'T']):
    """
    Draw `n_seqs` independent sequences from a position weight matrix.

    Uses the Rust backend (rust_optim) when available for ~7x speedup; falls
    back to a pure-numpy implementation otherwise.

    Parameters
    ----------
    pwm_prob_mat  : (L, A) probability matrix
    n_seqs : int, default 1
        How many sequences to generate.
    rng    : numpy.random.Generator or None
        Used only by the numpy fallback path. Leave None for default_rng().
    alphabet  : letters to be used

    Returns
    -------
    str | list[str]
        A single string if n_seqs==1, otherwise a list of strings.
    """
    if _RUST_AVAILABLE and alphabet == ['A', 'C', 'G', 'T']:
        return _sample_from_pwm_rust(pwm_prob_mat, n_seqs=n_seqs, rng=rng)
    else:
        return _sample_from_pwm_numpy(pwm_prob_mat, n_seqs=n_seqs, rng=rng, alphabet=alphabet)

def get_prob_mat_from_motif_pwm(pwm_dict, alphabet=['A', 'C', 'G', 'T']):
    pwm_prob_mat = np.column_stack([pwm_dict[b] for b in alphabet])
    return pwm_prob_mat


def insert_motif_into_seq(
    seq,
    motif,
    num_motifs=3,
    start_buffer=50,
    end_buffer=50,
    rng=np.random.default_rng(1),
):
    
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
        seq_ins[motif_start : (motif_start + len(motif))] = motif.sample_instance()

        pos_motif_overlap[(motif_start - len(motif)) : (motif_start + len(motif))] = 0
        num_insert_motifs += 1
    if num_insert_motifs < num_motifs:
        logger.warning(
            f"Only inserted {num_insert_motifs} out of {num_motifs} motifs for motif {motif.name}"
        )
    return "".join(seq_ins)


class ConsensusMotif:
    def __init__(self, name, consensus):
        self.name = name
        self.matrix_id = "custom"
        self.consensus = consensus.upper()

    def __len__(self):
        return len(self.consensus)

    def reverse_complement(self):
        rc = deepcopy(self)
        rc.consensus = Bio.Seq.reverse_complement(self.consensus)
        rc.name = self.name + "_rc"
        return rc

    def sample_instance(self):
        return self.consensus

class PermutedConsensusMotif:
    def __init__(self, name, consensus, seed=None, min_shift=0.3):
        self.name = name
        self.consensus = consensus.upper()
        self.rng = np.random.default_rng(seed)
        self.min_shift = min_shift

    def __len__(self):
        return len(self.consensus)

    def reverse_complement(self):
        rc = deepcopy(self)
        rc.consensus = Bio.Seq.reverse_complement(self.consensus)
        rc.name = self.name + "_rc"
        return rc

    def sample_instance(self):
        return self._permute(self.consensus)

    def _permute(self, max_attempts=100):
        """
        Permute the consensus sequence, return the object
        """
        permuted = deepcopy(self)

        L = len(self.consensus)
        
        count = 0
        while True:
            perm = self.rng.permutation(L)
            frac_moved = np.mean(perm != np.arange(L))
            if frac_moved >= self.min_shift:
                break
            else:
                count += 1
            if count > max_attempts:
                raise ValueError(
                    f"Could not generate a permutation with min_shift={self.min_shift} for motif {self.name}"
                )
        permuted_consensus = ''.join([self.consensus[i] for i in perm])

        return permuted_consensus

class PermutedPWMMotif:
    BASES = ("A", "C", "G", "T") # alphabet must be symmetric!
    RC_MAP = {"A": "T", "C": "G", "G": "C", "T": "A"}

    def __init__(self, motif, seed=None, min_shift=0.3, name_suffix="_perm"):
        """
        motif: Bio.motifs.Motif
        seed: RNG seed
        min_shift: fraction of positions that must move
        """
        self.original_motif = motif
        self.name = motif.name + name_suffix if motif.name else "permuted_motif"
        self.length = motif.length
        self.alphabet = motif.alphabet
        self.min_shift = min_shift
        self.rng = np.random.default_rng(seed)

        # extract PWM as dict of lists
        pwm = {b: list(motif.pwm[b]) for b in self.BASES}
        self.pwm_prob_mat = get_prob_mat_from_motif_pwm(pwm)

    def __len__(self):
        return self.length

    def _permute_pwm_positions(self, max_attempts=100):
        L = self.length

        count = 0
        while True:
            perm = self.rng.permutation(L)
            frac_moved = np.mean(perm != np.arange(L))
            if frac_moved >= self.min_shift:
                break
            else:
                count += 1
            if count > max_attempts:
                raise ValueError(
                    f"Could not generate a permutation with min_shift={self.min_shift} for motif {self.original_motif.name}"
                )

        permuted_pwm_prob_mat = self.pwm_prob_mat[perm, :]

        return permuted_pwm_prob_mat

    def reverse_complement(self):
        """
        Return a NEW PermutedMotif with reverse-complemented PWM
        """
        rc = deepcopy(self)
        rc.pwm_prob_mat = self.pwm_prob_mat[::-1, ::-1]
        rc.name = self.name + "_rc"
        return rc

    def sample_instance(self):
        return sample_from_pwm(self._permute_pwm_positions(), rng=self.rng)

class BioMotifWrapped:
    def __init__(self, motif, seed=None):
        """
        Wrapper class for Bio Motif
        """
        self.motif = motif
        self.name = motif.name
        self.length = len(motif.consensus)
        self.pwm_prob_mat = get_prob_mat_from_motif_pwm(motif.pwm)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.length

    def reverse_complement(self):
        rc = deepcopy(self)
        rc.motif = self.motif.reverse_complement()
        rc.name = self.motif.name + "_rc"
        rc.pwm_prob_mat = get_prob_mat_from_motif_pwm(rc.motif.pwm)

        return rc

    def sample_instance(self):
        return sample_from_pwm(self.pwm_prob_mat, rng=self.rng)

def write_meme_v4(motifs_dict, output_file, background=0.25):
    """
    motifs_dict: {name: numpy array of shape (L, 4)} with columns order A, C, G, T
    """
    with open(output_file, "w") as f:
        f.write("MEME version 4\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("strands: + -\n\n")
        f.write(f"Background letter frequencies\n")
        f.write(f"A {background} C {background} G {background} T {background}\n\n")

        for name, pwm in motifs_dict.items():
            npos = pwm.shape[0]
            f.write(f"MOTIF {name}\n\n")
            f.write(f"letter-probability matrix: alength= 4 w= {npos} nsites= 20 E= 0\n")
            for row in pwm:
                f.write(f"  {row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}  {row[3]:.6f}\n")
            f.write("\n")

def merge_meme_files(meme_files, background=0.25):
    """
    Merge multiple MEME v4 files into a single temporary MEME v4 file.
    
    Args:
        meme_files: list of paths to MEME files
        background: background frequency for each nucleotide
    
    Returns:
        path to temporary merged MEME file
    """
    all_motifs = {}

    for meme_file in meme_files:
        with open(meme_file) as f:
            meme = motifs.parse(f, fmt="minimal")
        for motif in meme:
            # convert biopython counts to probability numpy array (L, 4)
            pwm = motif.pwm
            arr = np.array([pwm[base] for base in "ACGT"]).T  # (L, 4)

            # handle duplicate names
            name = motif.name
            unique_name = name
            counter = 1
            while unique_name in all_motifs:
                unique_name = f"{name}_{counter}"
                counter += 1
            all_motifs[unique_name] = arr

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".meme",
        delete=False
    )
    tmp_path = tmp.name
    tmp.close()

    write_meme_v4(all_motifs, tmp_path, background=background)

    return tmp_path

def merge_consensus_motif_files(motif_files):
    """
    Merge multiple text motif files by concatenation into a temporary file.
    
    Args:
        motif_files: list of paths to text motif files
    
    Returns:
        path to temporary merged text motif file
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False
    )
    tmp_path = tmp.name

    with tmp:
        for motif_file in motif_files:
            with open(motif_file) as f:
                content = f.read()
            tmp.write(content)
            if not content.endswith("\n"):
                tmp.write("\n")

    return tmp_path
