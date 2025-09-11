#!/usr/bin/env python3

"""
Generate motif concepts for TCAV
"""

from copy import deepcopy

import Bio
import numpy as np
import seqchromloader as scl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from deeplift.dinuc_shuffle import dinuc_shuffle

scl.mute_warning()


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
        return self


def saturate_motif_in_seq(seq, motif, start=0, gap=10, mode="consensus"):
    "Insert motif into sequence in a tiling way"
    seq_ins = list(deepcopy(seq))
    for i in range(start, len(seq), len(motif) + gap):
        if i + len(motif) <= len(seq):
            seq_ins[i : (i + len(motif))] = (
                list(motif.consensus) if mode == "consensus" else sample_from_pwm(motif)
            )
    assert len(seq_ins) == len(seq), "Length of sequence is changed!"
    return "".join(seq_ins)


def insert_region_into_seq(
    seq,
    regions_df,
    genome,
    rng=np.random.default_rng(1),
):
    "Insert a sample sequence from the given regions dataframe into the sequence"
    seq_ins = list(deepcopy(seq))
    region_sampled = regions_df.sample(1, random_state=rng)

    region_sampled_seq = genome[region_sampled.chrom][
        region_sampled.start : region_sampled.end
    ]
    if getattr(region_sampled, "strand", "+") == "-":
        region_sampled_seq = region_sampled_seq.reverse_complement()

    region_sampled_seq = str(region_sampled_seq).upper()

    if len(region_sampled_seq) >= len(seq_ins):
        return region_sampled_seq[: len(seq_ins)]
    else:
        start = rng.integers(low=0, high=len(seq_ins) - len(region_sampled_seq))
        seq[start : (start + len(region_sampled_seq))] = list(region_sampled_seq)
        return "".join(seq_ins)


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
    if num_insert_motifs < num_motifs * 0.5:
        print(
            "Less than 50% of specified # motifs inserted, make sure this is the desired behavior, consider increasing the length of the sequence or decreasing the number of motifs"
        )
    return "".join(seq_ins)


def dinucleotide_shuffle_seq(dnaSeqs):
    "Dinucleotide shuffle the given DNA sequences"
    records = []
    for idx, record in enumerate(dnaSeqs):
        seq = str(record.seq)
        seq_shuf = dinuc_shuffle(seq)
        records.append(
            SeqRecord(
                Seq(seq_shuf),
                id=f"dinucleotide shuffle idx",
                description=f"dinucleotide shuffled dna seqs inserted by motifs",
            )
        )
    return records
