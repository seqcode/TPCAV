#!/usr/bin/env python3

"""
Generate motif concepts for TCAV
"""

import argparse
import os
from collections import defaultdict
from copy import deepcopy
from multiprocessing import Pool

import Bio
import numpy as np
import pandas as pd
import seqchromloader as scl
from Bio import SeqIO, motifs
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from deeplift.dinuc_shuffle import dinuc_shuffle
from pyfaidx import Fasta
from pyjaspar import jaspardb
from yaml import safe_dump

scl.mute_warning()


def generate_random_seq(l: int, gc_ratio=0.4, seed=1, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed)  # create rng if need

    a_or_t = (1 - gc_ratio) / 2
    g_or_c = gc_ratio / 2
    rs = "".join(
        rng.choice(
            ["A", "T", "C", "G"], l, replace=True, p=[a_or_t, a_or_t, g_or_c, g_or_c]
        )
    )

    return rs


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


def insert_motif_into_seq(
    seq,
    motif,
    num_motifs=3,
    start_buffer=50,
    end_buffer=50,
    rng=np.random.default_rng(1),
    mode="consensus",
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


def generate_motif_concept_given_random_regs(
    motif,
    random_regs,
    genome_fasta: str,
    l: int,
    start_buffer=50,
    end_buffer=50,
    rep=1,
    num_motifs=1,
    seed=1,
    rng=None,
    mode="consensus",
):
    "Generate concept example tensors for given motif"
    assert mode in ["consensus", "pwm"]
    if rng is None:
        rng = np.random.default_rng(seed)  # generate rng if need

    # convert region into dna sequence, insert given motif
    genome = Fasta(genome_fasta)
    dnaSeqs = []
    for idx, item in enumerate(random_regs.itertuples()):
        seq = genome[item.chrom][item.start : item.end]
        if len(seq) != l:
            print(f"Sequence length {len(seq)} != specified length {l}, skip")
            continue
        seq = seq.seq
        for r in range(rep):
            seq_ins = insert_motif_into_seq(
                seq,
                motif,
                num_motifs=num_motifs,
                start_buffer=start_buffer,
                end_buffer=end_buffer,
                rng=rng,
                mode=mode,
            )

            record = SeqRecord(
                Seq("".join(seq_ins)),
                id=f"{idx}_{item.chrom}:{item.start}-{item.end}_rep_{r}",
                description=f"Random genomic regions with {num_motifs} {motif.name} motif inserted",
            )
            dnaSeqs.append(record)

    return dnaSeqs


def generate_random_seq_concept(l: int, n: int, genome_fasta: str, seed=1, rng=None):
    # generate rng if need
    if rng is None:
        rng = np.random.default_rng(seed)

    random_regs = scl.random_coords(gs=f"{genome_fasta}.fai", l=l, n=n, seed=seed)
    # convert region into dna sequence
    genome = Fasta(genome_fasta)
    dnaSeqs = []
    randRegs = []
    for idx, item in enumerate(random_regs.itertuples()):
        seq = genome[item.chrom][item.start : item.end]
        if len(seq) != l:
            print(f"Sequence length {len(seq)} != specified length {l}, skip")
            continue
        if rng.integers(2) == 0:
            seq = seq.reverse.complement

        # skip random seqs with only N
        unique_letters = np.unique(list(seq.seq.upper()))
        if len(unique_letters) == 1 and unique_letters[0] == "N":
            continue

        record = SeqRecord(
            Seq(seq.seq),
            id=f"{idx}_{item.chrom}:{item.start}-{item.end}",
            description=f"Random genomic regions",
        )
        dnaSeqs.append(record)
        randRegs.append(item)

    return pd.DataFrame(randRegs)[["chrom", "start", "end"]], dnaSeqs


def generate_pure_random_seq_concept(l: int, n: int, gc_ratio=0.4, seed=1, rng=None):
    # generate rng if need
    if rng is None:
        rng = np.random.default_rng(seed)

    a_or_t = (1 - gc_ratio) / 2
    g_or_c = gc_ratio / 2
    randomSeqs = []
    for idx in range(n):
        rs = "".join(
            rng.choice(
                ["A", "T", "C", "G"],
                l,
                replace=True,
                p=[a_or_t, a_or_t, g_or_c, g_or_c],
            )
        )
        record = SeqRecord(
            Seq(rs), id=f"{idx}_random_seq", description=f"Random sequences"
        )
        randomSeqs.append(record)

    return randomSeqs


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


def write_motif_concept_given_random_regs(
    motif,
    random_regs,
    rep_id,
    num_motifs,
    genome_fasta,
    length,
    mode,
    output_dir,
    seed,
    rng,
    start_buffer=50,
    end_buffer=50,
):
    # insert motifs into given random regions
    dnaSeqs = generate_motif_concept_given_random_regs(
        motif,
        random_regs,
        genome_fasta,
        l=length,
        start_buffer=start_buffer,
        end_buffer=end_buffer,
        mode=mode,
        rep=1,
        num_motifs=num_motifs,
        seed=seed,
        rng=rng,
    )
    if hasattr(motif, "matrix_id"):
        fasta_file = f"{output_dir}/{motif.name.replace('/', '-')}_{motif.matrix_id}_motif_rep{rep_id}_seq.fa"
    else:
        fasta_file = (
            f"{output_dir}/{motif.name.replace('/', '-')}_motif_rep{rep_id}_seq.fa"
        )
    SeqIO.write(dnaSeqs, fasta_file, format="fasta")

    return fasta_file

    # rdnaSeqs = generate_motif_concept_given_random_regs(motif.reverse_complement(), random_regs, genome_fasta, l=length, start_buffer=start_buffer, end_buffer=end_buffer, mode=mode, rep=1, num_motifs=num_motifs, seed=seed, rng=rng)
    # rc_fasta_file = f'{output_dir}/rc_{motif.name}_{motif.matrix_id}_motif_rep{rep_id}_seq.fa'
    # SeqIO.write(rdnaSeqs, rc_fasta_file, format='fasta')

    # create a dinucleotide shuffled version
    # dnaSeqs_shuf = dinucleotide_shuffle_seq(dnaSeqs)
    # SeqIO.write(dnaSeqs_shuf, f'{output_dir}/{motif.name}_{motif.matrix_id}_motif_rep{rep_id}_seq_dinuc_shuffle.fa', format='fasta')


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("genome_fasta", help="Genome fasta file")
    parser.add_argument("length", type=int, help="Length of the sequences")
    parser.add_argument("num_seqs", type=int, help="Number of seqs per concept")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--motifs", nargs="+", default=[], help="Motif ids")
    parser.add_argument(
        "--jaspar-all",
        default=False,
        action="store_true",
        help="create motif concepts on all JASPAR non-redundant motifs",
    )
    parser.add_argument(
        "--motif-list",
        type=str,
        default=None,
        help="A single column text file containing motif ids",
    )
    parser.add_argument(
        "--custom-motifs",
        type=str,
        default=None,
        help="Provide your own motifs, a two column text file [Motif_name Consensus_seq]",
    )
    parser.add_argument(
        "--meme-motifs",
        type=str,
        default=None,
        help="Motif file in MEME minimal format",
    )
    parser.add_argument(
        "--num-motifs",
        type=int,
        default=1,
        help="How many motifs to be inserted per sequence",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--spacing", type=int, default=0, help="Spacing bewteen paired motifs if any"
    )
    parser.add_argument(
        "--start-buffer",
        type=int,
        default=50,
        help="Buffer regions at the start of regions that wouldn't be inserted by motifs",
    )
    parser.add_argument(
        "--end-buffer",
        type=int,
        default=50,
        help="Buffer regions at the end of regions that wouldn't be inserted by motifs",
    )
    parser.add_argument(
        "--motif-mode",
        default="consensus",
        help="How to generate motif seq? consensus or pwm",
    )
    parser.add_argument("--reps", default=2, type=int, help="Number of reps")
    args = parser.parse_args()

    RNG = np.random.default_rng(seed=args.seed)

    if os.path.exists(args.output_dir):
        print("Output directory already exists! Content could be overwritten")
    else:
        os.makedirs(args.output_dir)

    db = jaspardb(release="JASPAR2024")

    ms = []
    # all jaspar motifs
    if args.jaspar_all:
        ms.extend(
            db.fetch_motifs(
                collection="CORE", tax_group=["vertebrates"], species=9606  # human
            )
        )
    # motif list
    if args.motif_list is not None:
        with open(args.motif_list) as l:
            for m in l:
                m = m.strip()
                ms.append(db.fetch_motif_by_id(m))
    # specified motifs
    for m in args.motifs:
        if "+" not in m:
            ms.append(db.fetch_motif_by_id(m))
        else:
            m1, m2 = m.split("+")
            ms.append(
                PairedMotif(
                    db.fetch_motif_by_id(m1),
                    db.fetch_motif_by_id(m2),
                    spacing=args.spacing,
                )
            )

    # meme motif
    if args.meme_motifs is not None:
        with open(args.meme_motifs) as handle:
            ms.extend(motifs.parse(handle, fmt="MINIMAL"))

    # customized motif
    if args.custom_motifs is not None:
        with open(args.custom_motifs) as l:
            for m in l:
                motif_name, consensus = m.strip().split("\t")
                ms.append(CustomMotif(motif_name, consensus))

    fasta_dict = defaultdict(dict)
    for i in range(args.reps):
        random_regs, randomSeqs = generate_random_seq_concept(
            l=args.length,
            n=args.num_seqs,
            genome_fasta=args.genome_fasta,
            seed=args.seed + i,
            rng=RNG,
        )
        random_fasta = f"{args.output_dir}/Random_genome_seq_rep{i}.fa"
        SeqIO.write(randomSeqs, random_fasta, format="fasta")
        with Pool() as pool:
            motif_fastas = pool.starmap(
                write_motif_concept_given_random_regs,
                [
                    (
                        motif,
                        random_regs,
                        i,
                        args.num_motifs,
                        args.genome_fasta,
                        args.length,
                        args.motif_mode,
                        args.output_dir,
                        args.seed,
                        RNG,
                        args.start_buffer,
                        args.end_buffer,
                    )
                    for motif in ms
                ],
            )
        random_bed = f"{args.output_dir}/Random_genome_rep{i}.bed"
        random_regs.to_csv(random_bed, header=False, index=False, sep="\t")

        for motif, motif_fasta in zip(ms, motif_fastas):
            fasta_dict[f"{motif.name}_rep{i}"]["seq"] = motif_fasta
            fasta_dict[f"{motif.name}_rep{i}"]["chrom"] = random_bed

    with open(f"{args.output_dir}/motif_concept_pairs.yaml", "w") as f:
        safe_dump(dict(fasta_dict), f)


if __name__ == "__main__":
    main()
