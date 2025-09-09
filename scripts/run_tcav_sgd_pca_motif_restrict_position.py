#!/usr/bin/env python3

DESCRIPTION = """
    Given concepts, train linear classifier and evaluate
"""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import logging
import os
import time
from argparse import ArgumentParser
from multiprocessing import Pool
from os import makedirs, path

import numpy as np
import seqchromloader as scl
import torch
import utils
from Bio import motifs
from captum.concept import Concept
from run_tcav_sgd_pca import (
    construct_motif_concept_dataloader_from_control,
    train_classifier,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")
logger.info("TCAV~")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

SEED = 1001
random_state = np.random.RandomState(SEED)


# ================================================ Concept construction =========================================#
def get_buffer_list(window_len, num_bins):
    pos_list = np.linspace(0, window_len, num_bins + 1, dtype=int)
    buffer_list = []
    for i in range(len(pos_list) - 1):
        start = pos_list[i]
        end = pos_list[i + 1]
        buffer_list.append((start, window_len - end))
    return buffer_list


def main():

    def pair(arg):
        # assume arg is a pair of strings separated by comma
        return [x for x in arg.split(",")]

    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument("output_dir", help="Output folder of the run")
    parser.add_argument("tpcav_model", help="Output folder of the run")
    parser.add_argument(
        "input_window_length", default=1024, type=int, help="input window length"
    )
    parser.add_argument("genome_fasta_file")
    parser.add_argument("genome_size_file")
    parser.add_argument(
        "--custom-motifs",
        type=str,
        default=None,
        help="Provide your own motifs via a two column text file in format: Motif_name\tConsensus_seq",
    )
    parser.add_argument(
        "--meme-motifs",
        type=str,
        default=None,
        help="Motif file in MEME minimal format",
    )
    parser.add_argument(
        "--num-motifs",
        default=3,
        type=int,
        help="Number of motifs to insert per bin",
    )
    parser.add_argument(
        "--num-bins",
        default=3,
        type=int,
        help="Number of bins to split the region for inserting motifs",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum number of samples to draw for training classifier on each concept",
    )
    parser.add_argument(
        "--bws",
        nargs="+",
        default=None,
        help="List of bigwig files to extract chromatin signal from, use this option if your model takes chromatin data as input",
    )
    parser.add_argument(
        "--classifier", default="sgd", help="choose from sgd or cuml_sgd"
    )
    parser.add_argument("--SGD-penalty", default="l2", help="SGD penalty type")
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="Save activations per layer per concept",
    )
    args = parser.parse_args()

    if path.exists(args.output_dir):
        logger.warning(
            f"Output directory {args.output_dir} exists! Contents could be overwritten"
        )
    makedirs(args.output_dir, exist_ok=True)  # make output directory
    logger.info(f"Created output directory {args.output_dir}")

    BATCH_SIZE = 8

    # load concepts, be cautious about dataset specific transforms
    idx = 0
    ## load control concepts first
    random_regions_fn = f"{args.output_dir}/random_regions.bed"
    random_regions_df = scl.random_coords(
        gs=args.genome_size_file, l=args.input_window_length, n=args.max_samples
    )
    random_regions_df.to_csv(random_regions_fn, sep="\t", header=False, index=False)
    control_concepts = []

    control_seq_dl = utils.seq_dataloader_from_bed(
        random_regions_fn,
        args.genome_fasta_file,
        args.input_window_length,
        BATCH_SIZE,
    )
    control_chrom_dl = utils.chrom_dataloader_from_bed(
        random_regions_fn,
        args.genome_fasta_file,
        args.input_window_length,
        args.bws,
        BATCH_SIZE,
    )

    control_concept = Concept(
        id=-idx,
        name="random_regions",
        data_iter=zip(control_seq_dl, control_chrom_dl),
    )
    control_concepts.append(control_concept)
    idx += 1

    ## load test concepts
    concepts = []

    ## custom motifs, use the first control concept as a template
    ### according to # bins, get the list of buffer
    buffer_list = get_buffer_list(args.input_window_length, args.num_bins)

    if args.custom_motifs is not None:
        with open(args.custom_motifs) as f:

            for m in f:
                for buffer_idx, (start_buffer, end_buffer) in enumerate(buffer_list):
                    motif_name, consensus_seq = m.strip().split("\t")
                    motif = utils.CustomMotif("motif", consensus_seq)
                    cn = f"{motif_name}_bin_{buffer_idx}"
                    seq_dl = construct_motif_concept_dataloader_from_control(
                        random_regions_df,
                        args.genome_fasta_file,
                        motif=motif,
                        num_motifs=args.num_motifs,
                        start_buffer=start_buffer,
                        end_buffer=end_buffer,
                        batch_size=BATCH_SIZE,
                    )
                    concepts.append(
                        Concept(
                            id=idx,
                            name=cn,
                            data_iter=zip(seq_dl, control_chrom_dl),
                        )
                    )
                    idx += 1
    if args.meme_motifs is not None:
        with open(args.meme_motifs) as f:
            for motif in motifs.parse(f, fmt="MINIMAL"):
                for buffer_idx, (start_buffer, end_buffer) in enumerate(buffer_list):
                    cn = f"{motif.name.replace('/', '-')}_bin_{buffer_idx}"
                    seq_dl = construct_motif_concept_dataloader_from_control(
                        random_regions_df,
                        args.genome_fasta_file,
                        motif=motif,
                        num_motifs=args.num_motifs,
                        start_buffer=start_buffer,
                        end_buffer=end_buffer,
                        batch_size=BATCH_SIZE,
                    )
                    concepts.append(
                        Concept(
                            id=idx,
                            name=cn,
                            data_iter=zip(seq_dl, control_chrom_dl),
                        )
                    )
                    idx += 1

    logger.info("Constructed concepts")
    logger.info(concepts)

    # load tpcav model
    model = torch.load(args.tpcav_model, map_location=device)
    model.eval()

    def get_tpcav_activations(concept):
        avs_pca = []
        for seq, chrom in concept.data_iter:
            seq = utils.seq_transform_fn(seq)
            chrom = utils.chrom_transform_fn(chrom)
            if chrom is not None:
                av = model.forward_until_select_layer(seq.to(device), chrom.to(device))
            else:
                av = model.forward_until_select_layer(seq.to(device))
            av_residual, av_projected = model.project_avs_to_pca(
                av.flatten(start_dim=1)
            )
            if av_projected is not None:
                av_pca = torch.cat((av_projected, av_residual), dim=1)
            else:
                av_pca = av_residual
            avs_pca.append(av_pca.detach().cpu())
            with torch.no_grad():
                del seq, av, av_projected, av_residual
        return torch.cat(avs_pca).detach().cpu()

    # get activations of each concept and train classifier for each pair
    pool = Pool()
    pending = []
    max_pending_jobs = 4
    control_concept_avs = {
        cc.name: get_tpcav_activations(cc) for cc in control_concepts
    }
    logging.info(f"Collected all activations for control concepts")

    for c in concepts:
        idx = c.id
        try:
            test_avs = get_tpcav_activations(c)
        except RuntimeError as e:
            print(c)
            raise e
        for cc in control_concepts:
            cp = {0: test_avs, 1: control_concept_avs[cc.name]}
            dir_name = f"{args.output_dir}/cavs/{c.name}_control_{cc.name}"
            os.makedirs(dir_name, exist_ok=True)
            concept_pair = (cp, args, dir_name, args.SGD_penalty)

            while len(pending) >= max_pending_jobs:
                for job in pending:
                    if job.ready():
                        job.get()  # raise exception if any
                # keep all pending jobs
                pending = [job for job in pending if not job.ready()]
                if len(pending) >= max_pending_jobs:
                    time.sleep(1)
            async_result = pool.apply_async(train_classifier, args=concept_pair)
            logging.info(
                f"Started training classifier for concept {c.name} vs control {cc.name}"
            )
            pending.append(async_result)

        del c

    # wait for jobs finish
    if len(pending) > 0:
        for job in pending:
            job.wait()
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
