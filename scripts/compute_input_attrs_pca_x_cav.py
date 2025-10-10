#!/usr/bin/env python3

DESCRIPTION = """
    Provide test regions and cavs, compute attribution scores on the input that in parallel to the cav directions and those orthogonal to cav directions.
"""


import logging
import os
import warnings
from argparse import ArgumentParser
from glob import glob

import numpy as np
import pandas as pd
import seqchromloader as scl
import torch
import utils
from captum.attr import DeepLift
from pybedtools import BedTool
from sklearn.metrics import precision_recall_fscore_support as score
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

SEED = 1001
random_state = np.random.RandomState(SEED)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")
logger.info("Compute Layer Attrs~")


def abs_attribution_func(multipliers, inputs, baselines):
    "Multiplier x abs(inputs - baselines), this is to avoid duplicate sign coming from both inputs-baselines and concept cavs"
    attributions = tuple(
        (input - baseline).abs() * multiplier
        for input, baseline, multiplier in zip(inputs, baselines, multipliers)
    )
    return attributions


def main():
    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument("tpcav_model", help="TPCAV model")
    parser.add_argument(
        "test_bed",
        type=str,
        default=None,
        help="Test bed files will compute layer attributions on",
    )
    parser.add_argument(
        "input_window_length", default=1024, type=int, help="input window length"
    )
    parser.add_argument("genome_fasta_file", help="Genome fasta file")
    parser.add_argument("genome_size_file", help="Genome size file")
    parser.add_argument(
        "output_prefix",
        help="Prefix of output file name for attributions saved in npz format",
    )
    parser.add_argument(
        "--cavs-dir",
        type=str,
        default=None,
        help="Directory containing CAV subdirs, this would be used to disentangle the attributions into those explained by CAVs and those not",
    )
    parser.add_argument(
        "--cavs",
        type=str,
        nargs="+",
        default=None,
        help="CAVs to use, each CAV should be a folder containing classifier_weights.pt and classifier_perform_on_test.txt",
    )
    parser.add_argument(
        "--num-baselines-per-sample",
        type=int,
        default=10,
        help="How many regions sampled as background when attributing each region",
    )
    args = parser.parse_args()

    # ============================== load cavs =========================================
    # iterate through all cavs, check performance of classifier
    cavs_list = []
    total_cavs = 0
    if args.cavs_dir is not None:
        cav_weight_fn_list = glob(
            os.path.join(args.cavs_dir, "**/*classifier_weights.pt")
        )
        logger.info(f"Found {len(cav_weight_fn_list)} CAVs in {args.cavs_dir}")
        for fn in cav_weight_fn_list:
            perform = pd.read_table(
                fn.replace("classifier_weights.pt", "classifier_perform_on_test.txt"),
                comment="#",
            )  # two headers [Pred, Truth]
            _, _, fscores, _ = score(perform.Truth, perform.Pred)
            cavs_list.append(torch.load(fn, map_location="cpu")[0])
            logger.info(f"Loading CAV from {fn}, fscore: {np.mean(fscores):.3f}")
            total_cavs += 1
    if args.cavs is not None:
        for cav_dir in args.cavs:
            cav_weight_fn = glob(os.path.join(cav_dir, "*classifier_weights.pt"))[0]
            cav_perform_fn = cav_weight_fn.replace(
                "classifier_weights.pt", "classifier_perform_on_test.txt"
            )
            perform = pd.read_table(cav_perform_fn, comment="#")
            _, _, fscores, _ = score(perform.Truth, perform.Pred)
            cavs_list.append(torch.load(cav_weight_fn, map_location="cpu")[0])
            logger.info(
                f"Loading CAV from {cav_weight_fn}, fscore: {np.mean(fscores):.3f}"
            )
            total_cavs += 1

    logger.info(f"{len(cavs_list)} CAVs loaded")
    if len(cavs_list) == 0:
        cavs_list = None

    # ============================== load data =========================================
    ## target test bed samples
    target_df = BedTool(args.test_bed).to_dataframe()
    target_df = utils.center_windows(target_df, window_len=args.input_window_length)
    target_df["label"] = -1
    target_df["strand"] = "+"
    target_dl = scl.SeqChromDatasetByDataFrame(
        target_df,
        genome_fasta=args.genome_fasta_file,
        bigwig_filelist=[],
        return_region=True,
        dataloader_kws={"batch_size": 8, "drop_last": False},
    )
    ## baseline samples
    random_regs = scl.random_coords(
        gs=args.genome_size_file,
        l=args.input_window_length,
        n=len(target_df) * 20,
    )
    random_regs["label"] = -1
    random_regs["strand"] = "+"
    baseline_dl = scl.SeqChromDatasetByDataFrame(
        random_regs,
        genome_fasta=args.genome_fasta_file,
        bigwig_filelist=[],
        dataloader_kws={
            "batch_size": 8 * args.num_baselines_per_sample,
            "drop_last": True,
        },
    )

    # ============================== Attribution in PCA space ================================
    # load the TPCAV model
    tpcav_model = torch.load(args.tpcav_model)
    tpcav_model.eval()
    tpcav_model.to(device)

    tpcav_model.forward = tpcav_model.forward_from_start  # set forward function
    deeplift = DeepLift(tpcav_model, multiply_by_inputs=True)

    # attribution each test sample
    # NOTE: there should be only one attribution tensor coming out of layer attribution
    attributions_seq = []
    attributions_chrom = []
    attributions_x_avs_seq = []
    attributions_remainder_seq = []
    attributions_x_avs_chrom = []
    attributions_remainder_chrom = []
    regions_save = []
    for (region, seq, chrom, _, _), (bseq, bchrom, _, _) in tqdm(
        zip(target_dl, baseline_dl)
    ):
        assert bseq.shape[0] == args.num_baselines_per_sample * 8

        regions_save.extend(region)

        # match repeated input shape
        seq = torch.repeat_interleave(seq, repeats=args.num_baselines_per_sample, dim=0)
        chrom = torch.repeat_interleave(chrom, repeats=args.num_baselines_per_sample, dim=0)
        bseq = bseq[: seq.shape[0]]
        bchrom = bchrom[: chrom.shape[0]]

        seq = utils.seq_transform_fn(seq.to(device))
        bseq = utils.seq_transform_fn(bseq.to(device))

        chrom = utils.chrom_transform_fn(chrom.to(device))
        bchrom = utils.chrom_transform_fn(bchrom.to(device))
        
        inputs = seq if chrom is None else (seq, chrom)
        binputs = bseq if chrom is None else (bseq, bchrom)
        # attribution on full sequence
        attribution = deeplift.attribute(
            inputs,
            baselines=binputs,
            additional_forward_args=(
                cavs_list,
                False,
                False,
            ),
            # custom_attribution_func=(
            #    None if args.no_multiply_by_inputs else abs_attribution_func
            # ),
        )  # [# batch, dim_projected+dim_residual]

        def reduce_attrs(attrs):
            return attrs.reshape(-1, args.num_baselines_per_sample, *attrs.shape[1:]).mean(axis=1).detach().cpu()

        if chrom is None:
            attributions_seq.append(reduce_attrs(attribution))
        else:
            attributions_seq.append(reduce_attrs(attribution[0]))
            attributions_chrom.append(reduce_attrs(attribution[1]))

        if cavs_list is not None:
            # attribution on x avs directions
            attribution_x_avs = deeplift.attribute(
                inputs,
                baselines=binputs,
                additional_forward_args=(
                    cavs_list,
                    False,
                    True,
                ),
            )  # [# batch, dim_projected+dim_residual]
            # attribution on remainder
            attribution_remainder = deeplift.attribute(
                inputs,
                baselines=binputs,
                additional_forward_args=(
                    cavs_list,
                    True,
                    False,
                ),
            )  # [# batch, dim_projected+dim_residual]

            if chrom is None:
                attributions_x_avs_seq.append(reduce_attrs(attribution_x_avs))
                attributions_remainder_seq.append(reduce_attrs(attribution_remainder))
            else:
                attributions_x_avs_seq.append(reduce_attrs(attribution_x_avs[0]))
                attributions_x_avs_chrom.append(reduce_attrs(attribution_x_avs[1]))
                attributions_remainder_seq.append(reduce_attrs(attribution_remainder[0]))
                attributions_remainder_chrom.append(reduce_attrs(attribution_remainder[1]))

        with torch.no_grad():
            del (
                seq,
                bseq,
                attribution,
            )
            torch.cuda.empty_cache()

    # save regions
    np.savetxt(f"{args.output_prefix}.regions.txt", regions_save, fmt="%s")

    # save attributions
    def save_attrs(attrs, name):
        attrs = torch.cat(attrs)
        assert len(attrs.shape) == 3 and attrs.shape[2] == 4
        torch.save(attrs, f"{args.output_prefix}.{name}.pt")
        return attrs

    # sum over the last dimension to get per base pair attributions
    attrs_all_seq = save_attrs(attributions_seq, "attributions_seq").sum(dim=2)
    if chrom is not None:
        attrs_all_chrom = save_attrs(attributions_chrom, "attributions_chrom").sum(dim=2)

    if cavs_list is not None:
        attrs_x_avs_seq = save_attrs(attributions_x_avs_seq, "attributions_x_avs_seq").sum(dim=2)
        attrs_remainder_seq = save_attrs(
            attributions_remainder_seq, "attributions_remainder_seq"
        ).sum(dim=2)
        if chrom is not None:
            attrs_x_avs_chrom = save_attrs(attributions_x_avs_chrom, "attributions_x_avs_chrom").sum(dim=2)
            attrs_remainder_chrom = save_attrs(
                attributions_remainder_chrom, "attributions_remainder_chrom"
            ).sum(dim=2)

        # save attr x avs and total attrs per region
        if chrom is not None:
            pd.DataFrame(
                {
                    "region": regions_save,
                    "attrs_total_seq": attrs_all_seq.sum(dim=1).numpy(),
                    "attrs_x_avs_seq": attrs_x_avs_seq.sum(dim=1).numpy(),
                    "attrs_total_chrom": attrs_all_chrom.sum(dim=1).numpy(),
                    "attrs_x_avs_chrom": attrs_x_avs_chrom.sum(dim=1).numpy(),
                }
                ).assign(attrs_x_avs_total=lambda x: x['attrs_x_avs_seq'] + x['attrs_x_avs_chrom']).sort_values("attrs_x_avs_total", ascending=False).to_csv(
                f"{args.output_prefix}.regions_with_attrs_x_avs.txt",
                index=False,
                header=True,
                sep="\t",
            )
        else:
            pd.DataFrame(
                {
                    "region": regions_save,
                    "attrs_total_seq": attrs_all_seq.sum(dim=1).numpy(),
                    "attrs_x_avs_seq": attrs_x_avs_seq.sum(dim=1).numpy(),
                }
                ).sort_values("attrs_x_avs_seq", ascending=False).to_csv(
                f"{args.output_prefix}.regions_with_attrs_x_avs.txt",
                index=False,
                header=True,
                sep="\t",
            )


if __name__ == "__main__":
    main()
