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
        "output_key",
        help="Which output should be attributed to, available keys [Oct4_profile, Oct4_counts, Sox2_profile, Sox2_counts, Nanog_profile, Nanog_counts, Klf4_profile, Klf4_counts]",
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
    attributions = []
    attributions_x_avs = []
    attributions_remainder = []
    regions_save = []
    for (region, seq, chrom, _, _), (bseq, bchrom, _, _) in tqdm(
        zip(target_dl, baseline_dl)
    ):
        assert bseq.shape[0] == args.num_baselines_per_sample * 8

        regions_save.extend(region)

        # match repeated input shape
        seq = torch.repeat_interleave(seq, repeats=args.num_baselines_per_sample, dim=0)
        bseq = bseq[: seq.shape[0]]
        inputs = utils.seq_transform_fn(seq.to(device))
        binputs = utils.seq_transform_fn(bseq.to(device))

        neutral_biases = {k: v for k, v in inputs.items() if k != "seq"}

        # attribution on full sequence
        attribution = deeplift.attribute(
            inputs["seq"],
            baselines=binputs["seq"],
            additional_forward_args=(
                neutral_biases,
                args.output_key,
                True,
                cavs_list,
                False,
                False,
            ),
            # custom_attribution_func=(
            #    None if args.no_multiply_by_inputs else abs_attribution_func
            # ),
        )  # [# batch, dim_projected+dim_residual]
        attributions.append(
            attribution.reshape(
                -1, args.num_baselines_per_sample, *attribution.shape[1:]
            )
            .mean(axis=1)
            .detach()
            .cpu()
        )

        if cavs_list is not None:
            # attribution on x avs directions
            attribution_x_avs = deeplift.attribute(
                inputs["seq"],
                baselines=binputs["seq"],
                additional_forward_args=(
                    neutral_biases,
                    args.output_key,
                    True,
                    cavs_list,
                    False,
                    True,
                ),
                # custom_attribution_func=(
                #    None if args.no_multiply_by_inputs else abs_attribution_func
                # ),
            )  # [# batch, dim_projected+dim_residual]
            # attribution on remainder
            attribution_remainder = deeplift.attribute(
                inputs["seq"],
                baselines=binputs["seq"],
                additional_forward_args=(
                    neutral_biases,
                    args.output_key,
                    True,
                    cavs_list,
                    True,
                    False,
                ),
                # custom_attribution_func=(
                #    None if args.no_multiply_by_inputs else abs_attribution_func
                # ),
            )  # [# batch, dim_projected+dim_residual]
            attributions_x_avs.append(
                attribution_x_avs.reshape(
                    -1, args.num_baselines_per_sample, *attribution_x_avs.shape[1:]
                )
                .mean(axis=1)
                .detach()
                .cpu()
            )
            attributions_remainder.append(
                attribution_remainder.reshape(
                    -1, args.num_baselines_per_sample, *attribution_remainder.shape[1:]
                )
                .mean(axis=1)
                .detach()
                .cpu()
            )

        # make predictions
        # target_preds[output_key].append(tpcav_model(inpt_projected.to(device), avs_residual.to(device), args.output_key).detach().cpu())
        # baseline_preds[output_key].append(tpcav_model(bavs_projected.to(device), bavs_residual.to(device), args.output_key).detach().cpu())

        with torch.no_grad():
            del (
                seq,
                bseq,
                attribution,
            )
            torch.cuda.empty_cache()

    # save attributions
    def save_attrs(attrs, name):
        attrs = torch.cat(attrs)
        assert len(attrs.shape) == 3 and attrs.shape[2] == 4
        torch.save(attrs, f"{args.output_prefix}.{name}.pt")
        return attrs

    # sum over the last dimension to get per base pair attributions
    attrs_all = save_attrs(attributions, "attributions").sum(dim=2)
    # save regions
    np.savetxt(f"{args.output_prefix}.regions.txt", regions_save, fmt="%s")

    if cavs_list is not None:
        attrs_x_avs = save_attrs(attributions_x_avs, "attributions_x_avs").sum(dim=2)
        attrs_remainder = save_attrs(
            attributions_remainder, "attributions_remainder"
        ).sum(dim=2)

        # print summary statistics
        def compute_attr_contrib(sign="+"):
            idx = attrs_all < 0 if sign == "-" else attrs_all > 0
            attrs_all_signed = attrs_all[idx]
            attrs_x_avs_signed = attrs_x_avs[idx]
            attrs_x_avs_signed[
                (attrs_x_avs_signed > 0) if sign == "-" else (attrs_x_avs_signed < 0)
            ] = 0  # set impatible signed attrs as 0
            attrs_x_avs_contrib = (
                attrs_x_avs_signed / attrs_all_signed
            )  # get element-wise contribution ratio
            attrs_x_avs_contrib[attrs_x_avs_contrib > 1] = (
                1  # ceiling the max ratio as 1
            )
            print(
                f"{sign} contribution ratio of x avs attributions to all attributions: {attrs_x_avs_contrib.mean():.3f}"
            )
            return attrs_x_avs_contrib, idx

        pos_contrib_ratio, pos_contrib_index = compute_attr_contrib(sign="+")
        neg_contrib_ratio, neg_contrib_indx = compute_attr_contrib(sign="-")

        with open(f"{args.output_prefix}.contrib_ratio.txt", "w") as f:
            f.write(
                f"Positive contribution ratio of x avs attributions to all attributions: {pos_contrib_ratio.mean().item():.3f}\n"
            )
            f.write(
                f"Negative contribution ratio of x avs attributions to all attributions: {neg_contrib_ratio.mean().item():.3f}\n"
            )
            f.write(
                f"Total contribution ratio of x avs attributions to all attributions: {torch.cat([pos_contrib_ratio, neg_contrib_ratio]).mean().item():.3f}\n"
            )
        # save regions with the attrib ratios
        contrib_ratio = torch.zeros_like(attrs_all)
        assert len(contrib_ratio.shape) == 2
        contrib_ratio[pos_contrib_index] = pos_contrib_ratio
        contrib_ratio[neg_contrib_indx] = neg_contrib_ratio
        contrib_ratio_per_region = contrib_ratio.mean(dim=1)

        with open(f"{args.output_prefix}.regions_with_contrib.txt", "w") as o:
            for r, cr in zip(regions_save, contrib_ratio_per_region):
                o.write(f"{r}\t{cr.item()}\n")

        # save attr x avs and total attrs per region
        pd.DataFrame(
            {
                "region": regions_save,
                "attrs_total": attrs_all.sum(dim=1).numpy(),
                "attrs_x_avs": attrs_x_avs.sum(dim=1).numpy(),
            }
        ).sort_values("attrs_x_avs", ascending=False).to_csv(
            f"{args.output_prefix}.regions_with_attrs_x_avs.txt",
            index=False,
            header=True,
            sep="\t",
        )


if __name__ == "__main__":
    main()
