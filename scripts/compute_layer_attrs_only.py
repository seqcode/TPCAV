#!/usr/bin/env python3

DESCRIPTION = """
    Given concepts, train linear classifier and evaluate
"""

import argparse
import logging
import re

import numpy as np
import pandas as pd
import seqchromloader as scl
import torch
import utils
from captum.attr import DeepLift
from pybedtools import BedTool

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

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=DESCRIPTION
    )
    parser.add_argument("tpcav_model", help="saved TPCAV model")
    parser.add_argument(
        "test_bed",
        type=str,
        default=None,
        help="Test bed files will compute layer attributions on",
    )
    parser.add_argument(
        "input_window_length", default=1024, type=int, help="input window length"
    )
    parser.add_argument("genome_fasta_file")
    parser.add_argument("genome_size_file")
    parser.add_argument(
        "output_prefix",
        help="Prefix of output file name for attributions saved in npz format",
    )
    parser.add_argument(
        "--no-multiply-by-inputs",
        action="store_true",
        default=False,
        help="Specify if you don't want to factor in the activation difference between input and baseline",
    )
    parser.add_argument(
        "--num-baselines-per-sample",
        type=int,
        default=10,
        help="How many regions sampled as background when attributing each region",
    )
    parser.add_argument(
        "--chroms",
        nargs="+",
        default=None,
        help="Restrict attribution to certain chromosomes",
    )
    args = parser.parse_args()

    ## target test bed samples
    target_df = BedTool(args.test_bed).to_dataframe()
    target_df = utils.center_windows(target_df, window_len=args.input_window_length)
    target_df["label"] = -1
    target_df["strand"] = "+"
    if args.chroms is not None:
        target_df = scl.filter_chromosomes(target_df, to_keep=args.chroms)
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

    # load the saved model with tpcav parameters defined
    tpcav_model = torch.load(args.tpcav_model, map_location=device)
    tpcav_model.forward = (
        tpcav_model.forward_from_projected_and_residual
    )  # set forward pass starting from projected and residual
    tpcav_model.eval()
    deeplift = DeepLift(
        tpcav_model, multiply_by_inputs=False if args.no_multiply_by_inputs else True
    )

    # Atrtribute on the PCA embedding space
    attributions = []
    avs_projected_save = []
    avs_residual_save = []
    regions = []
    target_preds = []
    baseline_preds = []
    # for inpt, binpt in zip(target_dl, baseline_dl):
    for (region, seq, chrom, _, _), (bseq, bchrom, _, _) in zip(target_dl, baseline_dl):
        assert bseq.shape[0] == 8 * args.num_baselines_per_sample
        # save regions
        for r in region:
            regions.append(re.split(r"[:-]", r))

        # project input
        seq = utils.seq_transform_fn(seq)
        chrom = utils.chrom_transform_fn(chrom)
        if chrom is not None:
            avs = tpcav_model.forward_until_select_layer(
                seq.to(device), chrom.to(device)
            )
        else:
            avs = tpcav_model.forward_until_select_layer(seq.to(device))
        avs_residual, avs_projected = tpcav_model.project_avs_to_pca(
            avs.flatten(start_dim=1).to(device)
        )
        ## save
        if avs_projected is not None:
            avs_projected_save.append(avs_projected.detach().cpu())
        avs_residual_save.append(avs_residual.detach().cpu())
        # match repeated input shape
        avs_projected = (
            torch.repeat_interleave(
                avs_projected, repeats=args.num_baselines_per_sample, dim=0
            )
            if avs_projected is not None
            else None
        )
        avs_residual = torch.repeat_interleave(
            avs_residual, repeats=args.num_baselines_per_sample, dim=0
        )
        bseq = bseq[: (avs.shape[0] * args.num_baselines_per_sample)]
        bchrom = bchrom[: (avs.shape[0] * args.num_baselines_per_sample)]

        # do normal model forward on baselines to get activations
        bseq = utils.seq_transform_fn(bseq)
        bchrom = utils.chrom_transform_fn(bchrom)
        if bchrom is not None:
            bavs = tpcav_model.forward_until_select_layer(
                bseq.to(device), bchrom.to(device)
            )
        else:
            bavs = tpcav_model.forward_until_select_layer(bseq.to(device))
        bavs_residual, bavs_projected = tpcav_model.project_avs_to_pca(
            bavs.flatten(start_dim=1).to(device)
        )

        # attribution
        if avs_projected is not None:
            attribution = deeplift.attribute(
                (avs_residual.to(device), avs_projected.to(device)),
                baselines=(bavs_residual.to(device), bavs_projected.to(device)),
                custom_attribution_func=(
                    None if args.no_multiply_by_inputs else abs_attribution_func
                ),
            )
            attr_residual, attr_projected = attribution
            attribution = torch.cat((attr_projected, attr_residual), dim=1)
            with torch.no_grad():
                del attr_residual, attr_projected
        else:
            attribution = deeplift.attribute(
                (avs_residual.to(device),),
                baselines=(bavs_residual.to(device),),
                additional_forward_args=(None,),
                custom_attribution_func=(
                    None if args.no_multiply_by_inputs else abs_attribution_func
                ),
            )
            assert len(attribution) == 1
            attribution = attribution[0]
        attributions.append(attribution.detach().cpu())

        # make predictions, save both projected activations and residuals
        if avs_projected is not None:
            target_preds.append(
                tpcav_model(avs_residual.to(device), avs_projected.to(device))
                .detach()
                .cpu()
            )
            baseline_preds.append(
                tpcav_model(bavs_residual.to(device), bavs_projected.to(device))
                .detach()
                .cpu()
            )
        else:
            target_preds.append(tpcav_model(avs_residual.to(device), None).detach().cpu())
            baseline_preds.append(tpcav_model(bavs_residual.to(device), None).detach().cpu())

        with torch.no_grad():
            del avs, avs_projected, avs_residual, bavs, bavs_projected, bavs_residual, seq, chrom, bseq, bchrom, attribution
            torch.cuda.empty_cache()

    print(f"Average target predictions: {torch.concat(target_preds).mean()}")
    print(f"Average baseline predictions: {torch.concat(baseline_preds).mean()}")

    # save regions
    regions = pd.DataFrame(regions, columns=["chrom", "start", "end"])
    regions.to_csv(
        f"{args.output_prefix}.regions.bed",
        header=False,
        index=False,
        sep="\t",
    )

    # save activations
    if len(avs_projected_save) > 0:
        avs_projected_save = torch.cat(avs_projected_save)
        torch.save(
            avs_projected_save,
            f"{args.output_prefix}.activations_projected.pt",
        )
    else:
        torch.save(None, f"{args.output_prefix}.activations_projected.pt")

    avs_residual_save = torch.cat(avs_residual_save)
    torch.save(
        avs_residual_save,
        f"{args.output_prefix}.activations_residual.pt",
    )

    # save attributions
    attrs = torch.cat(attributions)
    attrs = attrs.reshape(-1, args.num_baselines_per_sample, *attrs.shape[1:]).mean(
        axis=1
    )
    torch.save(attrs, f"{args.output_prefix}.attributions.pt")

    with torch.no_grad():
        del attrs


if __name__ == "__main__":
    main()
