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
import pandas as pd
import seqchromloader as scl
import torch
import webdataset as wds
from Bio import motifs as Bio_motifs
from captum.concept import Classifier, Concept
from scipy.linalg import svd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader, TensorDataset, random_split

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
logger.info("TCAV~")

import numpy as np
import seqchromloader as scl
import utils


def load_all_tensors_to_numpy(dataloaders):
    if not isinstance(dataloaders, list):
        dataloaders = [
            dataloaders,
        ]
    avs = []
    ls = []
    for dataloader in dataloaders:
        for av, l in dataloader:
            avs.append(av.cpu().numpy())
            ls.append(l.cpu().numpy())
    avs = np.concatenate(avs)
    ls = np.concatenate(ls)

    return avs, ls


class SGD(Classifier):
    def __init__(self, n_jobs=-1, penalty="l2"):
        self.lm = SGDClassifier(
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            learning_rate="optimal",
            n_iter_no_change=10,
            n_jobs=n_jobs,
            penalty=penalty,
        )
        if penalty == "l2":
            self.parameters = {"alpha": [1e-2, 1e-4, 1e-6]}
        elif penalty == "l1":
            self.parameters = {"alpha": [1e-1, 1]}
        else:
            raise Exception(f"Unexpected penalty type {penalty} for SGD classifier")
        self.search = GridSearchCV(self.lm, self.parameters)

    def train_and_eval(self, train_dataloader, val_dataloader):
        # get all train/val tensors and convert into numpy
        train_avs, train_ls = load_all_tensors_to_numpy(
            [train_dataloader, val_dataloader]
        )
        # fit
        self.search.fit(train_avs, train_ls)
        self.lm = self.search.best_estimator_
        # print status
        logger.info(f"Best Params of hyperparam search: {self.search.best_params_}")
        logger.info(f"SGD iterations: {self.lm.n_iter_}")

        return

    def weights(self):
        if len(self.lm.coef_) == 1:
            return torch.tensor(np.array([-1 * self.lm.coef_[0], self.lm.coef_[0]]))
        else:
            return torch.tensor(self.lm.coef_)

    def classes(self):
        return self.lm.classes_


# train linear classifier
def train_classifier(avs, args, output_dir, sgd_penalty="l2"):
    train_avds = []
    val_avds = []
    test_avds = []
    for idx, av in avs.items():
        avd = TensorDataset(av, torch.full((av.shape[0],), idx))
        train_avd, val_avd, test_avd = random_split(avd, [0.8, 0.1, 0.1])
        print(f"training dataset size: {len(train_avd)}")
        print(f"validation dataset size: {len(val_avd)}")
        print(f"test dataset size: {len(test_avd)}")
        train_avds.append(train_avd)
        val_avds.append(val_avd)
        test_avds.append(test_avd)
    train_avds = wds.RandomMix(train_avds)
    val_avds = wds.RandomMix(val_avds)
    test_avds = wds.RandomMix(test_avds)
    train_avdl = DataLoader(train_avds, batch_size=32)
    val_avdl = DataLoader(val_avds, batch_size=32)
    test_avdl = DataLoader(test_avds, batch_size=32)
    logger.info(f"Collected all activations for")

    input_dim = next(iter(train_avdl))[0].shape[1]
    logger.info(f"Layer activation input dimension is {input_dim}")
    classifier = SGD(penalty=sgd_penalty)
    classifier.train_and_eval(train_avdl, val_avdl)
    logger.info(f"Trained classifier")

    # summarize performance and save predictions
    def make_predictions_and_save(avdl, name):
        y_preds = []
        y_trues = []
        for x, y in avdl:
            y_pred = classifier.lm.predict(x.cpu().numpy())
            y_preds.append(y_pred)
            y_trues.append(y.cpu().numpy())
        y_preds = np.concatenate(y_preds)
        y_trues = np.concatenate(y_trues)

        df_plot = pd.DataFrame(
            # {"Pred": (y_preds.flatten() > 0.5) * 1, "Truth": y_trues.flatten()}
            {"Pred": y_preds.flatten(), "Truth": y_trues.flatten()}
        )

        with open(
            f"{output_dir}/classifier_perform_on_{name}.txt", "w"
        ) as fout_perform:
            for i in range(len(avs)):
                df_sub = df_plot.loc[df_plot.Truth == i]
                logger.info(
                    f"[{name}] Accuracy of concept id {i}: {(df_sub.Pred == df_sub.Truth).sum()/len(df_sub)}"
                )
                fout_perform.write(
                    f"#Accuracy of concept id {i}: {(df_sub.Pred == df_sub.Truth).sum()/len(df_sub)}\n"
                )

            logger.info(
                f"[{name}] Overall accuracy: {(df_plot.Pred == df_plot.Truth).sum()/len(df_plot)}"
            )
            fout_perform.write(
                f"#Overall accuracy: {(df_plot.Pred == df_plot.Truth).sum()/len(df_plot)}\n"
            )

            df_plot.to_csv(fout_perform, header=True, index=False, sep="\t")

    make_predictions_and_save(train_avdl, "train")
    make_predictions_and_save(val_avdl, "val")
    make_predictions_and_save(test_avdl, "test")

    # save classifier weights
    torch.save(classifier.weights(), f"{output_dir}/classifier_weights.pt")


# ================================================ Concept construction =========================================#
def construct_motif_concept_dataloader_from_control(
    control_seq_bed_df,
    genome_fasta,
    motifs: list,
    num_motifs=128,
    motif_mode="pwm",
    start_buffer=0,
    end_buffer=0,
    batch_size=8,
    num_workers=0,
    infinite=False,
):
    "Construct a concept from control sequence bed file and insert motif sequence"
    dss = []
    for motif in motifs:
        ds = utils.IterateSeqDataFrame(
            control_seq_bed_df,
            genome_fasta,
            motif=motif,
            num_motifs=num_motifs,
            motif_mode=motif_mode,
            start_buffer=start_buffer,
            end_buffer=end_buffer,
            print_warning=False,
            infinite=infinite,
        )
        dss.append(ds)

    dl = torch.utils.data.DataLoader(
        wds.RandomMix(dss),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dl


def construct_bed_concept_dataloader_from_control(
    control_seq_bed_df,
    genome_fasta,
    regions_df,
    batch_size=8,
    num_workers=0,
    infinite=False,
):
    "Construct a concept from control sequence bed file and insert motif sequence"
    # take shard of the dataframe if specified
    dl = torch.utils.data.DataLoader(
        utils.IterateSeqDataFrame(
            control_seq_bed_df,
            genome_fasta,
            regions_insert=regions_df,
            print_warning=False,
            infinite=infinite,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return dl


def construct_concept(
    cn,
    c_seq,
    chrom_bed,
    idx,
    genome_fasta,
    signal_bws,
    transforms=None,
    window_len=1024,
    batch_size=8,
):
    if c_seq.endswith(".fa"):
        seq_fa = c_seq
        sc_concept = utils.SeqChromConcept(
            seq_bed=None,
            seq_fa=seq_fa,
            chrom_bed=chrom_bed,
            genome_fasta=genome_fasta,
            bws=signal_bws,
            window_len=window_len,
            transforms=transforms,
            batch_size=batch_size,
        )
    elif c_seq.endswith(".bed"):
        seq_bed = c_seq
        sc_concept = utils.SeqChromConcept(
            seq_bed=seq_bed,
            seq_fa=None,
            chrom_bed=chrom_bed,
            genome_fasta=genome_fasta,
            bws=signal_bws,
            window_len=window_len,
            transforms=transforms,
            batch_size=batch_size,
        )
    else:
        raise Exception(
            f"Sequence concept file has to end with either .fa or .bed for concept {cn}!"
        )

    return Concept(id=idx, name=cn, data_iter=sc_concept.seq_dataloader())


def main():

    def pair(arg):
        # assume arg is a pair of strings separated by comma
        return [x for x in arg.split(",")]

    parser = ArgumentParser(description=DESCRIPTION)
    parser.add_argument("output_dir", help="Output folder of the run")
    parser.add_argument(
        "input_window_length", default=1024, type=int, help="input window length"
    )
    parser.add_argument("genome_fasta_file")
    parser.add_argument("genome_size_file")
    parser.add_argument(
        "--concepts",
        default=[],
        nargs="+",
        type=pair,
        help="Prefixes of concepts in format: name1,concept1_seq.fa,concept1_chrom.bed name2,concept2_seq.fa,concept2_chrom.bed...",
    )
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
        "--bed-seq-concepts",
        type=str,
        nargs="+",
        default=None,
        help="Bed file of regions provided as sequence concepts, format: [chrom, start, end, strand, concept_name]",
    )
    parser.add_argument(
        "--bed-chrom-concepts",
        type=str,
        nargs="+",
        default=None,
        help="Bed file of regions provided as chromatin concepts, format: [chrom, start, end, strand, concept_name]",
    )
    parser.add_argument(
        "--bws",
        nargs="+",
        default=None,
        help="List of bigwig files to extract chromatin signal from, use this option if your model takes chromatin data as input",
    )
    parser.add_argument(
        "--num-motifs", type=int, default=12, help="Number of motifs to insert"
    )
    parser.add_argument(
        "--include-reverse-complement",
        action="store_true",
        help="Use both forward and reverse complement version of motifs",
    )
    parser.add_argument(
        "--num-samples-per-concept",
        type=int,
        default=10,
        help="Number of samples per concept to draw to compute PCA matrix",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Number of samples required for each concept to train classifiers, this would apply to filter out bed concepts that don't have enough samples",
    )
    parser.add_argument(
        "--num-pc", default="full", help="Number of PCs to keep, or 'full' to keep all"
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

    # load model, this part is model specific
    model = utils.load_model()
    model.eval()
    model.to(device)
    logger.info(f"Loaded model")

    # load PCA model
    # model_pca = torch.load(args.tpcav_model).to(device)
    # model_pca.eval()

    # load concepts, be cautious about dataset specific transforms
    idx = 0
    ## load control concepts first
    random_regions_fn = f"{args.output_dir}/random_regions.bed"
    random_regions_df = scl.random_coords(
        gs=args.genome_size_file, l=args.input_window_length, n=args.num_samples
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
    if args.custom_motifs is not None:
        df = pd.read_table(args.custom_motifs, names=["motif_name", "consensus_seq"])
        for m in np.unique(df.motif_name):
            motif_name = m
            consensus_seqs = df.loc[
                df.motif_name == m, "consensus_seq"
            ].tolist()  # take all consensus seqs that correspond to the same motif name
            motifs = []
            for i, c in enumerate(consensus_seqs):
                motif = utils.CustomMotif(f"{m}_{i}", c)
                motifs.append(motif)
                if (
                    args.include_reverse_complement
                ):  # add reverse complement if specified
                    motif_rc = motif.reverse_complement()
                    motifs.append(motif_rc)

            cn = f"{motif_name}"
            seq_dl = construct_motif_concept_dataloader_from_control(
                random_regions_df,
                args.genome_fasta_file,
                motifs=motifs,
                num_motifs=args.num_motifs,
                motif_mode="consensus",
                batch_size=BATCH_SIZE,
                infinite=False,
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
            for motif in Bio_motifs.parse(f, fmt="MINIMAL"):
                motifs = []
                motifs.append(motif)
                if args.include_reverse_complement:
                    motif_rc = motif.reverse_complement()
                    motifs.append(motif_rc)
                cn = f"{motif.name.replace('/', '-')}"
                seq_dl = construct_motif_concept_dataloader_from_control(
                    random_regions_df,
                    args.genome_fasta_file,
                    motifs=motifs,
                    num_motifs=args.num_motifs,
                    motif_mode="pwm",
                    batch_size=BATCH_SIZE,
                    infinite=False,
                )
                concepts.append(
                    Concept(
                        id=idx,
                        name=cn,
                        data_iter=zip(seq_dl, control_chrom_dl),
                    )
                )
                idx += 1
    if args.bed_seq_concepts is not None:
        for bed in args.bed_seq_concepts:
            bed_seq_df = pd.read_table(
                bed,
                header=None,
                usecols=[0, 1, 2, 3, 4],
                names=["chrom", "start", "end", "strand", "concept_name"],
            )
            for cn in bed_seq_df.concept_name.unique():
                cn_bed_seq_df = bed_seq_df.loc[bed_seq_df.concept_name == cn]
                if len(cn_bed_seq_df) < args.num_samples:
                    logger.warning(
                        f"Concept {cn} has only {len(cn_bed_seq_df)} samples, less than minimum required {args.num_samples}, skip"
                    )
                    continue
                seq_dl = utils.seq_dataloader_from_dataframe(
                    cn_bed_seq_df.sample(n=args.num_samples),
                    args.genome_fasta_file,
                    args.input_window_length,
                    BATCH_SIZE,
                    num_workers=0,
                )
                concepts.append(
                    Concept(id=idx, name=cn, data_iter=zip(seq_dl, control_chrom_dl))
                )
                idx += 1

    if args.bed_chrom_concepts is not None:
        for bed in args.bed_chrom_concepts:
            bed_chrom_df = pd.read_table(
                bed,
                header=None,
                usecols=[0, 1, 2, 3, 4],
                names=["chrom", "start", "end", "strand", "concept_name"],
            )
            for cn in bed_chrom_df.concept_name.unique():
                cn_bed_chrom_df = bed_chrom_df.loc[bed_chrom_df.concept_name == cn]
                if len(cn_bed_chrom_df) < args.num_samples:
                    logger.warning(
                        f"Concept {cn} has only {len(cn_bed_chrom_df)} samples, less than minimum required {args.num_samples}, skip"
                    )
                    continue
                chrom_dl = utils.chrom_dataloader_from_dataframe(
                    cn_bed_chrom_df.sample(n=args.num_samples),
                    args.genome_fasta_file,
                    args.input_window_length,
                    args.bws,
                    BATCH_SIZE,
                    num_workers=0,
                )
                concepts.append(
                    Concept(id=idx, name=cn, data_iter=zip(control_seq_dl, chrom_dl))
                )
                idx += 1

    logger.info("Constructed concepts")
    logger.info(concepts)

    # register hook
    def get_activation(concept, num_samples=args.num_samples_per_concept):
        avs = []
        num = 0
        for seq, chrom in concept.data_iter:
            # print(seq)
            # print(concept)
            seq = utils.seq_transform_fn(seq)
            chrom = utils.chrom_transform_fn(chrom)
            if chrom is not None:
                av = model.forward_until_select_layer(seq.to(device), chrom.to(device))
            else:
                av = model.forward_until_select_layer(seq.to(device))
            avs.append(av.detach().cpu())
            num += av.shape[0]
            if num >= num_samples:
                break

        return torch.cat(avs)[:num_samples]

    # load sampled concept examples and build tpcav model
    sampled_avs = []
    for c in concepts + control_concepts:
        sampled_avs.append(get_activation(c, num_samples=args.num_samples_per_concept))
        logging.info(f"Sampled {sampled_avs[-1].shape[0]} from concept {c.name}")
    sampled_avs = torch.cat(sampled_avs)
    orig_shape = sampled_avs.shape
    sampled_avs = sampled_avs.flatten(start_dim=1)

    # Zscore standardication
    avs_mean = sampled_avs.mean(dim=0)
    avs_std = sampled_avs.std(dim=0)
    avs_std[avs_std == 0] = (
        -1
    )  # to avoid division by zero when all values are the same along a dimension
    avs_standardized = (sampled_avs - avs_mean) / avs_std
    logger.warning(f"Activation matrix is of shape {avs_standardized.shape}")
    torch.save(avs_standardized, f"{args.output_dir}/avs_standardized.pt")
    # PCA
    if args.num_pc == "full":
        max_pc = min(avs_standardized.shape)
        U, S, Vh = svd(avs_standardized, lapack_driver="gesvd", full_matrices=False)
        V_inverse = torch.tensor(Vh)
    elif int(args.num_pc) == 0:
        V = None
        V_inverse = None
    else:
        U, S, Vh = svd(avs_standardized, lapack_driver="gesvd", full_matrices=False)
        V_inverse = torch.tensor(Vh)
        # estimate minimal explained variance ratio
        max_pc = min(avs_standardized.shape)
        S_imputed = torch.concat([S, S[-1].repeat(max_pc - len(S))])
        min_pca_explained_ratio = (S_imputed / S_imputed.sum())[: len(S)].sum()
        logger.info(
            f"Estimated minimal explained PCA Ratio is {min_pca_explained_ratio}"
        )

    # create tpcav model
    model.register_buffer("zscore_mean", avs_mean.to(device))
    model.register_buffer("zscore_std", avs_std.to(device))
    model.register_buffer(
        "pca_inv", V_inverse.to(device) if V_inverse is not None else None
    )
    model.register_buffer("orig_shape", torch.tensor(orig_shape).to(device))
    # save the new model
    torch.save(model, f"{args.output_dir}/tpcav_model.pt")
    # set to eval mode for sanity
    model.eval()

    def get_tpcav_activations(concept, num_samples=args.num_samples):
        avs_pca = []
        num = 0
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

            num += av_pca.shape[0]
            if num >= num_samples:
                break
            with torch.no_grad():
                del seq, av, av_projected, av_residual
        return torch.cat(avs_pca).detach().cpu()[:num_samples]

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
