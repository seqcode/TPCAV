import unittest
from functools import partial
from pathlib import Path

import torch
from Bio import motifs as Bio_motifs
from captum.attr import DeepLift

from tpcav import helper, run_tpcav, utils
from tpcav.cavs import CavTrainer
from tpcav.concepts import ConceptBuilder
from tpcav.tpcav_model import TPCAV, _abs_attribution_func

from line_profiler import LineProfiler

class DummyModelSeq(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(1024, 1)
        self.layer2 = torch.nn.Linear(4, 1)

    def forward(self, seq):
        y_hat = self.layer1(seq)
        y_hat = y_hat.squeeze(-1)
        y_hat = self.layer2(y_hat)
        return y_hat

    def foward_from_layer1(self, y_hat):
        y_hat = y_hat.squeeze(-1)
        y_hat = self.layer2(y_hat)
        return y_hat


class DummyModelSeqChrom(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(1024, 1)
        self.layer2 = torch.nn.Linear(4, 1)

    def forward(self, seq, chrom):
        y_hat = self.layer1(seq)
        y_hat = y_hat.squeeze(-1)
        y_hat = self.layer2(y_hat)
        return y_hat


def transform_fasta_to_one_hot_seq(seq, chrom):
    return (helper.fasta_to_one_hot_sequences(seq),)


class TPCAVTest(unittest.TestCase):

    def test_motif_concepts_insertion(self):
        motif_path = Path("data") / "motif-clustering-v2.1beta_consensus_pwms.test.meme"
        self.assertTrue(motif_path.exists(), "Motif file is missing")

        builder = ConceptBuilder(
            genome_fasta="data/hg38.analysisSet.fa",
            input_window_length=1024,
            bws=None,
            num_motifs=16,
            include_reverse_complement=True,
            min_samples=1000,
            batch_size=8,
        )

        builder.build_control()

        builder.add_meme_motif_concepts(str(motif_path))

        # load motifs
        motifs = Bio_motifs.parse(open(motif_path), fmt="minimal")

        for motif in motifs:
            motif_name = motif.name.replace("/", "-")

            concept = None
            for c in builder.concepts:
                if c.name == motif_name:
                    concept = c
                    break

            self.assertIsNotNone(concept)

            seq, chrom = next(iter(concept.data_iter))

            matches = list(motif.pssm.search(seq[0], threshold=2.0))

            self.assertGreaterEqual(
                len(matches),
                16,
                f"Motif concept {motif_name} has insufficient matches {matches}",
            )

            control_seq, _ = next(iter(builder.control_concepts[0].data_iter))

            control_matches = list(motif.pssm.search(control_seq[0], threshold=2.0))

            self.assertGreater(
                len(matches),
                len(control_matches),
                f"Control concept has more motif matches than Motif concept, motif concept: {len(matches)}, control concept: {len(control_matches)}",
            )

    def test_run_tpcav(self):
        motif_path = Path("data") / "motif-clustering-v2.1beta_consensus_pwms.test.meme"
        genome_fasta = "data/hg38.analysisSet.fa"
        model = DummyModelSeq()
        layer_name = "layer1"

        cavs_fscores_df, motif_cav_trainers, bed_cav_trainer = run_tpcav(
            model=model,
            layer_name=layer_name,
            meme_motif_file=str(motif_path),
            genome_fasta=genome_fasta,
            num_motif_insertions=[4, 8],
            bed_seq_file="data/hg38_rmsk.head50k.bed",
            output_dir="data/test_run_tpcav_output/",
        )

    def test_run_tpcav_no_pca(self):
        motif_path = Path("data") / "motif-clustering-v2.1beta_consensus_pwms.test.meme"
        genome_fasta = "data/hg38.analysisSet.fa"
        model = DummyModelSeq()
        layer_name = "layer1"

        cavs_fscores_df, motif_cav_trainers, bed_cav_trainer = run_tpcav(
            model=model,
            layer_name=layer_name,
            meme_motif_file=str(motif_path),
            genome_fasta=genome_fasta,
            num_motif_insertions=[4, 8],
            bed_seq_file="data/hg38_rmsk.head50k.bed",
            output_dir="data/test_run_tpcav_output/",
            num_pc=0
        )

    def test_write_bw(self):
        random_regions_1 = helper.random_regions_dataframe(
            "data/hg38.analysisSet.fa.fai", 1024, 100, seed=1
        )
        helper.write_attrs_to_bw(torch.rand((100, 1024)).numpy(), 
                                 random_regions_1.apply(lambda x: f"{x.chrom}:{x.start}-{x.end}", axis=1).tolist(), 
                                 "data/hg38.analysisSet.fa.fai", "data/test_run_tpcav_output/input_attrs.bw")

    def test_motif_concepts_against_permute_control(self):
        motif_path = Path("data") / "motif-clustering-v2.1beta_consensus_pwms.test.meme"
        self.assertTrue(motif_path.exists(), "Motif file is missing")

        builder = ConceptBuilder(
            genome_fasta="data/hg38.analysisSet.fa",
            input_window_length=1024,
            bws=None,
            num_motifs=16,
            include_reverse_complement=True,
            min_samples=1000,
            batch_size=8,
        )

        builder.build_control()

        concepts_pairs = builder.add_meme_motif_concepts(str(motif_path), build_permute_control=True)
        builder.apply_transform(transform_fasta_to_one_hot_seq)

        tpcav_model = TPCAV(DummyModelSeq(), layer_name="layer1")
        tpcav_model.fit_pca(
            concepts=builder.all_concepts(),
            num_samples_per_concept=10,
            num_pc="full",
        )
        cav_trainer = CavTrainer(tpcav_model)

        cav_trainer.train_concepts_pairs(concepts_pairs, 200, output_dir="data/cavs_permute/", num_processes=2)


    def test_all(self):
        lp = LineProfiler()
                # Add installed-package functions you care about
        lp.add_function(utils.iterate_seq_df_chunk)
        lp.add_function(CavTrainer.train_concepts)
        lp.enable_by_count()

        motif_path = Path("data") / "motif-clustering-v2.1beta_consensus_pwms.test.meme"
        self.assertTrue(motif_path.exists(), "Motif file is missing")

        builder = ConceptBuilder(
            genome_fasta="data/hg38.analysisSet.fa",
            input_window_length=1024,
            bws=None,
            num_motifs=12,
            include_reverse_complement=True,
            min_samples=1000,
            batch_size=8,
        )

        builder.build_control()

        builder.add_meme_motif_concepts(str(motif_path))

        builder.apply_transform(transform_fasta_to_one_hot_seq)

        batch = next(iter(builder.all_concepts()[0].data_iter))

        self.assertTupleEqual(batch[0].shape, (builder.batch_size, 4, 1024))

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        tpcav_model = TPCAV(DummyModelSeq(), layer_name="layer1").to(device)
        tpcav_model.fit_pca(
            concepts=builder.all_concepts(),
            num_samples_per_concept=10,
            num_pc="full",
        )
        torch.save(tpcav_model, "data/tmp_tpcav_model.pt")

        cav_trainer = CavTrainer(tpcav_model, penalty="l2")
        cav_trainer.set_control(builder.control_concepts[0], num_samples=100)

        cav_trainer.train_concepts(
            builder.concepts, 100, output_dir="data/cavs/", num_processes=2
        )
        torch.save(cav_trainer, "data/tmp_cav_trainer.pt")

        random_regions_1 = helper.random_regions_dataframe(
            "data/hg38.analysisSet.fa.fai", 1024, 100, seed=1
        )
        random_regions_2 = helper.random_regions_dataframe(
            "data/hg38.analysisSet.fa.fai", 1024, 100, seed=2
        )

        def pack_data_iters(df):
            seq_fasta_iter = helper.dataframe_to_fasta_iter(
                df, "data/hg38.analysisSet.fa", batch_size=8
            )
            seq_one_hot_iter = (
                helper.fasta_to_one_hot_sequences(seq_fasta)
                for seq_fasta in seq_fasta_iter
            )
            chrom_iter = helper.dataframe_to_chrom_tracks_iter(df, None, batch_size=8)
            return zip(
                seq_one_hot_iter,
            )

        attributions = tpcav_model.layer_attributions(
            pack_data_iters(random_regions_1), pack_data_iters(random_regions_2)
        ).cpu()

        cav_trainer.tpcav_score("AC0001:GATA-PROP:GATA", attributions)

        cav_trainer.plot_cavs_similaritiy_heatmap(attributions)

        input_attrs = tpcav_model.input_attributions(
            pack_data_iters(random_regions_1),
            pack_data_iters(random_regions_2),
            multiply_by_inputs=True,
            cavs_list=[
                cav_trainer.cav_weights["AC0001:GATA-PROP:GATA"],
            ],
        )

        # compute layer attributions using the old way
        random1_avs = []
        random2_avs = []
        for inputs in pack_data_iters(random_regions_1):
            av = tpcav_model._layer_output(*[i.to(tpcav_model.device) for i in inputs])
            random1_avs.append(av.detach().cpu())
        for inputs in pack_data_iters(random_regions_2):
            av = tpcav_model._layer_output(*[i.to(tpcav_model.device) for i in inputs])
            random2_avs.append(av.detach().cpu())
        random1_avs = torch.cat(random1_avs, dim=0)
        random2_avs = torch.cat(random2_avs, dim=0)

        random1_avs_residual, random1_avs_projected = tpcav_model.project_activations(
            random1_avs
        )
        random2_avs_residual, random2_avs_projected = tpcav_model.project_activations(
            random2_avs
        )

        def forward_from_layer_1_embeddings(tm, avs_residual, avs_projected):
            y_hat = tm.embedding_to_layer_activation(avs_residual, avs_projected)
            y_hat = tm.model.foward_from_layer1(y_hat)
            return y_hat

        tpcav_model.forward = partial(forward_from_layer_1_embeddings, tpcav_model)

        dl = DeepLift(tpcav_model)
        attributions_old = dl.attribute(
            (
                random1_avs_residual.to(tpcav_model.device),
                random1_avs_projected.to(tpcav_model.device),
            ),
            baselines=(
                random2_avs_residual.to(tpcav_model.device),
                random2_avs_projected.to(tpcav_model.device),
            ),
            custom_attribution_func=_abs_attribution_func,
        )
        attr_residual, attr_projected = attributions_old
        attributions_old = torch.cat((attr_projected, attr_residual), dim=1).cpu()

        self.assertTrue(
            torch.allclose(attributions.cpu(), attributions_old.cpu(), atol=1e-6),
            f"Attributions do not match, max difference is {torch.abs(attributions - attributions_old).max()}",
        )

        lp.disable_by_count()
        lp.print_stats()


if __name__ == "__main__":
    unittest.main()
