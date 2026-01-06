import unittest
from pathlib import Path

import torch
from Bio import motifs as Bio_motifs

from tpcav import helper
from tpcav.cavs import CavTrainer
from tpcav.concepts import ConceptBuilder
from tpcav.tpcav_model import TPCAV


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


class CavTrainerIntegrationTest(unittest.TestCase):

    def test_motif_concepts(self):
        motif_path = Path("data") / "motif-clustering-v2.1beta_consensus_pwms.test.meme"
        self.assertTrue(motif_path.exists(), "Motif file is missing")

        builder = ConceptBuilder(
            genome_fasta="data/hg38.analysisSet.fa",
            genome_size_file="data/hg38.analysisSet.fa.fai",
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

    def test_all(self):

        motif_path = Path("data") / "motif-clustering-v2.1beta_consensus_pwms.test.meme"
        self.assertTrue(motif_path.exists(), "Motif file is missing")

        builder = ConceptBuilder(
            genome_fasta="data/hg38.analysisSet.fa",
            genome_size_file="data/hg38.analysisSet.fa.fai",
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

        tpcav_model = TPCAV(DummyModelSeq(), layer_name="layer1")
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
        )["attributions"]

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


if __name__ == "__main__":
    unittest.main()
