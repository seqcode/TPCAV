# TPCAV (Testing with PCA projected Concept Activation Vectors)

This repository contains code to compute TPCAV (Testing with PCA projected Concept Activation Vectors) on deep learning models. TPCAV is an extension of the original TCAV method, which uses PCA to reduce the dimensionality of the activations at a selected intermediate layer before computing Concept Activation Vectors (CAVs)

## Installation

`pip install tpcav`

## Usage

Example usage on explaining a dummpy model using motif concepts

```python

from tpcav import helper
from tpcav.cavs import CavTrainer
from tpcav.concepts import ConceptBuilder
from tpcav.tpcav_model import TPCAV, _abs_attribution_func

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

if __name__ == "__main__":
    motif_path = Path("data") / "motif-clustering-v2.1beta_consensus_pwms.test.meme"
    
    # create concept builder to generate concepts
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
    # use random regions as control  
    builder.build_control()
    # use meme motif PWMs to build motif concepts, one concept per motif
    builder.add_meme_motif_concepts(str(motif_path))
    # apply transform to convert fasta sequences to one-hot encoded sequences
    def transform_fasta_to_one_hot_seq(seq, chrom):
        return (helper.fasta_to_one_hot_sequences(seq),)
    builder.apply_transform(transform_fasta_to_one_hot_seq)
    
    # create TPCAV model on top of your model
    tpcav_model = TPCAV(DummyModelSeq(), layer_name="layer1")
    # fit PCA on sampled all concept activations
    tpcav_model.fit_pca(
        concepts=builder.all_concepts(),
        num_samples_per_concept=10,
        num_pc="full",
    )
    torch.save(tpcav_model, "data/tmp_tpcav_model.pt")
    
    # create trainer for computing CAVs
    cav_trainer = CavTrainer(tpcav_model, penalty="l2")
    # set control concept for CAV training
    cav_trainer.set_control(builder.control_concepts[0], num_samples=100)
    # train CAVs for all concepts
    cav_trainer.train_concepts(
        builder.concepts, 100, output_dir="data/cavs/", num_processes=2
    )
    
    # create input regions and baseline regions for attribution
    random_regions_1 = helper.random_regions_dataframe(
        "data/hg38.analysisSet.fa.fai", 1024, 100, seed=1
    )
    random_regions_2 = helper.random_regions_dataframe(
        "data/hg38.analysisSet.fa.fai", 1024, 100, seed=2
    )
    # create iterators to yield one-hot encoded sequences from the region dataframes
    def pack_data_iters(df):
        seq_fasta_iter = helper.dataframe_to_fasta_iter(
            df, "data/hg38.analysisSet.fa", batch_size=8
        )
        seq_one_hot_iter = (
            helper.fasta_to_one_hot_sequences(seq_fasta)
            for seq_fasta in seq_fasta_iter
        )
        return zip(
            seq_one_hot_iter,
        )
    # compute layer attributions
    attributions = tpcav_model.layer_attributions(
        pack_data_iters(random_regions_1), pack_data_iters(random_regions_2)
    )["attributions"]
    # compute TPCAV scores for all concepts
    cav_trainer.tpcav_score_all_concepts_log_ratio("AC0001:GATA-PROP:GATA", attributions)
```
