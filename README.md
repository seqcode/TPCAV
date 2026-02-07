# TPCAV 
> Testing with PCA projected Concept Activation Vectors

This repository contains code to compute TPCAV (Testing with PCA projected Concept Activation Vectors) on deep learning models. TPCAV is an extension of the original TCAV method, which uses PCA to reduce the dimensionality of the activations at a selected intermediate layer before computing Concept Activation Vectors (CAVs) to improve the consistency of the results.

For more technical details, please check our manuscript on Biorxiv [TPCAV: Interpreting deep learning genomics models via concept attribution](https://doi.org/10.64898/2026.01.20.700723)!

## When should I use TPCAV?

TPCAV is a global feature attribution method that can be applied to any model, provided that a set of examples is available to represent the concept of interest. It is input-agnostic, meaning it can operate on raw inputs, engineered features, or **tokenized representations**, including **foundation models**.

Typical concepts in Genomics include:
- Transcription factor motifs
- Cis-regulatory regions
- DNA repeats

The same framework naturally extends to other domains, such as protein structure prediction, transcriptomics, or any field with a well established knowledge base, by defining appropriate concept sets.

## Installation

`pip install tpcav`

## Detailed Usage

For detailed usage for more flexibility on defining concepts, please refer to this [jupyter notebook](https://github.com/seqcode/TPCAV/tree/main/examples/tpcav_detailed_usage.ipynb)

## Quick start

> `tpcav` only works with Pytorch model, if your model is built using other libraries, you should port the model into Pytorch first. For Tensorflow models, you can use [tf2onnx](https://github.com/onnx/tensorflow-onnx) and [onnx2pytorch](https://github.com/Talmaj/onnx2pytorch) for the conversion.

```python
import torch
from tpcav import run_tpcav

#==================== Prepare Model and Data transform function ================================
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

# By default, every concept extracts fasta sequences and bigwig signals from the given region
# Use your own custom transformation function to get your desired inputs
# Here we transform them into one-hot coded DNA sequences
def transform_fasta_to_one_hot_seq(seq, chrom):
    # `seq` is a list of fasta sequences
    # `chrom` is a numpy array of bigwig signals of shape [-1, # bigwigs, len]
    return (helper.fasta_to_one_hot_sequences(seq),) # it has to return a tuple of inputs, even if there is only one input

#==================== Construct concepts ================================
motif_path = "data/motif-clustering-v2.1beta_consensus_pwms.test.meme" # motif file in meme format for constructing motif concepts
bed_seq_concept = "data/hg38_rmsk.head500k.bed" # a bed file to supply concepts described by a set of regions, format [chrom, start, end, label, concept_name]
genome_fasta = "data/hg38.analysisSet.fa"
model = DummyModelSeq() # load the model
layer_name = "layer1"   # name of the layer to be interpreted, you should be able to retrieve the layer object by getattr(model, layer_name)

# concept_fscores_dataframe: fscores of each concept
# motif_cav_trainers: each trainer contains the cav weights of motifs inserted different number of times
# bed_cav_trainer: trainer contains the cav weights of the sequence concepts provided in bed file
concept_fscores_dataframe, motif_cav_trainers, bed_cav_trainer = run_tpcav(
    model=model,
    layer_name=layer_name,
    meme_motif_file=motif_path,
    genome_fasta=genome_fasta,
    num_motif_insertions=[4, 8],
    bed_seq_file=bed_seq_concept, 
    output_dir="test_run_tpcav_output/",
    input_transform_func=transform_fasta_to_one_hot_seq)

#==================== Compuate layer attributions of target testing regions ================================
# retrieve the tpcav model
tpcav_model = bed_cav_trainer.tpcav

# create input regions and baseline regions for attribution
random_regions_1 = helper.random_regions_dataframe(genome_fasta + ".fai", 1024, 100, seed=1)
random_regions_2 = helper.random_regions_dataframe(genome_fasta + ".fai", 1024, 100, seed=2)
# create iterators to yield one-hot encoded sequences from the region dataframes
def pack_data_iters(df):
    seq_fasta_iter = helper.dataframe_to_fasta_iter(df, genome_fasta, batch_size=8)
    seq_one_hot_iter = (helper.fasta_to_one_hot_sequences(seq_fasta) for seq_fasta in seq_fasta_iter)
    return zip(seq_one_hot_iter, )
# compute layer attributions given the iterators of testing regions and control regions
attributions = tpcav_model.layer_attributions(pack_data_iters(random_regions_1), pack_data_iters(random_regions_2))["attributions"]
# compute TPCAV scores for the concept
# here uses bed_cav_trainer that contains the concepts provided from bed file
bed_cav_trainer.tpcav_score_all_concepts_log_ratio(attributions)
```



If you find any issue, feel free to open an issue (strongly suggested) or contact [Jianyu Yang](mailto:jmy5455@psu.edu).

