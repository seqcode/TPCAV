# TPCAV (Testing with PCA projected Concept Activation Vectors)

This repository contains code to compute TPCAV (Testing with PCA projected Concept Activation Vectors) on deep learning models. TPCAV is an extension of the original TCAV method, which uses PCA to reduce the dimensionality of the activations at a selected intermediate layer before computing Concept Activation Vectors (CAVs) to improve the consistency of the results.

## Installation

`pip install tpcav`

## Quick start

```python
import torch
from tpcav import run_tpcav

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

# transformation function to obtain one-hot encoded sequences
def transform_fasta_to_one_hot_seq(seq, chrom):
    # `seq` is a list of fasta sequences
    # `chrom` is a numpy array of bigwig signals of shape [-1, # bigwigs, len]
    return (helper.fasta_to_one_hot_sequences(seq),) # it has to return a tuple of inputs, even if there is only one input

motif_path = "data/motif-clustering-v2.1beta_consensus_pwms.test.meme"
bed_seq_concept = "data/hg38_rmsk.head500k.bed"
genome_fasta = "data/hg38.analysisSet.fa"
model = DummyModelSeq() # load the model
layer_name = "layer1"   # name of the layer to be interpreted

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
    input_transform_func=transform_fasta_to_one_hot_seq
)

# check each trainer for detailed weights
print(bed_cav_trainer.cav_weights)

```


## Detailed Usage

For detailed usage, please refer to this [jupyter notebook](https://github.com/seqcode/TPCAV/tree/main/examples/tpcav_detailed_usage.ipynb)

If you find any issue, feel free to open an issue (strongly suggested) or contact [Jianyu Yang](jmy5455@psu.edu).

