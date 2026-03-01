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

## Inputs

- Required inputs:
    - Genome fasa filet
    - Model in Pytorch

- Concept source file:
    - If you want to test motif concepts, please provide motifs in either of the two formats:
        - Candidate motif PWMs in MEME MINIMAL format, motif database used in our manuscript is from [Jeff Viestra Lab](https://resources.altius.org/~jvierstra/projects/motif-clustering-v2.1beta/)
        - A tab delimited file with consensus motif information, you can provide multiple consensus sequences for a single motif:
          ```text
          <motif_name>    <consensus_seq>
          motif1    ATATAAAA
          motif2    AACGGGCA
          motif2    ATTCCCAA
          ...
          ```
    
    - If you want to test concepts provided in genomic coordinates, please provide them as bed file in the following way, repeats coordinates in the manuscript are downloaded from [RepeatMasker database](https://www.repeatmasker.org/):
        ```text
        <chrom>  <start> <end>    <strand>    <concept name>  
        chr1	16363	16459	-	DNA/hAT-Charlie_Charlie15a
        chr1	16713	16744	+	Simple_repeat_(TGG)n
        chr1	18907	19048	+	LINE/L2_L2a
        ...
        ```

> `tpcav` only works with Pytorch model, if your model is built using other libraries, you should port the model into Pytorch first. For Tensorflow models, you can use [tf2onnx](https://github.com/onnx/tensorflow-onnx) and [onnx2pytorch](https://github.com/Talmaj/onnx2pytorch) for the conversion.

## Detailed Usage

For detailed usage for more flexibility on defining concepts, please refer to this [jupyter notebook](https://github.com/seqcode/TPCAV/tree/main/examples/tpcav_detailed_usage.ipynb)

## Quick start

Example usage on a simple model trained for predicting CTCF binding in MCF-7 cell line, you would need to download [hg38 genome](https://hgdownload.gi.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz) to run, other used files can be found in `data/` directory

Here we test a couple of motif concepts (including CTCF cognate motif) and some sampled repeat concepts.

```python
import torch
from tpcav import run_tpcav, helper
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#==================== Prepare Model and Data transform function ================================
class DummyModelSeq(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = torch.nn.Sequential(
            torch.nn.Conv1d(4, 64, 25, padding=12, bias=True),
            torch.nn.BatchNorm1d(64),
            torch.nn.LeakyReLU()
        )
        self.conv_layer_2 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, 3, padding=1, stride=2, bias=True),
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(),
        )

        self.linear_layer_1 = torch.nn.Sequential(
            torch.nn.Linear(512, 1),
            torch.nn.LeakyReLU()
        )
        self.linear_layer_2 = torch.nn.Linear(128, 1)

    def forward(self, seq):
        y_hat = self.conv_layer_1(seq)
        y_hat = self.conv_layer_2(y_hat)
        y_hat = self.linear_layer_1(y_hat)
        y_hat = y_hat.squeeze(-1)
        y_hat = self.linear_layer_2(y_hat)
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
bed_seq_concept = "data/hg38_rmsk.sample.bed" # a bed file to supply concepts described by a set of regions, format [chrom, start, end, label, concept_name]
genome_fasta = "data/hg38.analysisSet.fa"
model = torch.load("data/mcf7_ctcf_best.pt", map_location=device, weights_only=False) # load the model
layer_name = "linear_layer_1"   # name of the layer to be interpreted, you should be able to retrieve the layer object by getattr(model, layer_name)

# concept_fscores_dataframe: fscores of each concept
# motif_cav_trainers: each trainer contains the cav weights of motifs inserted different number of times
# bed_cav_trainer: trainer contains the cav weights of the sequence concepts provided in bed file
concept_fscores_dataframe, motif_cav_trainers, bed_cav_trainer = run_tpcav(
    model=model,
    layer_name=layer_name,
    motif_file=motif_path,
    motif_file_fmt='meme',  # specify your motif file format, either meme or consensus (tab delimited file in form [motif_name, consensus_sequence])
    genome_fasta=genome_fasta,
    num_motif_insertions=[12, 24, 36],
    bed_seq_file=bed_seq_concept, 
    output_dir="test_run_tpcav_output/",
    input_transform_func=transform_fasta_to_one_hot_seq,
    p=4) # number of concurrent SGDClassifier can be run at the same time, increase it if you have available CPU power, it speeds up training significantly
```

There will be a `report.html` file generated in the output folder for quick inspection of the results

```python
from tpcav import report

#==================== Compuate layer attributions of target testing regions ================================
# retrieve the tpcav model
tpcav_model = bed_cav_trainer.tpcav

# create input regions and baseline regions for attribution
ctcf_peaks = helper.load_bed_and_center("data/MCF-7_CTCF_ENCFF942TCG.bed", window=1024).sample(n=100)
random_regions = helper.random_regions_dataframe(genome_fasta + ".fai", 1024, 100, seed=2)

# create iterators to yield one-hot encoded sequences from the region dataframes
# adjust this funtion to fit your model input format requirements
def pack_data_iters(df):
    seq_fasta_iter = helper.dataframe_to_fasta_iter(df, genome_fasta, batch_size=8)
    seq_one_hot_iter = (helper.fasta_to_one_hot_sequences(seq_fasta) for seq_fasta in seq_fasta_iter)
    return zip(seq_one_hot_iter, )

# compute layer attributions given the iterators of testing regions and control regions
attributions = tpcav_model.layer_attributions(pack_data_iters(ctcf_peaks), pack_data_iters(random_regions))

# generate a new html report with TPCAV score computed
report.generate_tpcav_html_report("report_tpcav_score.html", motif_cav_trainers,
                               non_motif_cav_trainers = {'repeats': bed_cav_trainer},
                               attributions = [attributions, ],  # if you have multiple sets of attributions you can provide a list
                               motif_file=motif_path, motif_file_fmt='meme', fscore_thresh=0.8)

# save the trainers for future use
torch.save(motif_cav_trainers, "motif_cav_trainers.pt")
torch.save(bed_cav_trainer, "bed_cav_trainer.pt")
```

Check `report_tpcav_score.html` for Log TPCAV score of each concept.

## HTML output

An example report can be found [here](https://github.com/seqcode/TPCAV/blob/main/data/report_tpcav_score.html.zip?raw=true). The HTML output contains three sections in general:
1. Tables listing F-scores of all concepts, for motif concepts there are F-scores at different number of insertions.
2. Motif concept ranking by AUC F-scores or corrected AUC F-scores (motif concept sensitivity)
3. Concept activaton vectors (CAVs) similarity matrix heatmap and TPCAV score


## Restore trained concepts

The results of TPCAV are stored in `CavTrainer` object, it contains the F-score of each concept, the corresponding concept activation vector (CAV), and the model object decorated by TPCAV parameters & functions, given the example in Quick Usage:

```python
# reload trainers back
motif_cav_trainers = torch.load("motif_cav_trainers.pt")
bed_cav_trainer = torch.load("bed_cav_trainer.pt")

# inspect trainer properties
cav_trainer = motif_cav_trainers[0] # here we take the first motif cav trainer that correponds to the first number of motif insertions
# retrieve F-scores
motif_cav_trainers[0].cav_fscores
# retrieve CAVs
motif_cav_trainers[0].cav_weights
```

You can also retrieve the model decorated by TPCAV parameters by

```python
tpcav_mode = cav_trainer.tpcav
```

So that you can compute attributions for new inputs

```python
# compute layer attributions, and compute new tpcav score
attrs = tpcav_model.layer_attributions(target_batches, baseline_batches)
cav_trainer.tpcav_score_all_concepts_log_ratio(attrs)
```

You can also generate new reports using the computed attributions

```python
report.generate_tpcav_html_report("report.html", motif_cav_trainers,
                                 non_motif_cav_trainers = {'repeats': bed_cav_trainer},
                                 motif_file=motif_path, motif_file_fmt='meme',
                                 fscore_thresh=0.8)

```

You can also extract concept specific attribution score by providing a list of cavs

```python
# input attributions
input_attrs = tpcav_model.input_attributions(target_batches, baseline_batches, multiply_by_inputs=True,)
# or concept specific input attributions (parts explained by the provided concepts CAVs)
input_attrs = tpcav_model.input_attributions(target_batches, baseline_batches, multiply_by_inputs=True, cavs_list=[cav_trainer.cav_weights[concept_name])
```

If you find any issue, feel free to open an issue (strongly suggested) or contact [Jianyu Yang](mailto:yztxwd@gmail.com).

