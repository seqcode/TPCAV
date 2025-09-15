# TPCAV (Testing with PCA projected Concept Activation Vectors)

Analysis pipeline for TPCAV

## Dependencies

You can use your own environment for the model, in addition, you need to install the following packages:

- captum 0.7
- seqchromloader 0.8.5
- scikit-learn 1.5.2

## Workflow

1. Since not every saved pytorch model stores the computation graph, you need to manually add functions to let the script know how to get the activations of the intermediate layer and how to proceed from there.

    There are 3 places you need to insert your own code.
    
    - Model class definition in models.py
        - Please first copy your class definition into `Model_Class` in the script, it already has several pre-defined class functions, you need to fill in the following two functions:
            - `forward_until_select_layer`: this is the function that takes your model input and forward until the layer you want to compute TPCAV score on
            - `resume_forward_from_select_layer`: this is the function that starts from the activations of your select layer and forward all the way until the end
        -  There are also functions necessary for TPCAV computation, don't change them:
            - `forward_from_start`: this function calls `forward_until_select_layer` and `resume_forward_from_select_layer` to do a full forward pass
            - `forward_from_projected_and_residual`: this function takes the PCA projected activations and unexplained residual to do the forward pass
            - `project_avs_to_pca`: this function takes care of the PCA projection
    
        > NOTE: you can modify your final output tensor to specifically explain certain transformation of your output, for example, you can take weighted sum of base pair resolution signal prediction to emphasize high signal region.
    
    - Function `load_model` in utils.py
        - Take care of the model initialization and load saved parameters in `load_model`, return the model instance.
        > NOTE: you need to use your own model class definition in models.py, as we need the functions defined in step 1.
    
    - Function `seq_transform_fn` in utils.py
        - By default the dataloader provides one hot coded DNA array of shape (batch_size, 4, len), coded in the order [A, C, G, T], if your model takes a different kind of input, modify `seq_transform_fn` to transform the input

    - Function `chrom_transform_fn` in utils.py
        - By default the dataloader provides signal array from bigwig files of shape (batch_size, # bigwigs, len), if your model takes a different kind of chromatin input, modify `chrom_transform_fn` to transform the input, if your model is sequence only, leave it to return None.


2. Compute CAVs on your model, example command:

```bash
srun -n1 -c8 --gres=gpu:1 --mem=128G python scripts/run_tcav_sgd_pca.py \
  cavs_test 1024 data/hg19.fa data/hg19.fa.fai \
  --meme-motifs data/motif-clustering-v2.1beta_consensus_pwms.test.meme \
  --bed-chrom-concepts data/ENCODE_DNase_peaks.bed
```

3. Then compute the layer attributions, example command:

```bash
srun -n1 -c8 --gres=gpu:1 --mem=128G \
  python scripts/compute_layer_attrs_only.py cavs_test/tpcav_model.pt \
  data/ChIPseq.H1-hESC.MAX.conservative.all.shuf1k.narrowPeak \
  1024 data/hg19.fa data/hg19.fa.fai cavs_test/test 
```

4. run the jupyer notebook to generate summary of your results

```bash
papermill -f scripts/compute_tcav_v2_pwm.example.yaml scripts/compute_tcav_v2_pwm.py.ipynb cavs_test/tcav_report.py.ipynb
```

