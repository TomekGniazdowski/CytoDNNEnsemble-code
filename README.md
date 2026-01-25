# Source code for the _Towards Efficient Cytometry-Based Diagnosis with Deep Ensembles_ publication

[Paper Link]()

## Method

Flow cytometry is a cornerstone diagnostic technique, but manual analysis is laborious, subjective, and prone to variability. We propose a parameter-efficient deep ensemble model for automated analysis of high-dimensional flow cytometry data. Our method reduces the reliance on expert gating, captures complex cellular patterns, and improves diagnostic consistency. Extensive experiments demonstrate that the proposed model outperforms state-of-the-art approaches in accuracy and robustness for relatively large datasets, highlighting its potential for real-world clinical deployment.

![method_image](assets/CytoDNNEnsemble_scheme.png)

An example of processing a single sample with dimensionality _[num_cells,num_markers]_ is shown in the scheme. A single sample is processed independently from two complementary perspectives. In the first model—the 1D Convolutional Neural Network Extractor (CNN1DE, upper branch of the scheme)—the sample is fed into the model and processed row by row, using the marker values returned by individual cells. In the second model—the Latent Space Extractor (LSE, lower branch of the scheme)—the values corresponding to individual markers (columns) are separated, their histograms are computed and normalized, and their latent representations are then obtained using a pretrained convolutional encoder. These representations are concatenated and used for classification. In this case, the sample is analyzed column by column. The predictions of the individual models, CNN1DE and LSE, are subsequently concatenated and passed to a final classifier, which produces the final decision. Together, CNN1DE, LSE, and the final classifier form the CytoDNNEnsemble (CDE) model. The CNN1DE and LSE models are trained independently. In the final training stage, during training of the final classifier, the weights of CNN1DE and LSE can either be frozen or the entire pipeline can be trained end-to-end. This approach enables the model to capture fine-grained, cell-level information through CNN1DE and global, marker-level trends through LSE. By combining these two perspectives, the ensemble is able to extract more informative features and produce more accurate predictions.


## How to run
All the scripts that provide training all the models from the publication are in `experiments` folder.

### Single models
For example to train the `CNN1DE` model on AML data you must run:
```
cd experiments/ex_1_single_extractors/AML
python train_AML_cnn_1d_extractor.py
```

### CDE model
To train the final `CDE` model on AML data (with pretrained LSE and CNN1DE models) you must run:
```
cd experiments/ex_2_final_models/AML
python train_AML_final_model_clf_best.py
```

## Helpful code
We rely on the [CytoSet](https://github.com/CompCy-lab/cytoset) source code for fair comparsion.

## Citation
TBA