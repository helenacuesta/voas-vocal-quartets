# Voice Assignment of Vocal Quartets

**Note:**: *This repo is a work in progress. This README file is the ongoing documentation, 
and all scripts are not yet uploaded/up-to-date.*

This repository contains the accompanying code for the paper:

Helena Cuesta & Emilia Gómez. (2022). **Voice Assignment in Vocal Quartets Using Deep Learning Models Based on Pitch Salience**. Transactions of the International Society for Music Information Retrieval (TISMIR), 5(1), 99–112. DOI: http://doi.org/10.5334/tismir.121.

## Description
This is the accompanying code repository for the paper mentioned above. It currently contains the trained models for 
*Voice Assignment* (VA), namely `voas_cnn` and `voas_clstm` and the associated code to use them with pre-extracted 
pitch salience representations -- see **Priors** for information on how to obtain such representations.

Additionals scripts to reproduce/extend the examples from the *Synth-salience Choral Set* (SSCS), the 
synthetic dataset considered in this work, will be provided soon. 

The dataset can be downloaded following 
<a href="https://zenodo.org/record/6534429">this link</a>, and the 
training/validation/test data splits are available in `data/data_splits_hpc.json`.

## Usage

#### Priors
These VA models take polyphonic pitch salience representations as input, and they assume four different sources.
In our paper, we specifically propose a framework that combines the output of Late/Deep (https://github.com/helenacuesta/multif0-estimation-polyvocals, model3) with the 
proposed VA models---VoasCNN and VoasCLSTM, on vocal quartets.


#### Inference 

To use the trained models, please run the `predict_on_salience.py` script with the following parameters:


    --model: Model to use for prediction: voas_clstm | voas_cnn

    --saliencefile: Path to the input salience file. It expects a npy files.

    --saliencefolder: Path to the folder with salience files.

    --outputpath: Path to the folder to store the results. If nothing is provided, results will be stored in the same folder of the input(s).")
  
 

