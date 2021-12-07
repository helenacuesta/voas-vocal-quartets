# Voice Assignment of Vocal Quartets

**This repo is a work in progress. This README file is the ongoing documentation, and all scripts are not yet uploaded/up-to-date.**

This repository contains the accompanying code for the paper:

[Paper REF]

## Description
TBD

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
  
 

