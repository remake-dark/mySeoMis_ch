# Counterfactual Debiasing for Social Misinformation Detectors under Prevalence of Major Social Events

This repository contains the code for our **Capstone Project**, which focuses on counterfactual learning using a late fusion structure. Below are important tips for reproducing our experiments, details on the experimental setup, and acknowledgements for the frameworks and tools we utilized.

## 1. Introduction

This project is part of our **Capstone Project**. Unlike other debiasing methods, our project is based on **counterfactual learning**, which assumes that the validation set is ineffective to a certain degree. As a result, standard early stopping during training is not advisable.

## 2. Data

The dataset for this project can be downloaded from the following link:

[Download the dataset](https://drive.google.com/file/d/1keS6csdpgvTV9cAsjRoXn2xI5XTxeSnt/view?usp=sharing)

Please download the data and place it in the correct directory before running the experiments.

## 3. Important Tips for Reproducing the Project

Counterfactual learning differs from other methods, and it's commonly conducted in a traditional data mining workflow, which involves:
1. **Training for a set number of epochs** instead of using early stopping.
2. **Saving the model checkpoint** once training is completed.
3. **Manual control**: After loading the saved checkpoint, continue training manually in increments of two epochs. Observe the results and adjust the coefficients or decide whether to continue training.

This process relies on experience and can be time-consuming, especially if you're more familiar with large-scale language model training. It requires careful adjustment of the number of epochs and coefficients to achieve optimal results. Be patient, as this manual process can take time to fine-tune. Alternatively, you could set up a large-scale grid search to automate some of the tuning, though this approach is less efficient.

## 4. Experimental Setup

The following experimental setup is recommended for reproducing our results:
- For an experiment group with **batch size 64** and **sequence length 384**, a **NVIDIA 4090 (24GB) GPU** is required.
- For an experiment group with **batch size 128**, we recommend using a **NVIDIA A6000 or A100 PCIE 40G GPU**.
- Additionally, using a more powerful CPU can help improve the experiment speed.

## 5. Project Framework

Our project is built upon the framework of [ENDEF (2022)](https://github.com/ICTMCG/ENDEF-SIGIR2022). We extend this work and are grateful for the contributions of its authors.

To run the **Social Event Extraction** module, you need to install [OmniEvent](https://github.com/THU-KEG/OmniEvent) (developed by Tsinghua University) and download the corresponding model weights. Please note that **OmniEvent** recently had a problematic update, which might make the download process challenging.

## 6. Replication Notes

Although our method generally employs a **late fusion structure**, when replicating the results, please ensure that **all modules** (except the event extractor) use the same **tokenizer**.

## 7. Acknowledgements

We would like to express our sincere thanks to:
- The authors of [ENDEF](https://github.com/ICTMCG/ENDEF-SIGIR2022) for providing the framework.
- The authors of [OmniEvent](https://github.com/THU-KEG/OmniEvent) for their contributions to the Social Event Extraction module.
  
Finally, we sincerely thank our instructor for their guidance throughout this semester's course.
