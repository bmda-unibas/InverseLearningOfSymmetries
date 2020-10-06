
# Inverse Learning of Symmetries

This repository contains a basic implementation of our NeurIPS 2020 paper [Inverse Learning of Symmetries](https://arxiv.org/pdf/2002.02782.pdf). 

<p align="center">
  <img src="images/main.png" width=200/>
</p>


## Requirements

The code was tested under Python 2.7.12. To run the code, it is necessary to install the following dependencies from the requirements file.

Open a new terminal and create a virtualenv:
```
mkdir symmetries
cd symmetries
git clone https://github.com/bmda-unibas/InverseLearningOfSymmetries.git
cd ..
virtualenv symmetries/paper
```

Activate the environment:
```
source symmetries/paper/bin/activate
```
Install the dependencies:
```
pip install -r symmetries/InverseLearningOfSymmetries/requirements.txt
```

## Training

To train the artificial experiment in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```


## Evaluation

To evaluate the artificial experiment, run this command:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```


## Pre-trained Models

You can find the pretrained models here: [Models](pretrained/)


## Results

Here we report the results of our basic impplementation:

### Qualitative Results

### Quantitative Results
| Model name         | Top 1 Accuracy  | Top 5 Accuracy | Top 5 Accuracy |
| ------------------ |---------------- | -------------- | -------------- |
| VAE   |     85%         |      95%       | Top 5 Accuracy |
| My awesome model   |     85%         |      95%       | Top 5 Accuracy |
| My awesome model   |     85%         |      95%       | Top 5 Accuracy |
| My awesome model   |     85%         |      95%       | Top 5 Accuracy |
| My awesome model   |     85%         |      95%       | Top 5 Accuracy |


## Reference


If you like our paper and use it for your research, please cite us.

```
@incollection{Wieser,
title = {Improved Variational Inference with Inverse Autoregressive Flow},
author = {Wieser, Mario and Parbhoo, Sonali and Wieczorek, Aleksander and Roth, Volker},
booktitle = {Advances in Neural Information Processing Systems 34},
year = {2020}
}
```