# AuxFormer
AuxFormer: Robust Approach to Audiovisual Emotion Recognition

## Abstract
A challenging task in audiovisual emotion recognition is to implement neural network architectures that can leverage and fuse multimodal information while temporally aligning modalities, handling missing modalities, and capturing information from all modalities without losing information during training. These requirements are important to achieve model robustness and to increase accuracy on the emotion recognition task. A recent approach to perform multimodal fusion is to use the transformer architecture to properly fuse and align the modalities. This study proposes the AuxFormer framework, which addresses in a principled way the aforementioned challenges. AuxFormer combines the transformer framework with auxiliary networks. It uses shared losses to infuse information from single-modality networks that are separately embedded. The extra layer of audiovisual information added to our main network retains information that would otherwise be lost during training.

### Paper
Lucas Goncalves and Carlos Busso, "AuxFormer: Robust approach to audiovisual emotion recognition," in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2022), Singapore, May 2022.

## Using model

### Dependencies
* Python 3.8.5
* Pytorch 1.9.0
* CUDA 10.2

### Datasets Used
1. [CREMA-D.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4313618/) 
2. [MSP-IMPROV](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html)

### Scripts
For more details make sure to visit these files to look at script arguments and description:

`AuxFormer/main.py ` - main script to load the saved dataset files and input the model settings

`AuxFormer/src/dataset.py` - dataset loader

`AuxFormer/src/eval_metrics.py ` - evaluation metrics for testing model (F1-Scores)

`AuxFormer/src/models.py` - framework initialization and set-up

`AuxFormer/src/training.py ` - training script

`AuxFormer/src/utils.py` - utils script to retrieve data, load saved models, and save trained models

`AuxFormer/modules/` - folder containing tranformer framework and position_embedding configurations

### Running the Algorithm
1. create folders for dataset location (save the downloaded data in 'dataset/') and model saving 

       `mkdir dataset saved_models`
     
2. Execute main.py 

       `python3 main.py [--FLAGS]`
       
## Framework

The AuxFormer framework, which consists of the main audiovisual fusion network (middle) labelled fav(???), the auxiliary acoustic
network (top) labelled fa(???), and the auxiliary visual network (bottom) labelled fv(???).

<p align="center">
  <img src="./images/model.png" />
</p>


## Acknowledgement
Some portion of the code were adapted from the fairseq and yaohungt repo.
