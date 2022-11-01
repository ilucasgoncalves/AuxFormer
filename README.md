# AuxFormer
AuxFormer: Robust Approach to Audiovisual Emotion Recognition

## Abstract
A challenging task in audiovisual emotion recognition is to implement neural network architectures that can leverage and fuse multimodal information while temporally aligning modalities, handling missing modalities, and capturing information from all modalities without losing information during training. These requirements are important to achieve model robustness and to increase accuracy on the emotion recognition task. A recent approach to perform multimodal fusion is to use the transformer architecture to properly fuse and align the modalities. This study proposes the AuxFormer framework, which addresses in a principled way the aforementioned challenges. AuxFormer combines the transformer framework with auxiliary networks. It uses shared losses to infuse information from single-modality networks that are separately embedded. The extra layer of audiovisual information added to our main network retains information that would otherwise be lost during training.

### Paper
Lucas Goncalves and Carlos Busso, "AuxFormer: Robust approach to audiovisual emotion recognition," in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2022), Singapore, May 2022.

## Using model

#### Update (10/31/2022): 

Updated codes for easier implementation of the model and feature extracting. Releasing corpora partitions for experimental evaluations and features to be used. In this update, we are releasing the wav2vec2 features and EMOCA features version. Opensmile and VGG-Face features are being updated for upcoming release.

### Dependencies
* Python 3.9.7
* Pytorch 1.12.0
* To create conda environment based on requirements use: `conda install --name AuxFormer_env --file requirements.txt`
* Note: `pip install transformers` is needed after creating env
* Activate environment with: `conda activate AuxFormer_env`

### Datasets Used
1. [CREMA-D.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4313618/) 
2. [MSP-IMPROV](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html)

### Features
* Access to features [here](https://drive.google.com/drive/folders/1EXd-RLwyzoplM8HtNmpwKu1BwavJRxpl?usp=sharing)
Note: Download the features and enter the paths to where you saved the features locally in "main.py" lines 10,11, and 12.



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
     
2. Execute run_model.sh

       `conda activate AuxFormer_env`
       `bash run_model.sh`
       
## Framework

The AuxFormer framework, which consists of the main audiovisual fusion network (middle) labelled fav(•), the auxiliary acoustic
network (top) labelled fa(•), and the auxiliary visual network (bottom) labelled fv(•).

<p align="center">
  <img src="./images/model.png" />
</p>

