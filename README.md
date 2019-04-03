# Pytorch-yolo-phoc
Implementation on pytorch of the code from the ECCV 2018 paper - Single Shot Scene Text Retrieval. 
Paper: https://arxiv.org/abs/1808.09044

#### Training YOLO-PHOC
---
All paths are hardcoded and need to be edited accordingly.


##### Modify Cfg files for training
Change the cfg/XXXX.data file according to training objective
```
train  = path_to_file_with_list_of_files_to_train.txt
valid  = path_to_file_with_list_of_files_for_validation.txt
names = data/recognition.names
backup = backup
gpus  = 0
num_workers = 10
```
The file cfg/XXXX.cfg contains the config parameters for training.

##### Download Pretrained Convolutional Weights
Download weights from the convolutional layers (Imagenet pre-trained weights)
```
wget http://pjreddie.com/media/files/darknet19_448.conv.23
```
##### Train The Model
Modify the options in train.py file.
```
python train.py
```
---
#### Detection Using A Pre-Trained Model
---
The model has been trained, achieving the following results:
 - IIIT Scene Text Retrieval dataset: 67.26
 - IIIT Sports-10k dataset: 72.73
 - Street View Text (SVT) dataset: 83.14

The weights can be downloaded from: https://drive.google.com/open?id=10NSDA8zjs-EEA9f3rj1smSuvfBYu2jm3
