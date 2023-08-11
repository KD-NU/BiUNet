# BiUNet


This repo holds code for [BiUNet: Towards More Effective U-Net with Bi-Level Routing Attention](https://arxiv.org/pdf/2102.04306.pdf)

## Installation

```angular2html
conda create -n biunet python=3.7 -y
conda activate biunet
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3.1 torchaudio==0.10.1 -c pytorch -c conda-forge -y
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
#### 1.1. QaTa-COV19 and MoNuSeg Datasets
The original data can be downloaded in following links:
* QaTa-COV19 Dataset - [Link (Original)](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset)

* MoNuSeG Dataset - [Link (Original)](https://monuseg.grand-challenge.org/Data/)

#### 1.2. Format Preparation

Then prepare the datasets in the following format for easy use of the code:

```angular2html
├── datasets
    ├── QaTa-Covid19
    │   ├── Test_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   ├── Train_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    │       ├── img
    │       └── labelcol
    └── MoNuSeg
        ├── Test_Folder
        │   ├── img
        │   └── labelcol
        ├── Train_Folder
        │   ├── img
        │   └── labelcol
        └── Val_Folder
            ├── img
            └── labelcol
```



### 2. Training
The first step is to change the settings in Config.py, all the configurations including learning rate, batch size and etc. are in it. Then run:

```angular2html
python train_model.py
```




### 3. Testing
#### 3.1. Get Pre-trained Models
Here, we provide pre-trained weights on Covid19 and MoNuSeg, if you do not want to train the models by yourself, you can download them in the following links:
* Covid：https://pan.baidu.com/s/1hacTAlo2hkNIIUbNsyM26A
  
  training_log: https://pan.baidu.com/s/1LSg_omWie5rC_1hcPhIGDA
* MoNuSeg: https://pan.baidu.com/s/1j6o1Xz9j6Jpfc4q7rXWrKg

  training_log: https://pan.baidu.com/s/1cCKRNZTIxDv3QM9N1atDyA

password: 1234
#### 3.2. Test the Model and Visualize the Segmentation Results
First, change the session name in ```Config.py``` as the training phase. Then run:
```angular2html
python test_model.py
```
You can get the Dice and IoU scores and the visualization results. 


### 4. Reproducibility

In our code, we carefully set the random seed and set cudnn as 'deterministic' mode to eliminate the randomness. However, there still exsist some factors which may cause different training results, e.g., the cuda version, GPU types, the number of GPUs and etc. See https://pytorch.org/docs/stable/notes/randomness.html for more details.


## Reference


* [LViT](https://github.com/HUANGLIZI/LViT)
* [Biformer](https://github.com/rayleizhu/BiFormer)


## Citation

```bash
@article{li2023lvit,
  title={Lvit: language meets vision transformer in medical image segmentation},
  author={Li, Zihan and Li, Yunxiang and Li, Qingde and Wang, Puyang and Guo, Dazhou and Lu, Le and Jin, Dakai and Zhang, You and Hong, Qingqi},
  journal={IEEE Transactions on Medical Imaging},
  year={2023},
  publisher={IEEE}
}
```

Open an issue or mail me directly in case of any queries or suggestions.




