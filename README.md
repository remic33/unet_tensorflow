# 
The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
and @zhixuhao [work](https://github.com/zhixuhao/unet/)
---


## How to use 
### Installation
This new repo require [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to run.  
Use 
```bash
conda env create -f unet.yml
 ``` 
to create env.
Then : 
```bash
conda activate unet
```
You can now run the code using 
```bash 
python main.py
```
and the following options. 

### Python update
I started this project using python 3.5 and old keras implementation.  
Because python 3.5 is deprecated, epochs were a bit slow on CPU, and I needed new layers from keras when padding is on VALID option. 
I took the liberty to update the code to tensorflow 2.6 and python 3.9. 

### Clean code 
Code was formatted using black & lint using pylint.
I choose [abseil.io](https://abseil.io) for argument interpretation because it gives fast implementation, easy logs, as well as type/value input check. 
### Modes 
You can choose mode at launch by using --mode command. You can choose either train from nothing with train, transfer with transfer mode, or predict mode.
Predict mode can take any architecture and as an input model.
Full mode trains the model from the ground then predicts on the dataset. 

### Padding 
You can choose mode at launch by using --padding command. Possible paddings are "same" and "valid".

### Pooling operations
You can choose the number of pooling operation for both standard & resnet blocks. 
Use --pools 3 to get 3 pooling operations. 

### Number of layers 
You can choose the number of layers between each pooling operation for both standard & resnet blocks. 
use -lbp 2 to get two layers between each pooling. 
Layers in right part of the model will be the same. 
Resnet can only use even number of layers.

### Epochs 
Number of training epochs. 
Use --epochs 1 to train on only one epoch

### Model path
To train & save or to load a model you need to specify a path. 
Use --path data/model.hdf5 to choose path. 


### Blocks 
You can choose mode at launch by using --block command. 
Options are "standard" for standard blocks and "resnet" for... resnet blocks.
Standard model can be use with or without batch normalisation. 
### Usage 
Standard model : --block standard
Standard model no bn : --block standard_no_bn
Resnet model : --block resnet
Inception model : --block inception
After checking several publications, blocks are only used in the contracting part of Unet. Expanding parts are using regular convolution layers.

## Why not use blocks on the upsampling part of Unet? 
I choose to not use block in the right part of the model because concatenation from left to right part already skip connections. Adding resnet blocks should decrease training speed and increase material needs for little to no benefits.
## The shaping problem in resnet 
With resnet, we can have trouble with Add operation when input layers came from upsampling or pooling operation, because shapes are different (number of filters from last convolution is the cause)
I choose not to count that operation as an additional convolution because it is a necessary operation. 
## Inception inspired block
As part of optionnal request, I created an inception inspired unet. Contracting part of the network is the same as before, but blocks are inception inspired. Mode to use it is "inception"
