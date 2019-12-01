# Cloud segmentation project

### Matej Cief, Tomas Mizera

*Final assignment*

---

GitHub repo: [https://github.com/tomasMizera/nsiete-project](https://github.com/tomasMizera/nsiete-project)

**Content in this document**
1. Motivation
2. Dataset Description
3. Technical Documentation of NN
4. Training Routine
5. Conclusion & Related Work
6. Literature

---

#### 1. Motivation

In this project we are segmenting cloud satellite pictures and recognizing different types of clouds in them. There are 4 major cloud types - Fish, Gravel, Sugar and Flower[2]. The goal of this project is to automatize process of cloud types detection since it can help scientists to build greater environmental models that helps to predict future climate changes.

[1] mentions that:
 *There are many ways in which clouds can organize, but the boundaries between different forms of organization are murky. This makes it challenging to build traditional rule-based algorithms to separate cloud features.* 
Therefore there is a movement trying to classify clouds via Neural Networks.

#### 2. Datasets

Our dataset consists of train and test images downloaded from [Nasa Worldview](https://worldview.earthdata.nasa.gov/). Data was labeled by a team of of 68 scientists. There are 4 label names: Fish, Flower, Gravel, Sugar. And result value is 4 image masks, one for occurrence of each type of cloud. In total we have 5546 images in train dataset and 3698 images in test dataset.

Here is a visualized example from labeled train data:

![labeled cloud types](media/cloud.gif)

Dataset is further analysed in `data_analysis` jupyter notebook.

#### 3. Technical Documentation

In this section we describe used NN architecture and describe some challenges we were facing while working on project.

##### 3.1 Overview

We used 2 neural network architectures with several backbones:
1. Unet
2. EfficientNet [4]

We use Unet for predicting masks based on input images (data analysis can be found in `analysis/data_analysis.ipynb`) with backbone `resnet` that extracts features and passes it to Unet.

##### 3.2 Unet

![Unet for mask prediction](media/unet.png)

Complete implemented model architecture is in first appendix as it is too long, it is an outcome from `model.summary()` method. However these are our params:  
```
Total params: 24,456,589
Trainable params: 24,439,239
Non-trainable params: 17,350
```

**Input**
We had several phases but the one that worked the best takes as an input 4D array of (batch size * resolution_width * resolution_height * n_channels)

**Output**
Output is served as 4D array as well, but this time it looks like this: (batch size * resolution_width * resolution_height * n_cloud_classes). We get mask (it needs to be rounded) as an output for each class on each image.

Presentation is showed in `present.ipynb` notebook.

##### 3.3 Challenges & Solutions

While working on project we came across several challenges:

* run-length encoding (as described in analysis), labeled data provided from kaggle  and had to somehow transfer this encoding to mask image. We found several functions that transforms this encoding to images and used them.
* data streaming to model preventing memory overflow. Generator class was introduced - it also transforms data (e.g. run-length encoding to image and so on.)
* we also spent nice amount of time searching for possibilities to predict mask, not only category of clouds (not 4 output neurons, but entire convolutional layer) 



##### 3.4 Submitted Files

* `analysis/data_analysis.ipynb(.html)` - data analysis, also generated to html for simpler view 

* `main.py` - model definition and training

* `data/generator.py` - code for Generator class handling data manipulation and streaming to model

* `models/util.py`  - does dice coef

#### 4. Training Routine
Our training routine consists of following steps:
* Split data to train and test samples and create respective generators
* Download pretrained model backbone
* Compose own model (either `Unet` or `EfficientNet`)
* Fit model, provide train and validation generator, use callbacks to stop training if plateau

##### 4.1 What did we do?
* We started by implementing generator ([inspired by this notebook](https://www.kaggle.com/shahules/understanding-clounds-with-keras-unet)), which transforms image labels from `train.csv` to stream of images, produces input of dim (32, 256, 384, 3) - 32 is batch size
* Afterwards we split data to train and test and implemented our first model, `Unet` with backbone `resnet34`
* We have achieved around 55% accuracy (we used dice coefficient to measure that)
* Then to further improve our model, we tried to implement PR AUC callback ([source](https://www.kaggle.com/mobassir/keras-efficientnetb2-for-classifying-cloud))
* To do so, we had to remake our generator to provide image labels (y_true), so at the end of each epoch, PR AUC callback calls predict function over train dataset and computes AUC for precision and recall. If this recall stagnates for 5 epochs, it stops to train our model
* With PR AUC callback, we implemented EfficientNet, this time we were able to achieve only 21% accuracy and our model plateaued right at the beginning
* Then we tried to use other backbones, we again tried `Unet` with `densenet169`, but it ate all our RAM (32GB). After unsuccessful attempt with `densenet169`, we tried `inception`. This time we were able to achieve better results, than with `resnet34`. As I write this document, it ran 8 epochs and is 2 percentage points ahead
* We used `Adam` and `RAdam` as our optimizers everywhere



##### 4.2 First Complete Run

Our first big run was with Unet architecture and `resnet34` had 50 epochs and these were the results:

<p><span style="font-style:italic">Legenda: </span><span style="color:blue"><em>Validation data</em> </span> - <span style="color:orange"><em>Train data</em></span></p>


<img src="media/model_acc.png" alt="First run on data"/>

*Model accuracy on 50 epochs*



<img src="media/model_loss.png" alt="First run on data"  />

*Model loss on 50 epochs*

We can see that current run stagnates at 20-25 epochs and after that loss function starts to arise.  It would make sense to stop training after this amount of epochs - this feature will be implemented in final submission.

##### 4.3 Training Results

Hereby we show results for later trainings

|No.| Acc | Epochs | Model |
|---|---|---|---|
|1|0.5|50|Unet|

#### 5. Conclusion

We were able to train NN model for a different cloud types partitioning. Outcoming model gives us mask for each cloud type on every picture we provide, final mask than needs to be rounded. While training data we found out that the Unet architecture gives us the best results out of all the models we tried.

##### 5.1 Related Work

Original idea comes from this [Kaggle competition](https://www.kaggle.com/c/understanding_cloud_organization/overview/description)[1] from *Max Planck Institute for Meteorology*. Competition's goal is to recognize different cloud types on provided test data. There are also included example `jupyter notebooks` with data loading and processing that can helped us get started. 

Further (in proposal) we provided a list of related works that would help us getting started or gain additional information in the weather area:

* Scientific paper *Combining crowd-sourcing and deep learning to understand meso-scale organization of shallow convection*[2],
* Thesis written by Adam Rafajdus[3] that tries to predict weather based on multiple weather factors, including cloud movements.

However, after several discussions we sticked with kaggle competitions and attached jupyter notebooks since we did not predict weather, but simply cloud types.

#### Literature

[1] [Understanding cloud organization, Kaggle competition by Max Planck Institute for Meteorology](https://www.kaggle.com/c/understanding_cloud_organization/overview)

[2] [*Combining crowd-sourcing and deep learning to understand meso-scale organization of shallow convection*, Rasp Stephan et al., 2019](https://arxiv.org/pdf/1906.01906.pdf)

[3] Weather Forecast by Generative Adversarial Networks, Adam Rafajdus, 2018, Thesis at Faculty of Informatics and Information Technologies STU.

[4] EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks; *Mingxing Tan*, *Quoc Le*; Available at: [http://proceedings.mlr.press/v97/tan19a/tan19a.pdf](http://proceedings.mlr.press/v97/tan19a/tan19a.pdf)


#### Appendix A: model architecture  

Outcome from calling `model.summary()` method on builded model  

```
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
data (InputLayer)               (None, 256, 384, 3)  0                                            
__________________________________________________________________________________________________
bn_data (BatchNormalization)    (None, 256, 384, 3)  9           data[0][0]                       
__________________________________________________________________________________________________
zero_padding2d_1 (ZeroPadding2D (None, 262, 390, 3)  0           bn_data[0][0]                    
__________________________________________________________________________________________________
conv0 (Conv2D)                  (None, 128, 192, 64) 9408        zero_padding2d_1[0][0]           
__________________________________________________________________________________________________
bn0 (BatchNormalization)        (None, 128, 192, 64) 256         conv0[0][0]                      
__________________________________________________________________________________________________
relu0 (Activation)              (None, 128, 192, 64) 0           bn0[0][0]                        
__________________________________________________________________________________________________
zero_padding2d_2 (ZeroPadding2D (None, 130, 194, 64) 0           relu0[0][0]                      
__________________________________________________________________________________________________
pooling0 (MaxPooling2D)         (None, 64, 96, 64)   0           zero_padding2d_2[0][0]           
__________________________________________________________________________________________________
stage1_unit1_bn1 (BatchNormaliz (None, 64, 96, 64)   256         pooling0[0][0]                   
__________________________________________________________________________________________________
stage1_unit1_relu1 (Activation) (None, 64, 96, 64)   0           stage1_unit1_bn1[0][0]           
__________________________________________________________________________________________________
zero_padding2d_3 (ZeroPadding2D (None, 66, 98, 64)   0           stage1_unit1_relu1[0][0]         
__________________________________________________________________________________________________
stage1_unit1_conv1 (Conv2D)     (None, 64, 96, 64)   36864       zero_padding2d_3[0][0]           
__________________________________________________________________________________________________
stage1_unit1_bn2 (BatchNormaliz (None, 64, 96, 64)   256         stage1_unit1_conv1[0][0]         
__________________________________________________________________________________________________
stage1_unit1_relu2 (Activation) (None, 64, 96, 64)   0           stage1_unit1_bn2[0][0]           
__________________________________________________________________________________________________
zero_padding2d_4 (ZeroPadding2D (None, 66, 98, 64)   0           stage1_unit1_relu2[0][0]         
__________________________________________________________________________________________________
stage1_unit1_conv2 (Conv2D)     (None, 64, 96, 64)   36864       zero_padding2d_4[0][0]           
__________________________________________________________________________________________________
stage1_unit1_sc (Conv2D)        (None, 64, 96, 64)   4096        stage1_unit1_relu1[0][0]         
__________________________________________________________________________________________________
add_1 (Add)                     (None, 64, 96, 64)   0           stage1_unit1_conv2[0][0]         
                                                                 stage1_unit1_sc[0][0]            
__________________________________________________________________________________________________
stage1_unit2_bn1 (BatchNormaliz (None, 64, 96, 64)   256         add_1[0][0]                      
__________________________________________________________________________________________________
stage1_unit2_relu1 (Activation) (None, 64, 96, 64)   0           stage1_unit2_bn1[0][0]           
__________________________________________________________________________________________________
zero_padding2d_5 (ZeroPadding2D (None, 66, 98, 64)   0           stage1_unit2_relu1[0][0]         
__________________________________________________________________________________________________
stage1_unit2_conv1 (Conv2D)     (None, 64, 96, 64)   36864       zero_padding2d_5[0][0]           
__________________________________________________________________________________________________
stage1_unit2_bn2 (BatchNormaliz (None, 64, 96, 64)   256         stage1_unit2_conv1[0][0]         
__________________________________________________________________________________________________
stage1_unit2_relu2 (Activation) (None, 64, 96, 64)   0           stage1_unit2_bn2[0][0]           
__________________________________________________________________________________________________
zero_padding2d_6 (ZeroPadding2D (None, 66, 98, 64)   0           stage1_unit2_relu2[0][0]         
__________________________________________________________________________________________________
stage1_unit2_conv2 (Conv2D)     (None, 64, 96, 64)   36864       zero_padding2d_6[0][0]           
__________________________________________________________________________________________________
add_2 (Add)                     (None, 64, 96, 64)   0           stage1_unit2_conv2[0][0]         
                                                                 add_1[0][0]                      
__________________________________________________________________________________________________
stage1_unit3_bn1 (BatchNormaliz (None, 64, 96, 64)   256         add_2[0][0]                      
__________________________________________________________________________________________________
stage1_unit3_relu1 (Activation) (None, 64, 96, 64)   0           stage1_unit3_bn1[0][0]           
__________________________________________________________________________________________________
zero_padding2d_7 (ZeroPadding2D (None, 66, 98, 64)   0           stage1_unit3_relu1[0][0]         
__________________________________________________________________________________________________
stage1_unit3_conv1 (Conv2D)     (None, 64, 96, 64)   36864       zero_padding2d_7[0][0]           
__________________________________________________________________________________________________
stage1_unit3_bn2 (BatchNormaliz (None, 64, 96, 64)   256         stage1_unit3_conv1[0][0]         
__________________________________________________________________________________________________
stage1_unit3_relu2 (Activation) (None, 64, 96, 64)   0           stage1_unit3_bn2[0][0]           
__________________________________________________________________________________________________
zero_padding2d_8 (ZeroPadding2D (None, 66, 98, 64)   0           stage1_unit3_relu2[0][0]         
__________________________________________________________________________________________________
stage1_unit3_conv2 (Conv2D)     (None, 64, 96, 64)   36864       zero_padding2d_8[0][0]           
__________________________________________________________________________________________________
add_3 (Add)                     (None, 64, 96, 64)   0           stage1_unit3_conv2[0][0]         
                                                                 add_2[0][0]                      
__________________________________________________________________________________________________
stage2_unit1_bn1 (BatchNormaliz (None, 64, 96, 64)   256         add_3[0][0]                      
__________________________________________________________________________________________________
stage2_unit1_relu1 (Activation) (None, 64, 96, 64)   0           stage2_unit1_bn1[0][0]           
__________________________________________________________________________________________________
zero_padding2d_9 (ZeroPadding2D (None, 66, 98, 64)   0           stage2_unit1_relu1[0][0]         
__________________________________________________________________________________________________
stage2_unit1_conv1 (Conv2D)     (None, 32, 48, 128)  73728       zero_padding2d_9[0][0]           
__________________________________________________________________________________________________
stage2_unit1_bn2 (BatchNormaliz (None, 32, 48, 128)  512         stage2_unit1_conv1[0][0]         
__________________________________________________________________________________________________
stage2_unit1_relu2 (Activation) (None, 32, 48, 128)  0           stage2_unit1_bn2[0][0]           
__________________________________________________________________________________________________
zero_padding2d_10 (ZeroPadding2 (None, 34, 50, 128)  0           stage2_unit1_relu2[0][0]         
__________________________________________________________________________________________________
stage2_unit1_conv2 (Conv2D)     (None, 32, 48, 128)  147456      zero_padding2d_10[0][0]          
__________________________________________________________________________________________________
stage2_unit1_sc (Conv2D)        (None, 32, 48, 128)  8192        stage2_unit1_relu1[0][0]         
__________________________________________________________________________________________________
add_4 (Add)                     (None, 32, 48, 128)  0           stage2_unit1_conv2[0][0]         
                                                                 stage2_unit1_sc[0][0]            
__________________________________________________________________________________________________
stage2_unit2_bn1 (BatchNormaliz (None, 32, 48, 128)  512         add_4[0][0]                      
__________________________________________________________________________________________________
stage2_unit2_relu1 (Activation) (None, 32, 48, 128)  0           stage2_unit2_bn1[0][0]           
__________________________________________________________________________________________________
zero_padding2d_11 (ZeroPadding2 (None, 34, 50, 128)  0           stage2_unit2_relu1[0][0]         
__________________________________________________________________________________________________
stage2_unit2_conv1 (Conv2D)     (None, 32, 48, 128)  147456      zero_padding2d_11[0][0]          
__________________________________________________________________________________________________
stage2_unit2_bn2 (BatchNormaliz (None, 32, 48, 128)  512         stage2_unit2_conv1[0][0]         
__________________________________________________________________________________________________
stage2_unit2_relu2 (Activation) (None, 32, 48, 128)  0           stage2_unit2_bn2[0][0]           
__________________________________________________________________________________________________
zero_padding2d_12 (ZeroPadding2 (None, 34, 50, 128)  0           stage2_unit2_relu2[0][0]         
__________________________________________________________________________________________________
stage2_unit2_conv2 (Conv2D)     (None, 32, 48, 128)  147456      zero_padding2d_12[0][0]          
__________________________________________________________________________________________________
add_5 (Add)                     (None, 32, 48, 128)  0           stage2_unit2_conv2[0][0]         
                                                                 add_4[0][0]                      
__________________________________________________________________________________________________
stage2_unit3_bn1 (BatchNormaliz (None, 32, 48, 128)  512         add_5[0][0]                      
__________________________________________________________________________________________________
stage2_unit3_relu1 (Activation) (None, 32, 48, 128)  0           stage2_unit3_bn1[0][0]           
__________________________________________________________________________________________________
zero_padding2d_13 (ZeroPadding2 (None, 34, 50, 128)  0           stage2_unit3_relu1[0][0]         
__________________________________________________________________________________________________
stage2_unit3_conv1 (Conv2D)     (None, 32, 48, 128)  147456      zero_padding2d_13[0][0]          
__________________________________________________________________________________________________
stage2_unit3_bn2 (BatchNormaliz (None, 32, 48, 128)  512         stage2_unit3_conv1[0][0]         
__________________________________________________________________________________________________
stage2_unit3_relu2 (Activation) (None, 32, 48, 128)  0           stage2_unit3_bn2[0][0]           
__________________________________________________________________________________________________
zero_padding2d_14 (ZeroPadding2 (None, 34, 50, 128)  0           stage2_unit3_relu2[0][0]         
__________________________________________________________________________________________________
stage2_unit3_conv2 (Conv2D)     (None, 32, 48, 128)  147456      zero_padding2d_14[0][0]          
__________________________________________________________________________________________________
add_6 (Add)                     (None, 32, 48, 128)  0           stage2_unit3_conv2[0][0]         
                                                                 add_5[0][0]                      
__________________________________________________________________________________________________
stage2_unit4_bn1 (BatchNormaliz (None, 32, 48, 128)  512         add_6[0][0]                      
__________________________________________________________________________________________________
stage2_unit4_relu1 (Activation) (None, 32, 48, 128)  0           stage2_unit4_bn1[0][0]           
__________________________________________________________________________________________________
zero_padding2d_15 (ZeroPadding2 (None, 34, 50, 128)  0           stage2_unit4_relu1[0][0]         
__________________________________________________________________________________________________
stage2_unit4_conv1 (Conv2D)     (None, 32, 48, 128)  147456      zero_padding2d_15[0][0]          
__________________________________________________________________________________________________
stage2_unit4_bn2 (BatchNormaliz (None, 32, 48, 128)  512         stage2_unit4_conv1[0][0]         
__________________________________________________________________________________________________
stage2_unit4_relu2 (Activation) (None, 32, 48, 128)  0           stage2_unit4_bn2[0][0]           
__________________________________________________________________________________________________
zero_padding2d_16 (ZeroPadding2 (None, 34, 50, 128)  0           stage2_unit4_relu2[0][0]         
__________________________________________________________________________________________________
stage2_unit4_conv2 (Conv2D)     (None, 32, 48, 128)  147456      zero_padding2d_16[0][0]          
__________________________________________________________________________________________________
add_7 (Add)                     (None, 32, 48, 128)  0           stage2_unit4_conv2[0][0]         
                                                                 add_6[0][0]                      
__________________________________________________________________________________________________
stage3_unit1_bn1 (BatchNormaliz (None, 32, 48, 128)  512         add_7[0][0]                      
__________________________________________________________________________________________________
stage3_unit1_relu1 (Activation) (None, 32, 48, 128)  0           stage3_unit1_bn1[0][0]           
__________________________________________________________________________________________________
zero_padding2d_17 (ZeroPadding2 (None, 34, 50, 128)  0           stage3_unit1_relu1[0][0]         
__________________________________________________________________________________________________
stage3_unit1_conv1 (Conv2D)     (None, 16, 24, 256)  294912      zero_padding2d_17[0][0]          
__________________________________________________________________________________________________
stage3_unit1_bn2 (BatchNormaliz (None, 16, 24, 256)  1024        stage3_unit1_conv1[0][0]         
__________________________________________________________________________________________________
stage3_unit1_relu2 (Activation) (None, 16, 24, 256)  0           stage3_unit1_bn2[0][0]           
__________________________________________________________________________________________________
zero_padding2d_18 (ZeroPadding2 (None, 18, 26, 256)  0           stage3_unit1_relu2[0][0]         
__________________________________________________________________________________________________
stage3_unit1_conv2 (Conv2D)     (None, 16, 24, 256)  589824      zero_padding2d_18[0][0]          
__________________________________________________________________________________________________
stage3_unit1_sc (Conv2D)        (None, 16, 24, 256)  32768       stage3_unit1_relu1[0][0]         
__________________________________________________________________________________________________
add_8 (Add)                     (None, 16, 24, 256)  0           stage3_unit1_conv2[0][0]         
                                                                 stage3_unit1_sc[0][0]            
__________________________________________________________________________________________________
stage3_unit2_bn1 (BatchNormaliz (None, 16, 24, 256)  1024        add_8[0][0]                      
__________________________________________________________________________________________________
stage3_unit2_relu1 (Activation) (None, 16, 24, 256)  0           stage3_unit2_bn1[0][0]           
__________________________________________________________________________________________________
zero_padding2d_19 (ZeroPadding2 (None, 18, 26, 256)  0           stage3_unit2_relu1[0][0]         
__________________________________________________________________________________________________
stage3_unit2_conv1 (Conv2D)     (None, 16, 24, 256)  589824      zero_padding2d_19[0][0]          
__________________________________________________________________________________________________
stage3_unit2_bn2 (BatchNormaliz (None, 16, 24, 256)  1024        stage3_unit2_conv1[0][0]         
__________________________________________________________________________________________________
stage3_unit2_relu2 (Activation) (None, 16, 24, 256)  0           stage3_unit2_bn2[0][0]           
__________________________________________________________________________________________________
zero_padding2d_20 (ZeroPadding2 (None, 18, 26, 256)  0           stage3_unit2_relu2[0][0]         
__________________________________________________________________________________________________
stage3_unit2_conv2 (Conv2D)     (None, 16, 24, 256)  589824      zero_padding2d_20[0][0]          
__________________________________________________________________________________________________
add_9 (Add)                     (None, 16, 24, 256)  0           stage3_unit2_conv2[0][0]         
                                                                 add_8[0][0]                      
__________________________________________________________________________________________________
stage3_unit3_bn1 (BatchNormaliz (None, 16, 24, 256)  1024        add_9[0][0]                      
__________________________________________________________________________________________________
stage3_unit3_relu1 (Activation) (None, 16, 24, 256)  0           stage3_unit3_bn1[0][0]           
__________________________________________________________________________________________________
zero_padding2d_21 (ZeroPadding2 (None, 18, 26, 256)  0           stage3_unit3_relu1[0][0]         
__________________________________________________________________________________________________
stage3_unit3_conv1 (Conv2D)     (None, 16, 24, 256)  589824      zero_padding2d_21[0][0]          
__________________________________________________________________________________________________
stage3_unit3_bn2 (BatchNormaliz (None, 16, 24, 256)  1024        stage3_unit3_conv1[0][0]         
__________________________________________________________________________________________________
stage3_unit3_relu2 (Activation) (None, 16, 24, 256)  0           stage3_unit3_bn2[0][0]           
__________________________________________________________________________________________________
zero_padding2d_22 (ZeroPadding2 (None, 18, 26, 256)  0           stage3_unit3_relu2[0][0]         
__________________________________________________________________________________________________
stage3_unit3_conv2 (Conv2D)     (None, 16, 24, 256)  589824      zero_padding2d_22[0][0]          
__________________________________________________________________________________________________
add_10 (Add)                    (None, 16, 24, 256)  0           stage3_unit3_conv2[0][0]         
                                                                 add_9[0][0]                      
__________________________________________________________________________________________________
stage3_unit4_bn1 (BatchNormaliz (None, 16, 24, 256)  1024        add_10[0][0]                     
__________________________________________________________________________________________________
stage3_unit4_relu1 (Activation) (None, 16, 24, 256)  0           stage3_unit4_bn1[0][0]           
__________________________________________________________________________________________________
zero_padding2d_23 (ZeroPadding2 (None, 18, 26, 256)  0           stage3_unit4_relu1[0][0]         
__________________________________________________________________________________________________
stage3_unit4_conv1 (Conv2D)     (None, 16, 24, 256)  589824      zero_padding2d_23[0][0]          
__________________________________________________________________________________________________
stage3_unit4_bn2 (BatchNormaliz (None, 16, 24, 256)  1024        stage3_unit4_conv1[0][0]         
__________________________________________________________________________________________________
stage3_unit4_relu2 (Activation) (None, 16, 24, 256)  0           stage3_unit4_bn2[0][0]           
__________________________________________________________________________________________________
zero_padding2d_24 (ZeroPadding2 (None, 18, 26, 256)  0           stage3_unit4_relu2[0][0]         
__________________________________________________________________________________________________
stage3_unit4_conv2 (Conv2D)     (None, 16, 24, 256)  589824      zero_padding2d_24[0][0]          
__________________________________________________________________________________________________
add_11 (Add)                    (None, 16, 24, 256)  0           stage3_unit4_conv2[0][0]         
                                                                 add_10[0][0]                     
__________________________________________________________________________________________________
stage3_unit5_bn1 (BatchNormaliz (None, 16, 24, 256)  1024        add_11[0][0]                     
__________________________________________________________________________________________________
stage3_unit5_relu1 (Activation) (None, 16, 24, 256)  0           stage3_unit5_bn1[0][0]           
__________________________________________________________________________________________________
zero_padding2d_25 (ZeroPadding2 (None, 18, 26, 256)  0           stage3_unit5_relu1[0][0]         
__________________________________________________________________________________________________
stage3_unit5_conv1 (Conv2D)     (None, 16, 24, 256)  589824      zero_padding2d_25[0][0]          
__________________________________________________________________________________________________
stage3_unit5_bn2 (BatchNormaliz (None, 16, 24, 256)  1024        stage3_unit5_conv1[0][0]         
__________________________________________________________________________________________________
stage3_unit5_relu2 (Activation) (None, 16, 24, 256)  0           stage3_unit5_bn2[0][0]           
__________________________________________________________________________________________________
zero_padding2d_26 (ZeroPadding2 (None, 18, 26, 256)  0           stage3_unit5_relu2[0][0]         
__________________________________________________________________________________________________
stage3_unit5_conv2 (Conv2D)     (None, 16, 24, 256)  589824      zero_padding2d_26[0][0]          
__________________________________________________________________________________________________
add_12 (Add)                    (None, 16, 24, 256)  0           stage3_unit5_conv2[0][0]         
                                                                 add_11[0][0]                     
__________________________________________________________________________________________________
stage3_unit6_bn1 (BatchNormaliz (None, 16, 24, 256)  1024        add_12[0][0]                     
__________________________________________________________________________________________________
stage3_unit6_relu1 (Activation) (None, 16, 24, 256)  0           stage3_unit6_bn1[0][0]           
__________________________________________________________________________________________________
zero_padding2d_27 (ZeroPadding2 (None, 18, 26, 256)  0           stage3_unit6_relu1[0][0]         
__________________________________________________________________________________________________
stage3_unit6_conv1 (Conv2D)     (None, 16, 24, 256)  589824      zero_padding2d_27[0][0]          
__________________________________________________________________________________________________
stage3_unit6_bn2 (BatchNormaliz (None, 16, 24, 256)  1024        stage3_unit6_conv1[0][0]         
__________________________________________________________________________________________________
stage3_unit6_relu2 (Activation) (None, 16, 24, 256)  0           stage3_unit6_bn2[0][0]           
__________________________________________________________________________________________________
zero_padding2d_28 (ZeroPadding2 (None, 18, 26, 256)  0           stage3_unit6_relu2[0][0]         
__________________________________________________________________________________________________
stage3_unit6_conv2 (Conv2D)     (None, 16, 24, 256)  589824      zero_padding2d_28[0][0]          
__________________________________________________________________________________________________
add_13 (Add)                    (None, 16, 24, 256)  0           stage3_unit6_conv2[0][0]         
                                                                 add_12[0][0]                     
__________________________________________________________________________________________________
stage4_unit1_bn1 (BatchNormaliz (None, 16, 24, 256)  1024        add_13[0][0]                     
__________________________________________________________________________________________________
stage4_unit1_relu1 (Activation) (None, 16, 24, 256)  0           stage4_unit1_bn1[0][0]           
__________________________________________________________________________________________________
zero_padding2d_29 (ZeroPadding2 (None, 18, 26, 256)  0           stage4_unit1_relu1[0][0]         
__________________________________________________________________________________________________
stage4_unit1_conv1 (Conv2D)     (None, 8, 12, 512)   1179648     zero_padding2d_29[0][0]          
__________________________________________________________________________________________________
stage4_unit1_bn2 (BatchNormaliz (None, 8, 12, 512)   2048        stage4_unit1_conv1[0][0]         
__________________________________________________________________________________________________
stage4_unit1_relu2 (Activation) (None, 8, 12, 512)   0           stage4_unit1_bn2[0][0]           
__________________________________________________________________________________________________
zero_padding2d_30 (ZeroPadding2 (None, 10, 14, 512)  0           stage4_unit1_relu2[0][0]         
__________________________________________________________________________________________________
stage4_unit1_conv2 (Conv2D)     (None, 8, 12, 512)   2359296     zero_padding2d_30[0][0]          
__________________________________________________________________________________________________
stage4_unit1_sc (Conv2D)        (None, 8, 12, 512)   131072      stage4_unit1_relu1[0][0]         
__________________________________________________________________________________________________
add_14 (Add)                    (None, 8, 12, 512)   0           stage4_unit1_conv2[0][0]         
                                                                 stage4_unit1_sc[0][0]            
__________________________________________________________________________________________________
stage4_unit2_bn1 (BatchNormaliz (None, 8, 12, 512)   2048        add_14[0][0]                     
__________________________________________________________________________________________________
stage4_unit2_relu1 (Activation) (None, 8, 12, 512)   0           stage4_unit2_bn1[0][0]           
__________________________________________________________________________________________________
zero_padding2d_31 (ZeroPadding2 (None, 10, 14, 512)  0           stage4_unit2_relu1[0][0]         
__________________________________________________________________________________________________
stage4_unit2_conv1 (Conv2D)     (None, 8, 12, 512)   2359296     zero_padding2d_31[0][0]          
__________________________________________________________________________________________________
stage4_unit2_bn2 (BatchNormaliz (None, 8, 12, 512)   2048        stage4_unit2_conv1[0][0]         
__________________________________________________________________________________________________
stage4_unit2_relu2 (Activation) (None, 8, 12, 512)   0           stage4_unit2_bn2[0][0]           
__________________________________________________________________________________________________
zero_padding2d_32 (ZeroPadding2 (None, 10, 14, 512)  0           stage4_unit2_relu2[0][0]         
__________________________________________________________________________________________________
stage4_unit2_conv2 (Conv2D)     (None, 8, 12, 512)   2359296     zero_padding2d_32[0][0]          
__________________________________________________________________________________________________
add_15 (Add)                    (None, 8, 12, 512)   0           stage4_unit2_conv2[0][0]         
                                                                 add_14[0][0]                     
__________________________________________________________________________________________________
stage4_unit3_bn1 (BatchNormaliz (None, 8, 12, 512)   2048        add_15[0][0]                     
__________________________________________________________________________________________________
stage4_unit3_relu1 (Activation) (None, 8, 12, 512)   0           stage4_unit3_bn1[0][0]           
__________________________________________________________________________________________________
zero_padding2d_33 (ZeroPadding2 (None, 10, 14, 512)  0           stage4_unit3_relu1[0][0]         
__________________________________________________________________________________________________
stage4_unit3_conv1 (Conv2D)     (None, 8, 12, 512)   2359296     zero_padding2d_33[0][0]          
__________________________________________________________________________________________________
stage4_unit3_bn2 (BatchNormaliz (None, 8, 12, 512)   2048        stage4_unit3_conv1[0][0]         
__________________________________________________________________________________________________
stage4_unit3_relu2 (Activation) (None, 8, 12, 512)   0           stage4_unit3_bn2[0][0]           
__________________________________________________________________________________________________
zero_padding2d_34 (ZeroPadding2 (None, 10, 14, 512)  0           stage4_unit3_relu2[0][0]         
__________________________________________________________________________________________________
stage4_unit3_conv2 (Conv2D)     (None, 8, 12, 512)   2359296     zero_padding2d_34[0][0]          
__________________________________________________________________________________________________
add_16 (Add)                    (None, 8, 12, 512)   0           stage4_unit3_conv2[0][0]         
                                                                 add_15[0][0]                     
__________________________________________________________________________________________________
bn1 (BatchNormalization)        (None, 8, 12, 512)   2048        add_16[0][0]                     
__________________________________________________________________________________________________
relu1 (Activation)              (None, 8, 12, 512)   0           bn1[0][0]                        
__________________________________________________________________________________________________
decoder_stage0_upsampling (UpSa (None, 16, 24, 512)  0           relu1[0][0]                      
__________________________________________________________________________________________________
decoder_stage0_concat (Concaten (None, 16, 24, 768)  0           decoder_stage0_upsampling[0][0]  
                                                                 stage4_unit1_relu1[0][0]         
__________________________________________________________________________________________________
decoder_stage0a_conv (Conv2D)   (None, 16, 24, 256)  1769472     decoder_stage0_concat[0][0]      
__________________________________________________________________________________________________
decoder_stage0a_bn (BatchNormal (None, 16, 24, 256)  1024        decoder_stage0a_conv[0][0]       
__________________________________________________________________________________________________
decoder_stage0a_relu (Activatio (None, 16, 24, 256)  0           decoder_stage0a_bn[0][0]         
__________________________________________________________________________________________________
decoder_stage0b_conv (Conv2D)   (None, 16, 24, 256)  589824      decoder_stage0a_relu[0][0]       
__________________________________________________________________________________________________
decoder_stage0b_bn (BatchNormal (None, 16, 24, 256)  1024        decoder_stage0b_conv[0][0]       
__________________________________________________________________________________________________
decoder_stage0b_relu (Activatio (None, 16, 24, 256)  0           decoder_stage0b_bn[0][0]         
__________________________________________________________________________________________________
decoder_stage1_upsampling (UpSa (None, 32, 48, 256)  0           decoder_stage0b_relu[0][0]       
__________________________________________________________________________________________________
decoder_stage1_concat (Concaten (None, 32, 48, 384)  0           decoder_stage1_upsampling[0][0]  
                                                                 stage3_unit1_relu1[0][0]         
__________________________________________________________________________________________________
decoder_stage1a_conv (Conv2D)   (None, 32, 48, 128)  442368      decoder_stage1_concat[0][0]      
__________________________________________________________________________________________________
decoder_stage1a_bn (BatchNormal (None, 32, 48, 128)  512         decoder_stage1a_conv[0][0]       
__________________________________________________________________________________________________
decoder_stage1a_relu (Activatio (None, 32, 48, 128)  0           decoder_stage1a_bn[0][0]         
__________________________________________________________________________________________________
decoder_stage1b_conv (Conv2D)   (None, 32, 48, 128)  147456      decoder_stage1a_relu[0][0]       
__________________________________________________________________________________________________
decoder_stage1b_bn (BatchNormal (None, 32, 48, 128)  512         decoder_stage1b_conv[0][0]       
__________________________________________________________________________________________________
decoder_stage1b_relu (Activatio (None, 32, 48, 128)  0           decoder_stage1b_bn[0][0]         
__________________________________________________________________________________________________
decoder_stage2_upsampling (UpSa (None, 64, 96, 128)  0           decoder_stage1b_relu[0][0]       
__________________________________________________________________________________________________
decoder_stage2_concat (Concaten (None, 64, 96, 192)  0           decoder_stage2_upsampling[0][0]  
                                                                 stage2_unit1_relu1[0][0]         
__________________________________________________________________________________________________
decoder_stage2a_conv (Conv2D)   (None, 64, 96, 64)   110592      decoder_stage2_concat[0][0]      
__________________________________________________________________________________________________
decoder_stage2a_bn (BatchNormal (None, 64, 96, 64)   256         decoder_stage2a_conv[0][0]       
__________________________________________________________________________________________________
decoder_stage2a_relu (Activatio (None, 64, 96, 64)   0           decoder_stage2a_bn[0][0]         
__________________________________________________________________________________________________
decoder_stage2b_conv (Conv2D)   (None, 64, 96, 64)   36864       decoder_stage2a_relu[0][0]       
__________________________________________________________________________________________________
decoder_stage2b_bn (BatchNormal (None, 64, 96, 64)   256         decoder_stage2b_conv[0][0]       
__________________________________________________________________________________________________
decoder_stage2b_relu (Activatio (None, 64, 96, 64)   0           decoder_stage2b_bn[0][0]         
__________________________________________________________________________________________________
decoder_stage3_upsampling (UpSa (None, 128, 192, 64) 0           decoder_stage2b_relu[0][0]       
__________________________________________________________________________________________________
decoder_stage3_concat (Concaten (None, 128, 192, 128 0           decoder_stage3_upsampling[0][0]  
                                                                 relu0[0][0]                      
__________________________________________________________________________________________________
decoder_stage3a_conv (Conv2D)   (None, 128, 192, 32) 36864       decoder_stage3_concat[0][0]      
__________________________________________________________________________________________________
decoder_stage3a_bn (BatchNormal (None, 128, 192, 32) 128         decoder_stage3a_conv[0][0]       
__________________________________________________________________________________________________
decoder_stage3a_relu (Activatio (None, 128, 192, 32) 0           decoder_stage3a_bn[0][0]         
__________________________________________________________________________________________________
decoder_stage3b_conv (Conv2D)   (None, 128, 192, 32) 9216        decoder_stage3a_relu[0][0]       
__________________________________________________________________________________________________
decoder_stage3b_bn (BatchNormal (None, 128, 192, 32) 128         decoder_stage3b_conv[0][0]       
__________________________________________________________________________________________________
decoder_stage3b_relu (Activatio (None, 128, 192, 32) 0           decoder_stage3b_bn[0][0]         
__________________________________________________________________________________________________
decoder_stage4_upsampling (UpSa (None, 256, 384, 32) 0           decoder_stage3b_relu[0][0]       
__________________________________________________________________________________________________
decoder_stage4a_conv (Conv2D)   (None, 256, 384, 16) 4608        decoder_stage4_upsampling[0][0]  
__________________________________________________________________________________________________
decoder_stage4a_bn (BatchNormal (None, 256, 384, 16) 64          decoder_stage4a_conv[0][0]       
__________________________________________________________________________________________________
decoder_stage4a_relu (Activatio (None, 256, 384, 16) 0           decoder_stage4a_bn[0][0]         
__________________________________________________________________________________________________
decoder_stage4b_conv (Conv2D)   (None, 256, 384, 16) 2304        decoder_stage4a_relu[0][0]       
__________________________________________________________________________________________________
decoder_stage4b_bn (BatchNormal (None, 256, 384, 16) 64          decoder_stage4b_conv[0][0]       
__________________________________________________________________________________________________
decoder_stage4b_relu (Activatio (None, 256, 384, 16) 0           decoder_stage4b_bn[0][0]         
__________________________________________________________________________________________________
final_conv (Conv2D)             (None, 256, 384, 4)  580         decoder_stage4b_relu[0][0]       
__________________________________________________________________________________________________
sigmoid (Activation)            (None, 256, 384, 4)  0           final_conv[0][0]                 
==================================================================================================
Total params: 24,456,589
Trainable params: 24,439,239
Non-trainable params: 17,350
__________________________________________________________________________________________________
```