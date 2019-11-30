# Cloud segmentation project

### Matej Cief, Tomas Mizera

*Final assignment*

---

GitHub repo: [https://github.com/tomasMizera/nsiete-project](https://github.com/tomasMizera/nsiete-project)

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

#### 3. Technical documentation

**Overview**

We used 2 neural network architectures with several backbones:
1. EfficientNet [4]
2. Unet

We use Unet for predicting masks based on input images (data analysis can be found in `analysis/data_analysis.ipynb`) with backbone `resnet` that extracts features and passes it to Unet.

*EfficientNet*

*Unet*

![Unet for mask prediction](media/unet.png)

**Challenges & solutions**

While working on project we came across several challenges:

* run-length encoding (as described in analysis), labeled data provided from kaggle  and had to somehow transfer this encoding to mask image. We found several functions that transforms this encoding to images and used them.
* data streaming to model preventing memory overflow. Generator class was introduced - it also transforms data (e.g. run-length encoding to image and so on.)
* we also spent nice amount of time searching for possibilities to predict mask, not only category of clouds (not 4 output neurons, but entire convolutional layer) 



**Files** TODO rewrite

* `analysis/data_analysis.ipynb(.html)` - data analysis, also generated to html for simpler view 

* `main.py` - model definition and training

* `data/generator.py` - code for Generator class handling data manipulation and streaming to model

* `models/util.py`  - does dice coef

**First complete run**

Our first big run was with Unet architecture and `resnet34` had 50 epochs and these were the results:

<p><span style="font-style:italic">Legenda: </span><span style="color:blue"><em>Validation data</em> </span> - <span style="color:orange"><em>Train data</em></span></p>


<img src="media/model_acc.png" alt="First run on data"/>

*Model accuracy on 50 epochs*



<img src="media/model_loss.png" alt="First run on data"  />

*Model loss on 50 epochs*

We can see that current run stagnates at 20-25 epochs and after that loss function starts to arise.  It would make sense to stop training after this amount of epochs - this feature will be implemented in final submission.


#### 4. Related Work

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