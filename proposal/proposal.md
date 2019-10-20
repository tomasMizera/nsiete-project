# Neural Networks  

## Project Proposal - ***Cloud type segmentation CNN***

**Matej Čief, Tomáš Mizera**
*Seminary Monday 12:00 - Matúš Pikuliak, Ing.*

GitHub repo: [https://github.com/tomasMizera/nsiete-project](https://github.com/tomasMizera/nsiete-project)

#### 1. Motivation

In this project we will try to segment cloud satellite pictures and recognize different types of clouds in them. There are 4 major cloud types - Fish, Gravel, Sugar and Flower[2]. The goal of this project is to  automatize process of cloud types detection since it can help scientists to build greater environmental models that helps to predict future climate changes.

[1] mentions that:
 *There are many ways in which clouds can organize, but the boundaries between different forms of organization are murky. This makes it challenging to build traditional rule-based algorithms to separate cloud features.* 
Therefore there is a movement trying to classify clouds via Neural Networks.

#### 2. Related Work

Original idea comes from this [Kaggle competition](https://www.kaggle.com/c/understanding_cloud_organization/overview/description)[1] (*still opened*) where *Max Planck Institute for Meteorology*. Competition's goal is to recognize different cloud types on provided test data. There are also included example `jupyter notebooks` with data loading and processing that can help us get started. Moreover, there are several notebooks that  

Further we provide a list of related works that could help us getting started or gain additional information in the weather area:

* Scientific paper *Combining crowd-sourcing and deep learning to understand meso-scale organization of shallow convection*[2],
* Thesis written by Adam Rafajdus[3] that tries to predict weather based on multiple weather factors, including cloud movements.

#### 3. Datasets

Our dataset consists of train and test images downloaded from [Nasa Worldview](https://worldview.earthdata.nasa.gov/). Data was labeled by a team of of 68 scientists. There are 4 label names: Fish, Flower, Gravel, Sugar. And result value is 4 image masks, one for occurrence of each type of cloud. In total we have 5546 images in train dataset and 3698 images in test dataset.

Here is a visualized example from labeled train data:

![labeled cloud types](./media/cloud.gif)

#### 4. High-Level Solution Proposal

As far as we are currently concerned, we will use Mask R-CNN networks with a specific architecture that will be chosen based on discussions. Most probably we will use one of those architectures: *VGG, Inception, ResNet or Dense* Net.



#### Literature

[1] [Understanding cloud organization, Kaggle competition by Max Planck Institute for Meteorology](https://www.kaggle.com/c/understanding_cloud_organization/overview)

[2] [*Combining crowd-sourcing and deep learning to understand meso-scale organization of shallow convection*, Rasp Stephan et al., 2019](https://arxiv.org/pdf/1906.01906.pdf)

[3] Weather Forecast by Generative Adversarial Networks, Adam Rafajdus, 2018, Thesis at Faculty of Informatics and Information Technologies STU.