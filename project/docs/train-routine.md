# Training routine
Our training routine consists of following steps:
* Split data to train and test samples and create respective generators
* Download pretrained model backbone
* Compose own model (either `Unet` or `EfficientNet`)
* Fit model, provide train and validation generator, use callbacks to stop training if plateau

# What did we do?
* We started by implementing generator ([inspired by this notebook](https://www.kaggle.com/shahules/understanding-clounds-with-keras-unet)), which transforms image labels from `train.csv` to stream of images, produces input of dim (32, 256, 384, 3) - 32 is batch size
* Afterwards we split data to train and test and implemented our first model, `Unet` with backbone `resnet34`
* We have achieved around 55% accuracy (we used dice coefficient to measure that)
* Then to further improve our model, we tried to implement PR AUC callback ([source](https://www.kaggle.com/mobassir/keras-efficientnetb2-for-classifying-cloud))
* To do so, we had to remake our generator to provide image labels (y_true), so at the end of each epoch, PR AUC callback calls predict function over train dataset and computes AUC for precision and recall. If this recall stagnates for 5 epochs, it stops to train our model
* With PR AUC callback, we implemented EfficientNet, this time we were able to achieve only 21% accuracy and our model plateaued right at the beginning
* Then we tried to use other backbones, we again tried `Unet` with `densenet169`, but it ate all our RAM (32GB). After unsuccessful attempt with `densenet169`, we tried `inception`. This time we were able to achieve better results, than with `resnet34`. As I write this document, it ran 8 epochs and is 2 percentage points ahead
* We used `Adam` and `RAdam` as our optimizers everywhere
