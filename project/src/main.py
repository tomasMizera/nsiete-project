from albumentations import Compose, VerticalFlip, HorizontalFlip, Rotate, GridDistortion
import pandas as pd
import numpy as np
from datetime import datetime
import keras
import cv2
import os
import efficientnet.keras as efn
from keras_radam import RAdam
from sklearn.model_selection import train_test_split
from segmentation_models import Unet
from segmentation_models import get_preprocessing
import keras.backend as K
from keras.layers import Dense, Flatten
from keras.models import Model
from data.generator import DataGenerator
from models.util import dice_coef
from models.util import PrAucCallback

train = pd.read_csv('../data/train.csv')
train = train[~train['EncodedPixels'].isnull()]
train['Image'] = train['Image_Label'].map(lambda x: x.split('_')[0])
train['Class'] = train['Image_Label'].map(lambda x: x.split('_')[1])
classes = train['Class'].unique()
train = train.groupby('Image')['Class'].agg(set).reset_index()
for class_name in classes:
    train[class_name] = train['Class'].map(
        lambda x: 1 if class_name in x else 0)

BATCH_SIZE = 32

train_imgs, val_imgs = train_test_split(train['Image'].values,
                                        test_size=0.2,
                                        stratify=train['Class'].map(lambda x: str(sorted(list(x)))),
                                        random_state=2019)

img_2_ohe_vector = {img: vec for img, vec in zip(train['Image'], train.iloc[:, 2:].values)}

albumentations_train = Compose([
    VerticalFlip(), HorizontalFlip(), Rotate(limit=20), GridDistortion()
], p=1)

data_generator_train = DataGenerator(train_imgs, augmentation=albumentations_train, img_2_ohe_vector=img_2_ohe_vector)
data_generator_train_eval = DataGenerator(train_imgs, shuffle=False, img_2_ohe_vector=img_2_ohe_vector)
data_generator_val = DataGenerator(val_imgs, shuffle=False, img_2_ohe_vector=img_2_ohe_vector)

train_metric_callback = PrAucCallback(data_generator_train_eval)
val_callback = PrAucCallback(data_generator_val, stage='val')

# Unet model
# preprocess = get_preprocessing('resnet34')  # for resnet, img = (img-110.0)/1.0
# model = Unet('resnet34', input_shape=(256, 256, 3),
#              classes=4, activation='sigmoid')

# EfficientNet model 
def get_model():
    K.clear_session()
    base_model = efn.EfficientNetB2(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 256, 3))
    x = base_model.output
    y_pred = Dense(4, activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=y_pred)

model = get_model()

model.compile(optimizer=RAdam(warmup_proportion=0.1, min_lr=1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(
    data_generator_train, validation_data=data_generator_val, epochs=20, verbose=3, callbacks=[
        keras.callbacks.TensorBoard(
            log_dir=os.path.join("../logs", str(datetime.now())),
            histogram_freq=1,
            profile_batch=0),
        keras.callbacks.ModelCheckpoint(filepath='../models/efficient_net.ckpt', verbose=1),
        train_metric_callback, val_callback
    ])

