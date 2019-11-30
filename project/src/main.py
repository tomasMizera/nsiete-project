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
train['ImageId'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
train['cat'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
train['has_mask'] = ~pd.isna(train['EncodedPixels'])
train['missing'] = pd.isna(train['EncodedPixels'])
train_nan = train.groupby('ImageId').agg('sum')
train_nan.columns = ['No: of Masks', 'Missing masks']
train_nan['Missing masks'].hist()

mask_count_df = pd.DataFrame(train_nan)
mask_count_df = train.groupby('ImageId').agg(np.sum).reset_index()
mask_count_df.sort_values('has_mask', ascending=False, inplace=True)

BATCH_SIZE = 32

train_imgs, val_imgs = train_test_split(train['Image'].values,
                                        test_size=0.2,
                                        stratify=train['Class'].map(lambda x: str(sorted(list(x)))),
                                        random_state=2019)

img_2_ohe_vector = {img: vec for img, vec in zip(
    train['Image'], train.iloc[:, 2:].values)}

albumentations_train = Compose([
    VerticalFlip(), HorizontalFlip(), Rotate(limit=20), GridDistortion()
], p=1)

data_generator_train = DataGenerator(
    train_imgs, augmentation=albumentations_train, img_2_ohe_vector=img_2_ohe_vector)
data_generator_train_eval = DataGenerator(
    train_imgs, shuffle=False, img_2_ohe_vector=img_2_ohe_vector)
data_generator_val = DataGenerator(
    val_imgs, shuffle=False, img_2_ohe_vector=img_2_ohe_vector)

train_metric_callback = PrAucCallback(data_generator_train_eval)
val_callback = PrAucCallback(data_generator_val, stage='val')

# Unet model
preprocess = get_preprocessing('resnet34')  # for resnet, img = (img-110.0)/1.0
model = Unet('resnet34', input_shape=(256, 384, 3),
             classes=4, activation='sigmoid')

# EfficientNet model 
# def get_model():
#     K.clear_session()
#     base_model = efn.EfficientNetB2(weights='imagenet', include_top=False, pooling='avg', input_shape=(256, 384, 3))
#     x = base_model.output
#     y_pred = Dense(4, activation='sigmoid')(x)
#     return Model(inputs=base_model.input, outputs=y_pred)

# model = get_model()

# for base_layer in model.layers[:-3]:
#     base_layer.trainable = False

model.compile(optimizer=RAdam(warmup_proportion=0.1, min_lr=1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(
    data_generator_train, validation_data=data_generator_val, epochs=20, verbose=3, callbacks=[
        keras.callbacks.TensorBoard(
            log_dir=os.path.join("../logs", str(datetime.now())),
            histogram_freq=1,
            profile_batch=0),
        keras.callbacks.ModelCheckpoint(filepath='../models/model.ckpt', verbose=1),
        train_metric_callback, val_callback
    ])

