import pandas as pd
import numpy as np
from datetime import datetime
import keras
import cv2
import os
from sklearn.model_selection import train_test_split
from segmentation_models import Unet
from segmentation_models import get_preprocessing
from data.generator import DataGenerator
from models.util import dice_coef

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

train_idx, val_idx = train_test_split(
    mask_count_df.index, random_state=2019, test_size=0.15
)

train_generator = DataGenerator(
    train_idx,
    df=mask_count_df,
    target_df=train,
    batch_size=BATCH_SIZE,
    reshape=(256, 384),
    n_channels=3,
    n_classes=4
)

val_generator = DataGenerator(
    val_idx,
    df=mask_count_df,
    target_df=train,
    batch_size=BATCH_SIZE,
    reshape=(256, 384),
    n_channels=3,
    n_classes=4
)

preprocess = get_preprocessing('resnet34')  # for resnet, img = (img-110.0)/1.0
model = Unet('resnet34', input_shape=(256, 384, 3),
             classes=4, activation='sigmoid')
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=[dice_coef])

model.fit_generator(
    train_generator, validation_data=val_generator, epochs=1, verbose=3, callbacks=[
        keras.callbacks.TensorBoard(
            log_dir=os.path.join("../logs", str(datetime.now())),
            histogram_freq=1,
            profile_batch=0)
    ])

