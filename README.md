import numpy as np
from tensorflow.keras.layers import Activation, Conv1D, GlobalMaxPooling1D, Dense, Multiply, Add
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from tensorflow.keras.layers import Conv2D, GlobalMaxPooling2D, Reshape, GlobalAveragePooling1D, Concatenate, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow.keras.backend as K
import tensorflow as tf
from sklearn.metrics import classification_report
import os
import pandas as pd
import pywt
import cv2
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
def ECGnet():
    def block3(x, K, filter_shape):
        y = BatchNormalization()(x)
        y = Activation("relu")(y)
        y = Conv1D(K, 1, strides=2, padding="same")(y)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv1D(K, filter_shape, strides=2, padding="same")(x)
        for i in range(2):
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv1D(K, filter_shape, strides=1, padding="same")(x)
        z = GlobalMaxPooling1D()(x)
        z = Dense(4, activation="relu")(z)
        z = Dense(32, activation="hard_sigmoid")(z)
        x = Multiply()([x, z])
        y = Add()([x, y])
        for i in range(3):
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv1D(K, filter_shape, strides=1, padding="same")(x)
        z = GlobalMaxPooling1D()(x)
        z = Dense(4, activation="relu")(z)
        z = Dense(32, activation="hard_sigmoid")(z)
        x = Multiply()([x, z])
        return Add()([x, y])

    def block2(x, K, filter_shape):
        y = x
        for i in range(3):
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv2D(K, filter_shape, padding="same")(x)
        a = x
        x = GlobalMaxPooling2D()(x)
        x = Dense(8, activation="relu")(x)
        x = Dense(32, activation="hard_sigmoid")(x)
        x = Multiply()([a, x])
        return Add()([x, y])

    def block1(x, K, filter_shape):
        # SE ResNet
        y = Conv2D(K, (1, 1), strides=(2, 1))(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(K, filter_shape, strides=(2, 1), padding="same")(x)
        x1 = GlobalMaxPooling2D()(x)
        x1 = Dense(8, activation="relu")(x1)
        x1 = Dense(32, activation="hard_sigmoid")(x1)
        x1 = Multiply()([x, x1])
        return Add()([x1, y])

    def scale(x, filter_shape1, filter_shape2):
        for i in range(3):
            x = block2(x, 32, filter_shape1)
        x = Reshape([x.shape[1], x.shape[2]*x.shape[3]])(x)
        print(x)
        for i in range(4):
            x = block3(x, 32, filter_shape2)
        x = GlobalAveragePooling1D()(x)
        return x

    input = Input(shape=(1000, 12, 1))
    x = Conv2D(32, (50, 1), strides=(2, 1))(input)
    x = block1(x, 32, (15, 1))
    # x = block1(x, 32, (15, 1))
    # x = block1(x, 32, (15, 1))
    a = scale(x, (3, 1), 3)
    b = scale(x, (5, 1), 5)
    c = scale(x, (7, 1), 7)
    x = Concatenate()([a, b, c])
    x = Dense(2, activation="softmax")(x)   # 最后一层输出分类结果通常使用softmax
    return Model(inputs=input, outputs=x)
