import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import tensorflow as tf
from plot_cm import plot_conf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Convolution1D, \
    Dense, Conv2D, MaxPool2D, UpSampling2D, UpSampling1D, Convolution1D, MaxPooling1D, Dropout
from tensorflow.keras.models import Model, load_model
from input_process import deal_input

# num = 6
# label_num = {'label_train': 100000, 'label_da': 10000, 'label_te': 8000}
# x_train,y_train,x_train_labeled,y_train_labeled,x_test,y_test,x_valid,y_valid,testres,n_classes=deal_csv(num,label_num)
# reskey = list(testres.keys())
# resvalue = list(testres.values())

dataset_dict = {
        0: 'CICIDS',
        1: 'BoT-IoT',
        2: 'Car-Hacking',
        3: 'IVN',
        4: 'IVN_exp3'
    }
ds_num = 4
dataset = dataset_dict[ds_num]

if dataset == 'BoT-IoT':
    x_train, y_train, x_train_labeled, y_train_labeled, x_test, y_test, labels, label_codes, n_classes = deal_input(
        '../data/BoT-IoT/DL_10features/')
elif dataset == 'CICIDS':
    x_train, y_train, x_train_labeled, y_train_labeled, x_test, y_test, labels, label_codes, n_classes = deal_input(
        '../data/CICIDS/DL/')
elif dataset == 'Car-Hacking':
    x_train, y_train, x_train_labeled, y_train_labeled, x_test, y_test, labels, label_codes, n_classes = deal_input(
        '../data/Car-Hacking/DL/')
elif dataset == 'IVN_exp3':
    x_train, y_train, x_train_labeled, y_train_labeled, x_test, y_test, labels, label_codes, n_classes = deal_input(
        '../data/IVN/DL_exp3/', labelnumber=2)
else:
    x_train, y_train, x_train_labeled, y_train_labeled, x_test, y_test, labels, label_codes, n_classes = deal_input(
        '../data/IVN/DL/', labelnumber=2)


inp_size = x_train_labeled.shape[1]
# step3创建模型
input_c = Input(shape=(inp_size,))
x = tf.expand_dims(input_c, axis=2)
x = Convolution1D(32, 3, padding="same", activation="relu")(x)
x = MaxPooling1D(pool_size=(2))(x)
x = Convolution1D(64,3,padding="same",activation="relu")(x)
x = MaxPooling1D(pool_size=(2))(x)
x = Flatten()(x)
x = Dense(n_classes, activation="softmax")(x)
model = Model(input_c, x, name="CNN")
model.summary()
# step4模型训练
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train_labeled, y_train_labeled, batch_size=256, epochs=100)

# input_c = Input(shape=(inp_size,))
# x = tf.expand_dims(input_c, axis=2)
# x = Convolution1D(512, 3, padding="same", activation="sigmoid")(x)
# x = Dropout(rate=0.1)(x)
# x = MaxPooling1D(pool_size=(2))(x)
# x = Flatten()(x)
# x = Dense(n_classes, activation="softmax")(x)
# model = Model(input_c, x, name="CNN")
# model.summary()
# optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0001)
# model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
# model.fit(x_train_labeled, y_train_labeled, batch_size=256, epochs=200)

# 测试指标
# scores = model.evaluate(x_test, y_test, verbose=1)
y_pred = model.predict(x_test, batch_size=256)
# print("scores", scores)

result_dir = '../result/{}/CNN'.format(dataset)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
pred_true = pd.DataFrame(columns=['y_pred', 'y_true'])
pred_true['y_pred'] = y_pred.argmax(-1)
pred_true['y_true'] = y_test.argmax(-1)
pred_true.to_csv(os.path.join(result_dir, 'class{}_pred.csv'.format(n_classes)), index=False)

# savename = os.path.join(result_dir, 'class{}.png'.format(n_classes))
# plot_conf(y_pred.argmax(-1), y_test.argmax(-1), labels, label_codes, savename)

# 评价指标
report = classification_report(y_test.argmax(-1), y_pred.argmax(-1), target_names=labels, digits=4, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv(os.path.join(result_dir, 'class{}_report.csv'.format(n_classes)), index=True)
print(report)
