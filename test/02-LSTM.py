import os
import pandas as pd
from sklearn.metrics import classification_report
import tensorflow as tf
from plot_cm import plot_conf
from tensorflow.keras.layers import Input, Flatten, Dense, Bidirectional, GRU, LSTM
from keras.models import Model
from input_process import deal_input

# num=6
# label_num={'label_train':100000,'label_da':10000,'label_te':8000}
# x_train, y_train, x_train_labeled, y_train_labeled, x_test, y_test, x_valid, y_valid, testres, n_classes = deal_bot_iot('../BoT-IoT/UNSW_2018_IoT_Botnet_all.csv', 1100)
# LABELS=list(testres.keys())
# reskey = list(testres.keys())
# resvalue = list(testres.values())

dataset = 'IVN'
# In-vehicle
x_train, y_train, x_train_labeled, y_train_labeled, x_test, y_test, labels, label_codes, n_classes = deal_input(
    '../data/IVN/DL/', False, 2, 'std')
# x_train, y_train, x_train_labeled, y_train_labeled, x_test, y_test, labels, label_codes, n_classes = deal_input(
#     '../data/CICIDS/DL')

inp_size = x_train_labeled.shape[1]
input_c = Input(shape=(inp_size,))
x = tf.expand_dims(input_c, axis=2)
x = LSTM(64, return_sequences=True, dropout=0.2)(x)
x = LSTM(32, return_sequences=True, dropout=0.2)(x)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)
x = Dense(n_classes, activation='softmax')(x)
model = Model(input_c, x, name="GRU")
# model = tf.keras.Sequential([
#     Input(shape=(input_size, 1)),
#     LSTM(128, return_sequences=True, dropout=0.2),
#     LSTM(64, return_sequences=True, dropout=0.2),
#     LSTM(32, return_sequences=True, dropout=0.2),
#     Flatten(),
#     Dense(32, activation='relu'),
#     Dense(class_num, activation='softmax'),
# ])
model.summary()


#step4模型训练
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train_labeled, y_train_labeled, batch_size=256, epochs=100)

#测试指标
scores = model.evaluate(x_test, y_test, verbose=1)
y_pred = model.predict(x_test, batch_size=256)
print("scores", scores)

result_dir = '../result/{}/LSTM'.format(dataset)
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