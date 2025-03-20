import os

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Convolution1D, \
    Dense, Conv2D, MaxPool2D, UpSampling2D, UpSampling1D, Convolution1D, MaxPooling1D
from tensorflow.keras.models import Model, load_model
import pandas as pd
from sklearn.metrics import classification_report
from input_process import deal_input

latent_dim = 6
# num = 6
# label_num = {'label_train': 100000, 'label_da': 10000, 'label_te': 3000, 'label_va': 2000}
# x_train, y_train, x_train_labeled, y_train_labeled, x_test, y_test, x_valid, y_valid, testres, n_classes = deal_csv(num, label_num)
# LABELS = list(testres.keys())
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
    latent_dim = 20
elif dataset == 'CICIDS':
    x_train, y_train, x_train_labeled, y_train_labeled, x_test, y_test, labels, label_codes, n_classes = deal_input(
        '../data/CICIDS/DL/')
    latent_dim = 20
elif dataset == 'Car-Hacking':
    x_train, y_train, x_train_labeled, y_train_labeled, x_test, y_test, labels, label_codes, n_classes = deal_input(
        '../data/Car-Hacking/DL/')
    latent_dim = 4
elif dataset == 'IVN_exp3':
    x_train, y_train, x_train_labeled, y_train_labeled, x_test, y_test, labels, label_codes, n_classes = deal_input(
        '../data/IVN/DL_exp3/', labelnumber=2)
    latent_dim = 4
else:
    x_train, y_train, x_train_labeled, y_train_labeled, x_test, y_test, labels, label_codes, n_classes = deal_input(
        '../data/IVN/DL/', labelnumber=2)
    latent_dim = 4

inp_size = x_train.shape[1]

# 定义模型结构
# encoder
# encoder
input_shape = (inp_size)
input_e = Input(shape=input_shape)
x = Dense(150, activation="relu", name='dense_1')(input_e)
x = Dense(100, activation="relu", name='dense_2')(x)
x = Dense(50, activation="relu", name='dense_3')(x)
latent = Dense(latent_dim, activation="relu")(x)
encoder = Model(input_e, latent, name="encoder")
encoder.summary()
# decoder
input_d = Input(shape=(latent_dim))
x = Dense(50, activation="relu", name='dense_4')(input_d)
x = Dense(100, activation="relu", name='dense_5')(x)
x = Dense(150, activation="relu", name='dense_6')(x)
output = Dense(inp_size, activation="relu", name='dense_7')(x)
decoder = Model(input_d, output, name="decoder")
decoder.summary()
# mlp
input_c = Input(shape=(latent_dim,))
x = Dense(150, activation="relu", name='dense_8')(input_c)
x = Dense(100, activation="relu", name='dense_9')(x)
x = Dense(50, activation="relu", name='dense_10')(x)
x = Dense(n_classes, activation="softmax", name='dense_11')(x)
mlp = Model(input_c, x, name="mlp")
mlp.summary()

autoencoder = Model(input_e, decoder(encoder(input_e)))
autoencoder.encoder = encoder
classificationmodel = Model(input_e, mlp(encoder(input_e)))
classificationmodel.encoder = encoder
classificationmodel.mlp = mlp

# 训练模型
classificationmodel.mlp.trainable = False  # 将编码器部分设置为不可训练
# 编译模型
autoencoder.compile(loss='mse', optimizer='adam', metrics='mse')
autoencoder.fit(x_train, x_train,  # 输入数据
                epochs=10,  # 训练 10 轮
                batch_size=32,  # 批次大小为 32
                # validation_data=(x_valid, x_valid)
                )

classificationmodel.encoder.trainable = False  # 将编码器部分设置为不可训练
classificationmodel.mlp.trainable = True

classificationmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
classificationmodel.fit(x_train_labeled, y_train_labeled,  # 输入数据
                        epochs=100,
                        batch_size=128,
                        # validation_data=(x_valid, y_valid)
                        )
encoder2weights = classificationmodel.encoder.get_weights()

# 测试指标
scores = classificationmodel.evaluate(x_test, y_test, verbose=1)
y_pred = classificationmodel.mlp.predict(encoder(x_test), batch_size=128)
print("scores", scores)

result_dir = '../result/{}/AE_MLP'.format(dataset)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
pred_true = pd.DataFrame(columns=['y_pred', 'y_true'])
pred_true['y_pred'] = y_pred.argmax(-1)
pred_true['y_true'] = y_test.argmax(-1)
pred_true.to_csv(os.path.join(result_dir, 'class{}_pred.csv'.format(n_classes)), index=False)

# 评价指标
report = classification_report(y_test.argmax(-1), y_pred.argmax(-1), target_names=labels, digits=4, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv(os.path.join(result_dir, 'class{}_report.csv'.format(n_classes)), index=True)
print(report)
