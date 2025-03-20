import os
import time
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Convolution1D, \
    Dense, Conv2D, MaxPool2D, UpSampling2D, UpSampling1D, Convolution1D, MaxPooling1D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import tensorflow as tf
from input_process import deal_input


class VAE(tf.keras.Model):

    def __init__(self, latent_dim, inputsize, num_classes):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(inputsize,)),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(inputsize)
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


class preTrainTask():

    def __init__(self, latent_dim, epochs, feature=77):

        super(preTrainTask, self).__init__()
        self.latent_dim = latent_dim
        self.featuresize = feature
        self.epochs = epochs
        self.batch_size = 128

    @tf.function
    def compute_loss(self, model, x) \
            -> tf.Tensor:
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z)
        reconstruction_loss = tf.reduce_mean(tf.square(x - x_logit))
        kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
        total_loss = reconstruction_loss + kl_loss

        return total_loss

    @tf.function
    def train_step(self, model, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def train(self, train_dataset, test_dataset, rate):

        optimizer = tf.keras.optimizers.Adam(0.000008)
        # 修改
        model = VAE(self.latent_dim, self.featuresize, 4)
        model.encoder.summary()
        model.decoder.summary()
        for epoch in range(1, self.epochs + 1):
            # 训练模型
            start_time = time.time()
            for train_x in train_dataset:
                self.train_step(model, train_x, optimizer)
            end_time = time.time()

            # 评估模型
            loss = tf.keras.metrics.Mean()
            # print('loss:', loss)
            for test_x in test_dataset:
                loss(self.compute_loss(model, test_x))
                # print('loss:', loss)
            elbo = -loss.result()
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                  .format(epoch, elbo, end_time - start_time))
        return model.encoder

    def fit(self, dataframe, x_train_labeled, x_test, rate):
        datanew = dataframe.copy()
        # testdata=testdataframe.copy()
        train, test = train_test_split(datanew, test_size=0.1, random_state=0)
        train = train.astype('float32').reshape(len(train), self.featuresize)
        test = test.astype('float32').reshape(len(test), self.featuresize)
        train_dataset = (tf.data.Dataset.from_tensor_slices(train)
                         .batch(self.batch_size))
        test_dataset = (tf.data.Dataset.from_tensor_slices(test)
                        .batch(self.batch_size))
        task = preTrainTask(latent_dim=self.latent_dim, epochs=self.epochs, feature=self.featuresize)
        encode = task.train(train_dataset, test_dataset, rate)
        x_data_train = encode(datanew)
        # x_data_valid = encode(x_valid)
        x_data_test = encode(x_test)
        x_train_labeled = encode(x_train_labeled)
        return x_data_train, x_train_labeled, x_data_test


# num = 1
# label_num = {'label_train': 100000, 'label_da': 10000, 'label_te': 8000}
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

task = preTrainTask(latent_dim=int(latent_dim / 2), epochs=100, feature=inp_size)
x_train, x_train_labeled, x_test = task.fit(x_train, x_train_labeled, x_test, 0.5)
# x_train, x_train_labeled, x_valid, x_test = task.fit(x_train, x_train_labeled, x_valid, x_test, 0.5)
x_train = x_train.numpy()
# x_valid = x_valid.numpy()
x_test = x_test.numpy()
x_train_labeled = x_train_labeled.numpy()
# 定义模型结构
# encoder
# CNN
input_c = Input(shape=(latent_dim,))
x = Dense(np.prod((33, 32)))(input_c)
x = Reshape((33, 32))(x)
# x = Convolution1D(64, 3, padding="same", activation="relu")(x)
x = Convolution1D(128, 3, padding="same", activation="relu")(x)
x = MaxPooling1D(pool_size=(2))(x)
x = Convolution1D(128, 3, padding="same", activation="relu")(x)
# x = Convolution1D(64, 3, padding="same", activation="relu")(x)
x = MaxPooling1D(pool_size=(2))(x)
x = Flatten()(x)
# x = Dense(128, activation="relu")(x)
x = Dense(n_classes, activation="softmax")(x)
cnn = Model(input_c, x, name="VAECNN")
cnn.summary()

# 编译模型
cnn.compile(loss='categorical_crossentropy',  # 为每个输出指定损失函数
            optimizer='adam',  # 使用 adam 优化器
            metrics=['accuracy'])  # 为每个输出指定评估指标
cnn.fit(x_train_labeled, y_train_labeled,  # 输入数据
        epochs=20,  # 训练 10 轮
        batch_size=128,  # 批次大小为 32
        # validation_data=(x_valid, y_valid) # 提供验证数据
        )

# 测试指标
scores = cnn.evaluate(x_test, y_test, verbose=1)
y_pred = cnn.predict(x_test, batch_size=128)
print("scores", scores)

result_dir = '../result/{}/VAE_CNN'.format(dataset)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
pred_true = pd.DataFrame(columns=['y_pred', 'y_true'])
pred_true['y_pred'] = y_pred.argmax(-1)
pred_true['y_true'] = y_test.argmax(-1)
pred_true.to_csv(os.path.join(result_dir, 'class{}_pred.csv'.format(n_classes)), index=False)

# 评价指标
report = classification_report(y_test.argmax(-1), y_pred.argmax(-1), target_names=labels, digits=4, output_dict=True)
df = pd.DataFrame(report).transpose()
print(df)
df.to_csv(os.path.join(result_dir, 'class{}_report.csv'.format(n_classes)), index=True)

