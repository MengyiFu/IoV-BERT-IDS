import json
import os
import pandas as pd
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from plot_cm import plot_conf


def process_standard(x):
    scaler = StandardScaler()
    scaler.fit(x)
    return scaler.fit_transform(x)


def process_minmax(x):
    scaler = MinMaxScaler()
    scaler.fit(x)
    return scaler.fit_transform(x)


def process_da(dataframe, label_number, normalize):
    label = dataframe.pop(dataframe.columns[label_number]).values
    if normalize == 'std':
        data = process_standard(dataframe)
    else:
        data = dataframe.values / 255.0
    return data, label


def deal_dataset(df, label_number, isgan, normalize):
    if isgan:
        # df.pop('stime')
        df.pop('bwd_avg_bytes_bulk')
    # 数据归一化
    x, y = process_da(df, label_number, normalize)
    if not isgan:
        # 标签编码
        y = to_categorical(y)
    return x, y


def deal_input(data_path, isgan=False, labelnumber=-1, normalize='std'):
    # train_labeled_df: 有标记微调训练集
    # train_df: 无标记预训练集
    # test_df: 有标记微调测试集
    train_labeled_df = pd.read_csv(os.path.join(data_path, 'train_labeled_dataset.csv'))
    train_df = pd.read_csv(os.path.join(data_path, 'train_unlabeled.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test_labeled_dataset.csv'))
    # valid_df = pd.read_csv(os.path.join(data_path, 'valid_dataset.csv'))
    x_train_labeled, y_train_labeled = deal_dataset(train_labeled_df, labelnumber, isgan, normalize)
    x_train, y_train = deal_dataset(train_df, labelnumber, isgan, normalize)
    x_test, y_test = deal_dataset(test_df, labelnumber, isgan, normalize)
    # x_valid, y_valid = deal_dataset(valid_df, -1, isgan)
    with open(os.path.join(data_path, 'label_mapping.json'), 'r') as f:
        label_mapping = json.load(f)
    labels = list(label_mapping.keys())
    label_codes = list(label_mapping.values())
    n_classes = len(label_codes)
    if isgan:
        return x_train, y_train, x_train_labeled, y_train_labeled, x_test, y_test, label_mapping, n_classes
        # return x_train, y_train, x_train_labeled, y_train_labeled, x_test, y_test, x_valid, y_valid, label_mapping, n_classes
    else:
        # return x_train, y_train, x_train_labeled, y_train_labeled, x_test, y_test, x_valid, y_valid, labels, label_codes, n_classes
        return x_train, y_train, x_train_labeled, y_train_labeled, x_test, y_test, labels, label_codes, n_classes