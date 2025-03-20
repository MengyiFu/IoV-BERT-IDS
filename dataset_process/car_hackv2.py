import json

import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from tsv_process import bigram_generation


def hex_to_decimal(hex_str):
    try:
        decimal_value = int(hex_str, 16)
        return decimal_value
    except (ValueError, TypeError):
        return -1


class CANPreprocessor(object):
    def __init__(self, data_dir, save_dir, readme_dir, split_path, bert_save_dir=None, has_all=False):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.readme_dir = readme_dir
        self.bert_save_dir = bert_save_dir
        self.split_path = split_path
        self.has_all = has_all

        self.data = None
        self.columns = ['Ts', 'ID', 'DLC', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'label', 'filename']
        self.label_mapping = {'normal': 0, 'DoS': 1, 'Fuzzy': 2, 'gear': 3, 'RPM': 4}
        self.labels = self.label_mapping.keys()

    def read_data(self):
        """"""
        if self.has_all:
            self.data = pd.read_csv(os.path.join(self.save_dir, 'car-hacking-all.csv'))
        else:
            datasets = []
            filenames = glob.glob(os.path.join(self.data_dir, '*.csv'))
            for filename in filenames:
                print('read file:', filename)
                fname = filename.split('/')[-1].split('.')[0]
                label = fname.split('_')[0]
                dataset = pd.read_csv(filename)
                dataset['filename'] = fname
                dataset.replace({'T': label, 'R': 'normal'}, inplace=True)
                dataset.columns = self.columns
                dataset = self.__complete_missing_data(dataset)
                datasets.append(dataset)

            normal_file = os.path.join(self.data_dir, 'normal_run_data.txt')
            print('read file:', normal_file)
            label = 'normal'
            dataset = pd.read_csv(normal_file, sep=' ', header=None)
            dataset['filename'] = 'normal_run_data'
            columns_to_drop = [0] + list(range(2, 10)) + list(range(11, 19)) + list(range(20, 23))
            dataset.drop(columns=columns_to_drop, inplace=True)
            dataset.drop(dataset.index[-1], inplace=True)
            dataset.insert(11, 'label', label)
            # dataset.replace({'T': 'normal', 'R': label}, inplace=True)
            dataset.columns = self.columns
            dataset = dataset.fillna('')
            datasets.append(dataset)

            self.data = pd.concat(datasets, ignore_index=True)
            self.data['DLC'] = self.data['DLC'].astype(int)
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
            # Remove NaN, -Inf, +Inf, Duplicates
            print('remove')
            self.remove_duplicate_values()
            self.remove_missing_values()
            self.remove_infinite_values()
            self.data.to_csv(os.path.join(self.save_dir, 'car-hacking-all.csv'), index=False)

    def save_readme(self):
        rm = self.data.groupby('filename')['label'].value_counts()
        rm.to_frame()
        rm.to_csv(os.path.join(self.readme_dir, 'README.txt'))

    def save_bert_data(self):
        self.__generate_bigram()
        df = self.data[['label', 'text_a']].copy()
        df['label'] = df['label'].map(self.label_mapping).astype(int)
        if not os.path.exists(self.bert_save_dir):
            os.mkdir(self.bert_save_dir)
        df.to_csv(os.path.join(self.bert_save_dir, 'packet.tsv'), sep='\t', index=False)

    def __complete_missing_data(self, df):
        # 处理缺失值
        df['DLC'] = df['DLC'].astype('int')
        dlc_non8_rows = df[df['DLC'] != 8].index
        # 填充label值
        df.loc[dlc_non8_rows, 'label'] = df.loc[dlc_non8_rows].apply(lambda row: row[f'D{row["DLC"]}'], axis=1)
        column_to_replace = self.columns[3: 11]
        df[column_to_replace] = df[column_to_replace].replace(self.labels, '')
        df = df.fillna('')
        return df

    def remove_duplicate_values(self):
        """"""
        # Remove duplicate rows
        self.data.drop_duplicates(inplace=True, keep=False, ignore_index=True)

    def remove_missing_values(self):
        """"""
        # Remove missing values
        self.data.dropna(axis=0, inplace=True, how="any")

    def remove_infinite_values(self):
        """"""
        # Replace infinite values to NaN
        self.data.replace([-np.inf, np.inf], np.nan, inplace=True)

        # Remove infinte values
        self.data.dropna(axis=0, how='any', inplace=True)

    def __generate_bigram(self):
        self.data['text_a'] = self.data['D0']
        for i in range(1, 8):
            column = 'D' + str(i)
            self.data['text_a'] += self.data[column]
        self.data['text_a'] = self.data['text_a'].apply(lambda x: ' ' + bigram_generation(x))
        self.data['text_a'] = self.data['ID'] + self.data['text_a']

    # def __generate_bigram(self):
    #     self.data['text_a'] = self.data['ID']
    #     for i in range(0, 8):
    #         column = 'D' + str(i)
    #         self.data['text_a'] += self.data[column]
    #     self.data['text_a'] = self.data['text_a'].apply(bigram_generation)

    def split_dataset(self, total_num, dropcol):
        """
        split data
        :param total_num: 每一类的总数
        :param n_class: 3/5/7
        """
        # 取样
        df = self.data
        hex_col = ['ID', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']
        df[hex_col] = df[hex_col].map(lambda x: hex_to_decimal(x))

        df['label'] = df['label'].map(self.label_mapping, na_action='ignore')
        df.drop(columns=dropcol, inplace=True)
        df.dropna(axis=0, how='any', inplace=True)
        df['label'] = df['label'].astype(int)
        print(df.label.value_counts())

        train_df = df[df['label'] == 0].sample(n=100000, replace=True)

        # 对每个组进行随机抽样
        sample_df = df.groupby('label').apply(
            lambda x: x.sample(n=len(x)) if len(x) < total_num else x.sample(n=total_num, random_state=42))
        print(sample_df.label.value_counts())
        # train
        train_labeled_df, test_labeled_df = train_test_split(sample_df, test_size=0.2, stratify=sample_df['label'])
        print(train_labeled_df.label.value_counts())
        # # test
        # test_df, valid_df = train_test_split(test_df, test_size=0.5, stratify=test_df['label'])
        print(test_labeled_df.label.value_counts())
        # print(valid_df.label.value_counts())

        # save
        if not os.path.exists(self.split_path):
            os.makedirs(self.split_path)
        with open(os.path.join(self.split_path, 'label_mapping.json'), 'w') as f:
            json.dump(self.label_mapping, f)
        with open(os.path.join(self.split_path, 'README.txt'), 'w') as f:
            f.write('train_unlabeled:\n{}\n\n'.format(train_df.label.value_counts()))
            f.write('sample_label:\n{}\n\n'.format(sample_df.label.value_counts()))
            f.write('train_labeled_df:\n{}\n\n'.format(train_labeled_df.label.value_counts()))
            f.write('test_labeled_df:\n{}\n\n'.format(test_labeled_df.label.value_counts()))
            # f.write('valid_label:\n{}\n\n'.format(valid_df.label.value_counts()))

        train_df.to_csv(os.path.join(self.split_path, 'train_unlabeled.csv'), index=False)
        train_labeled_df.to_csv(os.path.join(self.split_path, 'train_labeled_dataset.csv'), index=False)
        test_labeled_df.to_csv(os.path.join(self.split_path, 'test_labeled_dataset.csv'), index=False)
        # valid_df.to_csv(os.path.join(self.split_path, 'valid_dataset.csv'), index=False)


if __name__ == "__main__":
    processor = CANPreprocessor(
        data_dir='../data/Car-Hacking/raw/',
        save_dir='../data/Car-Hacking/',
        readme_dir='../data/Car-Hacking/',
        bert_save_dir='../data/Car-Hacking/',
        has_all=True,
        split_path='../data/Car-Hacking/DL/'
    )

    # Read datasets
    print('read')
    processor.read_data()

    # print('save')
    # processor.save_bert_data()
    # processor.save_readme()
    processor.split_dataset(total_num=20000, dropcol=['Ts', 'filename'])

    # df = pd.read_csv('../data/Car-Hacking/car-hacking-all.csv')
    # print(df.label.value_counts())
