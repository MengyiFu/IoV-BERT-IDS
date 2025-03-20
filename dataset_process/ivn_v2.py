import glob
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from tsv_process import bigram_generation
from car_hackv2 import CANPreprocessor, hex_to_decimal


class IVNPreprocessor(CANPreprocessor):
    def __init__(self, data_dir, save_dir, readme_dir, split_path, bert_save_dir=None, has_all=False):
        super().__init__(data_dir, save_dir, readme_dir, split_path, bert_save_dir, has_all)
        self.data = None
        self.columns = ['Ts', 'ID', 'DLC', 'DATA', 'label', 'car_label', 'filename']
        self.label_mapping = {'Normal': 0, 'Flooding': 1, 'Fuzzy': 2, 'Malfunction': 3}
        self.car_label_map = {'CHEVROLET': 0, 'HYUNDAI': 1, 'KIA': 2}

    def read_data(self):
        """"""
        if self.has_all:
            self.data = pd.read_csv(os.path.join(self.save_dir, 'ivn-all.csv'))
        else:
            datasets = []
            filenames = glob.glob(os.path.join(self.data_dir, '*.csv'))
            for filename in filenames:
                print('read file:', filename)
                fname = filename.split('/')[-1].split('.')[0]
                label = fname.split('_')[0]
                car = fname.split('_')[-3]

                dataset = pd.read_csv(filename, header=None)
                if label == 'Attack':
                    dataset.columns = self.columns[0: 4]
                    dataset['label'] = 'Normal'
                else:
                    dataset.columns = self.columns[0: 5]
                    dataset['label'].replace({'T': label, 'R': 'Normal'}, inplace=True)
                dataset['filename'] = fname
                dataset['car_label'] = car
                dataset[['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']] = dataset['DATA'].str.split(' ', expand=True)
                dataset.drop(columns=['DATA'], inplace=True)
                datasets.append(dataset)
                self.data = pd.concat(datasets, ignore_index=True)
                self.data = self.data.fillna('')

                if not os.path.exists(self.save_dir):
                    os.mkdir(self.save_dir)

                # Remove NaN, -Inf, +Inf, Duplicates
                print('remove')
                self.remove_duplicate_values()
                self.remove_missing_values()
                self.remove_infinite_values()
                self.data.to_csv(os.path.join(self.save_dir, 'ivn-all.csv'), index=False)


if __name__ == "__main__":
    processor = IVNPreprocessor(
        data_dir='../data/IVN/In-Vehicle Network Intrusion Detection/car_track_preliminary_train/',
        save_dir='../data/IVN/',
        readme_dir='../data/IVN/',
        bert_save_dir='../data/IVN/',
        has_all=True,
        split_path='../data/IVN/DL/'
    )

    # # Read datasets
    # print('read')
    # processor.read_data()

    # print('save')
    # processor.save_bert_data()
    # processor.save_readme()
    # processor.split_dataset(total_num=20000, dropcol=['Ts', 'filename', 'car_label'])


    # # 划分DL_exp3实验数据
    # split_path = '../data/IVN/DL_exp3/'
    # dropcol = ['Ts', 'filename', 'car_label']
    # label_mapping = {'Normal': 0, 'Flooding': 1, 'Fuzzy': 2, 'Malfunction': 3}
    # df = pd.read_csv('../data/IVN/ivn-all.csv')
    # hex_col = ['ID', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']
    # df[hex_col] = df[hex_col].map(lambda x: hex_to_decimal(x))
    #
    # df['label'] = df['label'].map(label_mapping, na_action='ignore')
    # df.dropna(axis=0, how='any', inplace=True)
    # df['label'] = df['label'].astype(int)
    # print(df.label.value_counts())
    #
    # train_sample = df[df['car_label'] == 'KIA']
    # test_sample = df[df['car_label'] == 'HYUNDAI']
    # train_sample.drop(columns=dropcol, inplace=True)
    # test_sample.drop(columns=dropcol, inplace=True)
    #
    # train_df = train_sample[train_sample['label'] == 0].sample(n=100000, replace=True)
    # train_labeled_df = train_sample.groupby('label').apply(
    #     lambda x: x.sample(n=len(x)) if len(x) < 16000 else x.sample(n=16000, random_state=42))
    #
    # test_labeled_df = test_sample.groupby('label').apply(
    #     lambda x: x.sample(n=len(x)) if len(x) < 4000 else x.sample(n=4000, random_state=42))
    # print(train_labeled_df.label.value_counts())
    # print(test_labeled_df.label.value_counts())
    #
    # # save
    # if not os.path.exists(split_path):
    #     os.makedirs(split_path)
    # with open(os.path.join(split_path, 'label_mapping.json'), 'w') as f:
    #     json.dump(label_mapping, f)
    # with open(os.path.join(split_path, 'README.txt'), 'w') as f:
    #     f.write('train_unlabeled:\n{}\n\n'.format(train_df.label.value_counts()))
    #     f.write('train_labeled_df:\n{}\n\n'.format(train_labeled_df.label.value_counts()))
    #     f.write('test_labeled_df:\n{}\n\n'.format(test_labeled_df.label.value_counts()))
    #     # f.write('valid_label:\n{}\n\n'.format(valid_df.label.value_counts()))
    #
    # train_df.to_csv(os.path.join(split_path, 'train_unlabeled.csv'), index=False)
    # train_labeled_df.to_csv(os.path.join(split_path, 'train_labeled_dataset.csv'), index=False)
    # test_labeled_df.to_csv(os.path.join(split_path, 'test_labeled_dataset.csv'), index=False)


    # 划分BERT_exp3实验数据
    split_path = '../../ET-BERT-fmy/datasets/IVN_exp3/'
    dropcol = ['Ts', 'filename', 'car_label', 'ID', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'DLC']
    label_mapping = {'Normal': 0, 'Flooding': 1, 'Fuzzy': 2, 'Malfunction': 3}
    df = pd.read_csv('../data/IVN/ivn-all.csv')
    df['text_a'] = df['D0']
    for i in range(1, 8):
        column = 'D' + str(i)
        df['text_a'] += df[column]
    df['text_a'] = df['text_a'].apply(lambda x: ' ' + bigram_generation(x))
    df['text_a'] = df['ID'] + df['text_a']

    df['label'] = df['label'].map(label_mapping, na_action='ignore')
    df.dropna(axis=0, how='any', inplace=True)
    df['label'] = df['label'].astype(int)
    print(df.label.value_counts())

    train_sample = df[df['car_label'] == 'KIA']
    test_sample = df[df['car_label'] == 'HYUNDAI']
    train_sample = train_sample.drop(columns=dropcol)
    test_sample = test_sample.drop(columns=dropcol)
    print(train_sample.label.value_counts())
    print(test_sample.label.value_counts())

    # train_sample.reset_index(level='label', inplace=True)
    train_sample = train_sample.groupby('label').apply(
            lambda x: x.sample(n=len(x)) if len(x) < 20000 else x.sample(n=20000, random_state=42))
    # print(train_sample.label.value_counts())
    test_sample = test_sample.groupby('label').apply(
            lambda x: x.sample(n=len(x)) if len(x) < 8000 else x.sample(n=8000, random_state=42))
    # print(test_sample.label.value_counts())

    # train
    train_df, valid_df = train_test_split(train_sample, test_size=0.2, stratify=train_sample['label'])
    print('train_df\n', train_df.label.value_counts())
    print('valid_df\n', valid_df.label.value_counts())
    # test
    test_df, nolabel_df = train_test_split(test_sample, test_size=0.5, stratify=test_sample['label'])
    print('test_df\n', test_df.label.value_counts())
    print('nolabel_df\n', nolabel_df.label.value_counts())

    label = nolabel_df['label']
    label = pd.DataFrame(label, columns=['label'])
    nolabel_df.drop(columns=['label'], inplace=True)

    # save
    if not os.path.exists(split_path):
        os.makedirs(split_path)
    with open(os.path.join(split_path, 'README.txt'), 'w') as f:
        f.write('total_label:\n{}\n\n'.format(df.label.value_counts()))
        f.write('train_label:\n{}\n\n'.format(train_df.label.value_counts()))
        f.write('test_label:\n{}\n\n'.format(test_df.label.value_counts()))
        f.write('valid_label:\n{}\n\n'.format(valid_df.label.value_counts()))
        f.write('no_label:\n{}\n\n'.format(label.label.value_counts()))
    train_df.to_csv(os.path.join(split_path, 'train_dataset.tsv'), sep='\t', index=False)
    test_df.to_csv(os.path.join(split_path, 'test_dataset.tsv'), sep='\t', index=False)
    valid_df.to_csv(os.path.join(split_path, 'valid_dataset.tsv'), sep='\t', index=False)
    nolabel_df.to_csv(os.path.join(split_path, 'nolabel_dataset.tsv'), index=False)
    label.to_csv(os.path.join(split_path, 'label_dataset.tsv'), index=False)