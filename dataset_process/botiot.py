import json

import dpkt
import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split


class BoT_IoT_Preprocessor(object):
    def __init__(self, data_path, all_path, readme_path, has_all, split_path):
        self.data_path = data_path
        self.all_path = all_path
        self.readme_path = readme_path
        self.split_path = split_path
        self.has_all = has_all

        self.data = None

    def read_data(self):
        """"""
        if self.has_all:
            self.data = pd.read_csv(self.all_path)
        else:
            filenames = glob.glob(os.path.join(self.data_path, '*.csv'))
            datasets = []
            for filename in filenames:
                print('read file:', filename)
                name = filename.split('/')[-1].split('.')[0]
                dataset = pd.read_csv(filename)
                dataset['filename'] = name
                dataset = self.__sub_label(dataset)
                datasets.append(dataset)
                with open(os.path.join(self.readme_path, 'README.txt'), 'a') as f:
                    f.write('{}\ncategory:\n{}\n'.format(name, dataset.category.value_counts()))
                    f.write('sub_category:\n{}\n\n'.format(dataset.sub_category.value_counts()))
            self.data = pd.concat(datasets, axis=0, ignore_index=True)
            with open(os.path.join(self.readme_path, 'README.txt'), 'a') as f:
                f.write('All category:\n{}\n'.format(self.data.category.value_counts()))
                f.write('All sub_category:\n{}\n\n'.format(self.data.sub_category.value_counts()))

    def save_data(self):
        self.data.to_csv(self.all_path, index=False)

    def __sub_label(self, dataset):
        dataset.drop(columns=['pkSeqID', 'flgs', 'proto', 'saddr', 'sport', 'daddr', 'dport', 'state'], inplace=True)
        dataset['sub_category'] = dataset['category'] + dataset['subcategory']
        dataset['sub_category'].replace('NormalNormal', 'Normal', inplace=True)
        return dataset

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

    def split_dataset(self, total_num, n_class):
        """
        split data
        :param total_num: 每一类的总数
        :param n_class: 3/5/7
        """
        # 取样
        df = self.data
        if n_class == 3:
            label_mapping = {'DoS': 0, 'DDoS': 1, 'Reconnaissance': 2}
            label_col = 'category'
        elif n_class == 5:
            label_mapping = {'DoS': 0, 'DDoS': 1, 'Reconnaissance': 2, 'Theft': 3, 'Normal': 4}
            label_col = 'category'
        else:
            label_mapping = {
                'DoSUDP': 0,
                'DDoSTCP': 1,
                'DDoSUDP': 2,
                'DoSTCP': 3,
                'ReconnaissanceService_Scan': 4,
                'ReconnaissanceOS_Fingerprint': 5,
                'DoSHTTP': 6,
                'DDoSHTTP': 7,
                'Normal': 8,
                'TheftKeylogging': 9,
                'TheftData_Exfiltration': 10
            }
            label_col = 'sub_category'
        df['label'] = df[label_col].map(label_mapping, na_action='ignore')
        df = df[['seq', 'stddev', 'N_IN_Conn_P_SrcIP', 'min', 'state_number', 'mean', 'N_IN_Conn_P_DstIP', 'drate', 'srate', 'max', 'label']]
        # df = df.drop(columns=['attack', 'category', 'subcategory', 'filename', 'sub_category'], axis=1)
        df.dropna(axis=0, how='any', inplace=True)
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
            json.dump(label_mapping, f)
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
    bot_iot = BoT_IoT_Preprocessor(
        data_path='../data/BoT-IoT/raw/',
        readme_path='../data/BoT-IoT/',
        all_path='../data/BoT-IoT/UNSW_2018_IoT_Botnet_all.csv',
        split_path='../data/BoT-IoT/DL_10features',
        has_all=True
    )

    # Read datasets
    print('read')
    bot_iot.read_data()

    # Remove NaN, -Inf, +Inf, Duplicates
    print('remove')
    bot_iot.remove_duplicate_values()
    bot_iot.remove_missing_values()
    bot_iot.remove_infinite_values()

    print('save')
    bot_iot.save_data()
    bot_iot.split_dataset(total_num=20000, n_class=5)


    # # stime标记pcap
    # pcappathlist = glob.glob(os.path.join('../data/BoT-IoT/raw_pcap/DDoS_TCP', '*.pcap'))
    # pcap_label = pd.read_csv('../data/BoT-IoT/ground-truth/DDoS_TCP.csv', sep=';')
    # pcap_label = pcap_label[pcap_label['category'] != 'Normal']
    # stime = pcap_label['stime'].tolist()
    # # grouped = pcap_label.groupby('stime')['category']
    # # for st, group in grouped:
    # #     if group.nunique() > 1:
    # #         print(f'for stime = {st}, there are multiple unique labels:')
    # #         print(group.unique())
    # print('list:', len(stime))
    # stime = set(stime)
    # print('set:', len(stime))
    #
    # name = 'DDoSTCP'
    # filename = '../data/BoT-IoT/stime_extract_pcap/' + name + '.pcap'
    # # print('filename: ', filename)
    # filename_pkts = []
    #
    # for pcap in pcappathlist:
    #
    #     f = open(pcap, mode='rb')
    #     print("正在读取文件：", pcap)
    #
    #     pkts = dpkt.pcap.Reader(f)
    #     for ts, buf in pkts:
    #         # 获取五元组对应的标签
    #         if ts in stime:
    #             # 写入字典，应用名称：[数据包1, 数据包2, ...]
    #             if len(filename_pkts) != 0:
    #                 filename_pkts += [[buf, ts]]
    #             else:
    #                 filename_pkts = [[buf, ts]]
    #         else:
    #             continue
    #     f.close()
    #
    # f = open(filename, 'ab')
    # pw = dpkt.pcap.Writer(f)
    # print('正在写入文件：', filename)
    # for buf, ts in filename_pkts:
    #     pw.writepkt(buf, ts)
    #
    # print('{}: stime {}, pkt_num {}'.format(name, len(stime), len(filename_pkts)))

    # 提纯Normal.pcap
    # pcappathlist = glob.glob(os.path.join('../data/BoT-IoT/ground-truth', '*.csv'))
    # all_normal = set()
    # for file in pcappathlist:
    #     pcap_label = pd.read_csv(file, sep=';')
    #     pcap_label = pcap_label[pcap_label['category'] == 'Normal']
    #     stime = pcap_label['stime'].tolist()
    #     stime = set(stime)
    #     all_normal.update(stime)
    #
    # pcappathlist = glob.glob('../data/BoT-IoT/raw_pcap/**/*.pcap', recursive=True)
    # name = 'Normal'
    # filename = '../data/BoT-IoT/stime_extract_pcap/' + name + '.pcap'
    # # print('filename: ', filename)
    # filename_pkts = []
    #
    # for pcap in pcappathlist:
    #     f = open(pcap, mode='rb')
    #     print("正在读取文件：", pcap)
    #
    #     pkts = dpkt.pcap.Reader(f)
    #     for ts, buf in pkts:
    #         # 获取五元组对应的标签
    #         if ts in all_normal:
    #             # 写入字典，应用名称：[数据包1, 数据包2, ...]
    #             if len(filename_pkts) != 0:
    #                 filename_pkts += [[buf, ts]]
    #             else:
    #                 filename_pkts = [[buf, ts]]
    #         else:
    #             continue
    #     f.close()
    #
    # f = open(filename, 'ab')
    # pw = dpkt.pcap.Writer(f)
    # print('正在写入文件：', filename)
    # for buf, ts in filename_pkts:
    #     pw.writepkt(buf, ts)
    #
    # print('{}: stime {}, pkt_num {}'.format(name, len(all_normal), len(filename_pkts)))
