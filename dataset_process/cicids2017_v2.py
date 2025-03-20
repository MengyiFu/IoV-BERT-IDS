import datetime
import json
import socket
import dpkt
import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split


class CICIDS2017Preprocessor(object):
    def __init__(self, data_path, all_path, readme_path, has_all, split_path, pcap_label_path):
        self.data_path = data_path
        self.all_path = all_path
        self.readme_path = readme_path
        self.pcap_label_path = pcap_label_path
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
                if name == 'Thursday-WorkingHours-Morning-WebAttacks':
                    dataset = pd.read_csv(filename, encoding='windows-1252')
                else:
                    dataset = pd.read_csv(filename)
                dataset.columns = [self._clean_column_name(column) for column in dataset.columns]
                dataset['filename'] = name
                self._remove_missing_values(dataset)
                self._remove_duplicate_values(dataset)
                self._remove_infinite_values(dataset)

                # pcap标记
                weekday = name.split('-')[0]
                pcap_label = dataset[
                    ['source_ip', 'source_port', 'destination_ip', 'destination_port', 'protocol', 'timestamp',
                     'label']].copy()
                if weekday == 'Monday':
                    pcap_label['timestamp'] = pd.to_datetime(pcap_label['timestamp'], format='%d/%m/%Y %H:%M:%S')
                else:
                    pcap_label['timestamp'] = pd.to_datetime(pcap_label['timestamp'], format='%d/%m/%Y %H:%M')

                pcap_label['timestamp'] = pcap_label['timestamp'].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
                pcap_label['source_port'] = pcap_label['source_port'].astype(int)
                pcap_label['destination_port'] = pcap_label['destination_port'].astype(int)
                pcap_label['protocol'] = pcap_label['protocol'].astype(int)
                pcap_label.to_csv(os.path.join(self.pcap_label_path, weekday + '-WorkingHours.csv'), mode='a',
                                  index=False)
                with open(os.path.join(self.pcap_label_path, 'README.txt'), 'a') as f:
                    f.write('{}\ncategory:\n{}\n\n'.format(name, pcap_label.label.value_counts()))

                # 拼接所有文件
                datasets.append(dataset)
                with open(os.path.join(self.readme_path, 'README.txt'), 'a') as f:
                    f.write('{}\ncategory:\n{}\n\n'.format(name, dataset.label.value_counts()))
            self.data = pd.concat(datasets, axis=0, ignore_index=True)
            self._remove_missing_values(self.data)
            self._remove_duplicate_values(self.data)
            self._remove_infinite_values(self.data)
            with open(os.path.join(self.readme_path, 'README.txt'), 'a') as f:
                f.write('All category:\n{}\n\n'.format(self.data.label.value_counts()))

    def save_data(self):
        self.data.to_csv(self.all_path, index=False)

    def _clean_column_name(self, column):
        """"""
        column = column.strip(' ')
        column = column.replace('/', '_')
        column = column.replace(' ', '_')
        column = column.lower()
        return column

    def _remove_duplicate_values(self, df):
        """"""
        # Remove duplicate rows
        df.drop_duplicates(inplace=True, keep=False, ignore_index=True)

    def _remove_missing_values(self, df):
        """"""
        # Remove missing values
        df.dropna(axis=0, inplace=True, how="any")

    def _remove_infinite_values(self, df):
        """"""
        # Replace infinite values to NaN
        df.replace([-np.inf, np.inf], np.nan, inplace=True)

        # Remove infinte values
        df.dropna(axis=0, how='any', inplace=True)

    def split_dataset(self, total_num, n_class):
        """
        split data
        :param total_num: 每一类的总数
        :param n_class: 3/5/7
        """
        # 取样
        df = self.data
        if n_class == 12:
            label_mapping = {'BENIGN': 0,
                             'DoS Hulk': 1,
                             'PortScan': 2,
                             'DDoS': 3,
                             'DoS GoldenEye': 4,
                             'FTP-Patator': 5,
                             'SSH-Patator': 6,
                             'DoS slowloris': 7,
                             'DoS Slowhttptest': 8,
                             'Bot': 9,
                             'Web Attack – Brute Force': 10,
                             'Web Attack – XSS': 11}
        else:
            label_mapping = {'BENIGN': 0,
                             'DoS Hulk': 1,
                             'PortScan': 2,
                             'DDoS': 3,
                             'DoS GoldenEye': 4,
                             'FTP-Patator': 5,
                             'SSH-Patator': 6,
                             'DoS slowloris': 7,
                             'DoS Slowhttptest': 8,
                             'Bot': 9,
                             'Web Attack – Brute Force': 10,
                             'Web Attack – XSS': 11,
                             'Infiltration': 12,
                             'Web Attack – Sql Injection': 13,
                             'Heartbleed': 14}
        df['label'] = df['label'].map(label_mapping, na_action='ignore')
        df.drop(columns=['flow_id', 'source_ip', 'source_port', 'destination_ip', 'destination_port', 'protocol',
                         'timestamp', 'filename'], inplace=True)
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


def gettuple5(buf):
    dstip = srcip = dstport = srcport = protocol_index = ''
    eth = dpkt.ethernet.Ethernet(buf)
    if isinstance(eth.data, dpkt.ip.IP):
        ip = eth.data
        srcip = socket.inet_ntoa(ip.src)
        dstip = socket.inet_ntoa(ip.dst)
        protocol_index = ip.p
        if isinstance(ip.data, dpkt.tcp.TCP) or isinstance(ip.data, dpkt.udp.UDP):
            srcport = ip.data.sport
            dstport = ip.data.dport
    elif isinstance(eth.data, dpkt.ip6.IP6):
        ipv6 = eth.data
        srcip = socket.inet_ntop(socket.AF_INET6, ipv6.src)
        dstip = socket.inet_ntop(socket.AF_INET6, ipv6.dst)
        protocol_index = ipv6.nxt
        if isinstance(ipv6.data, dpkt.tcp.TCP) or isinstance(ipv6.data, dpkt.udp.UDP):
            srcport = ipv6.data.sport
            dstport = ipv6.data.dport
    # 五元组:Source IP, Source Port, Destination IP, Destination Port, Protocol,
    tuple5 = '{}-{}-{}-{}-{}'.format(srcip, srcport, dstip, dstport, protocol_index)
    return tuple5


if __name__ == "__main__":
    # # 查看csv文件编码类型
    # import chardet
    # filenames = glob.glob(os.path.join('../data/CICIDS/pcap_label/', '*.csv'))
    # datasets = []
    # for filename in filenames:
    #     with open(filename, 'rb') as f:
    #         result = chardet.detect(f.read())
    #     print(filename, result['encoding'])

    cicids2017 = CICIDS2017Preprocessor(
        data_path='../data/CICIDS/csv',
        readme_path='../data/CICIDS',
        all_path='../data//CICIDS/CICIDS_all.csv',
        split_path='../data/CICIDS/DL/',
        pcap_label_path='../data/CICIDS/pcap_label',
        has_all=True
    )

    # Read datasets
    print('read')
    cicids2017.read_data()

    # print('save')
    # cicids2017.save_data()

    # 划分数据集
    cicids2017.split_dataset(total_num=20000, n_class=12)
    # dataset = pd.read_csv('../data/CICIDS/pcap_label.csv')
    # print(dataset.label.value_counts())

    # # 标记pcap
    # pcappathlist = glob.glob(os.path.join('../data/CICIDS/pcap', '*.pcap'))
    # for pcap in pcappathlist[0: -1]:
    #     filename_pkts = {}
    #     unk = set()
    #     name = pcap.split('/')[-1].split('.')[0]
    #     pcap_label = pd.read_csv('../data/CICIDS/pcap_label/' + name + '.csv', dtype=str)
    #     pcap_label.drop_duplicates(inplace=True, keep=False, ignore_index=True)
    #     pcap_label.dropna(axis=0, inplace=True, how="any")
    #     pcap_label['timestamp'] = pd.to_datetime(pcap_label['timestamp'], format='%d/%m/%Y %H:%M')
    #     pcap_label['timestamp'] = pcap_label['timestamp'] + datetime.timedelta(hours=3)
    #
    #     if name == 'Tuesday-WorkingHours':
    #         pcap_label.loc[pcap_label['label'] == 'SSH-Patator', 'timestamp'] += datetime.timedelta(hours=12)
    #     elif name == 'Wednesday-WorkingHours':
    #         pcap_label.loc[pcap_label['label'] == 'Heartbleed', 'timestamp'] += datetime.timedelta(hours=12)
    #     elif name == 'Thursday-WorkingHours':
    #         pcap_label.loc[pcap_label['label'] == 'Infiltration', 'timestamp'] += datetime.timedelta(hours=12)
    #     elif name == 'Friday-WorkingHours':
    #         pcap_label.loc[pcap_label['label'] == 'DDoS', 'timestamp'] += datetime.timedelta(hours=12)
    #         pcap_label.loc[pcap_label['label'] == 'PortScan', 'timestamp'] += datetime.timedelta(hours=12)
    #
    #     pcap_label['timestamp'] = pcap_label['timestamp'].apply(lambda x: x.strftime('%d/%m/%Y %H:%M'))
    #     pcap_label['flow_id'] = pcap_label['source_ip'] + '-' + pcap_label['source_port'] + '-' + pcap_label['destination_ip'] + '-' + pcap_label['destination_port'] + '-' + pcap_label['protocol'] + '-' + pcap_label['timestamp']
    #     # result = pcap_label.groupby('flow_id')['label'].nunique()
    #     # print(result)
    #
    #     pcap_label_dict = pcap_label.set_index('flow_id')['label'].to_dict()
    #     print(set(pcap_label_dict.values()))
    #
    #     f = open(pcap, mode='rb')
    #     print("正在读取文件：", pcap)
    #
    #     pkts = dpkt.pcap.Reader(f)
    #     for ts, buf in pkts:
    #         timestamp = datetime.datetime.utcfromtimestamp(ts)
    #         # timestamp = timestamp - datetime.timedelta(hours=3)
    #         timestamp = timestamp.strftime('%d/%m/%Y %H:%M')
    #         # print(timestamp)
    #         # 提取IP五元组
    #         tuple5 = gettuple5(buf)
    #         # if tuple5 == '172.16.0.1-45022-192.168.10.51-444-6':
    #         #     print('Heartbleed', timestamp)
    #
    #         tuple5 = tuple5 + '-' + timestamp
    #         # print('tuple5:', tuple5)
    #         # 获取五元组对应的标签
    #         if tuple5 in pcap_label_dict.keys():
    #             label = pcap_label_dict[tuple5]
    #             # print(label)
    #             filename = '../data/CICIDS/pcap_labelled/' + label + '.pcap'
    #             # print('filename: ', filename)
    #
    #             # 写入字典，应用名称：[数据包1, 数据包2, ...]
    #             if filename in filename_pkts.keys():
    #                 filename_pkts[filename] += [[buf, ts]]
    #             else:
    #                 filename_pkts[filename] = [[buf, ts]]
    #             # print('apppacket:', filename_pkts[appname])
    #         else:
    #             unk.add(str(tuple5))
    #             continue
    #     f.close()
    #     # print(unk)
    #
    #     for key in filename_pkts.keys():
    #         f = open(key, 'ab')
    #         pw = dpkt.pcap.Writer(f)
    #         print('正在写入文件：', key)
    #         for buf, ts in filename_pkts[key]:
    #             # print(buf, ts)
    #             pw.writepkt(buf, ts)
