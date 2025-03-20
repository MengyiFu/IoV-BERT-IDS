import glob
import socket
import time
import dpkt
import binascii
import numpy as np
import os
import random

import pandas as pd
from IPy import IP
from sklearn.model_selection import train_test_split

random.seed(40)


class PktFeature():
    def __init__(self, ts, buf):
        self.timestamp = ts
        self.pkt = buf

    # 计算五元组
    def Get5kdp(self):
        try:
            eth = dpkt.ethernet.Ethernet(self.pkt)
            if isinstance(eth.data, dpkt.ip.IP):
                ip = eth.data
                if isinstance(ip, dpkt.ip6.IP6):
                    key = None
                    direction = 0
                    payload = ''
                else:
                    if eth.data.data.__class__.__name__ == "TCP":
                        srcip = socket.inet_ntoa(ip.src)
                        dstip = socket.inet_ntoa(ip.dst)
                        tcp = ip.data
                        srcport = tcp.sport
                        dstport = tcp.dport
                        protocol = "TCP"
                        key, direction = self.__CompareIP(srcip, dstip, srcport, dstport, protocol)
                        data = (binascii.hexlify(bytes(tcp)))
                        payload = data.decode()[8:]
                    elif eth.data.data.__class__.__name__ == "UDP":
                        srcip = socket.inet_ntoa(ip.src)
                        dstip = socket.inet_ntoa(ip.dst)
                        udp = ip.data
                        srcport = udp.sport
                        dstport = udp.dport
                        protocol = "UDP"
                        key, direction = self.__CompareIP(srcip, dstip, srcport, dstport, protocol)
                        data = (binascii.hexlify(bytes(udp)))
                        payload = data.decode()[8:]
                    else:
                        key = None
                        direction = 0
                        payload = ''  # 不是TCP或UDP协议
            else:
                key = None
                direction = 0
                payload = ''  # 不是ip协议或者没有IP层
        except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError):  # 捕获异常
            print("Malformed Packet:Ethernet")  # 错误的包
            key = 'error'
            direction = 0
            payload = ''

        return key, direction, payload

    def __CompareIP(self, srcip, dstip, srcport, dstport, protocol):
        if protocol == 'UDP':
            protocol = 17
        else:
            protocol = 6
        if IP(srcip) < IP(dstip):
            key = srcip + ',' + str(srcport) + ',' + dstip + ',' + str(dstport) + ',' + str(protocol)
            direction = 1
        else:
            key = dstip + ',' + str(dstport) + ',' + srcip + ',' + str(srcport) + ',' + str(protocol)
            direction = -1
        return key, direction


def calculate_statistic_feature(feature_list):
    mean = np.mean(feature_list)
    var = np.var(feature_list)
    std = np.std(feature_list)
    text = str(mean) + ',' + str(var) + ',' + str(std)
    return text


def cut(obj, sec):
    try:
        result = [obj[i:i + sec] for i in range(0, len(obj), sec)]
        remanent_count = len(result[0]) % 4
        if remanent_count == 0:
            pass
        else:
            result = [obj[i:i + sec + remanent_count] for i in range(0, len(obj), sec + remanent_count)]
    except TypeError:
        result = ''
    return result


def bigram_generation(packet_datagram, packet_len=128, flag=True):
    result = ''
    generated_datagram = cut(packet_datagram, 1)
    token_count = 0
    for sub_string_index in range(len(generated_datagram)):
        if sub_string_index != (len(generated_datagram) - 1):
            token_count += 1
            if token_count > packet_len:
                break
            else:
                merge_word_bigram = generated_datagram[sub_string_index] + generated_datagram[sub_string_index + 1]
        else:
            break
        result += merge_word_bigram
        result += ' '
    return result


def get_burst_feature(pcapfile, output_file):
    with open(pcapfile, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        sessions = {}
        print('正在读取文件', pcapfile)
        for ts, buf in pcap:
            pkt = PktFeature(ts, buf)
            key, direction, payload = pkt.Get5kdp()
            pktlen = len(payload)
            if key is not None and key != 'error' and pktlen != 0:
                if key not in sessions.keys():
                    sessions[key] = {'direction': [], 'payload': []}
                sessions[key]['direction'].append(direction)
                sessions[key]['payload'].append(payload)

        for key in sessions.keys():
            burst = []
            s_direction = sessions[key]['direction']
            s_payload = sessions[key]['payload']

            for i in range(len(s_direction)):
                if i == 0:
                    burst.append(s_payload[i])
                elif s_direction[i] == s_direction[i - 1]:
                    burst[-1] += s_payload[i]
                else:
                    burst.append(s_payload[i])

            for i in range(len(burst)):
                with open(output_file, 'a') as file:
                    if i != len(burst) - 1:
                        text_a = bigram_generation(burst[i])
                        text_b = bigram_generation(burst[i + 1])
                    elif len(burst) == 1:
                        text = burst[i]
                        a = text[:(len(text) // 2)]
                        b = text[(len(text) // 2):]
                        text_a = bigram_generation(a)
                        text_b = bigram_generation(b)
                    file.write(text_a + '\n')
                    file.write(text_b + '\n\n')


def get_packet_feature(pcapfile, label, output_file):
    with open(pcapfile, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        idx = 1
        print('正在读取文件', pcapfile)
        for ts, buf in pcap:
            pkt = PktFeature(ts, buf)
            key, direction, payload = pkt.Get5kdp()
            pktlen = len(payload)
            if key is not None and key != 'error' and pktlen != 0:
                # print('idx:', idx, 'key:', key, 'payload:', payload)
                with open(output_file, 'a') as file:
                    packet_txt = label + '\t' + bigram_generation(payload)
                    file.write(packet_txt + '\n')
            idx = idx + 1


def get_flow_feature(pcapfile, label, output_file, pkt_num=5):
    with open(pcapfile, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        flow = {}
        idx = 1
        print('extract flows')
        for ts, buf in pcap:
            pkt = PktFeature(ts, buf)
            key, direction, payload = pkt.Get5kdp()
            pktlen = len(payload)
            if key is not None and key != 'error' and pktlen != 0:
                # print('idx:', idx, 'key:', key, 'payload:', payload)
                if key not in flow.keys():
                    flow[key] = {}
                    flow[key]['payload'] = [payload]
                    # flow[key]['PacketLength'] = [pktlen]
                    # flow[key]['Timestamp'] = [ts]
                else:
                    flow[key]['payload'].append(payload)
                    # flow[key]['PacketLength'].append(pktlen)
                    # flow[key]['Timestamp'].append(ts)
            idx = idx + 1

        print('get and write flow_pkt')
        for key in flow.keys():
            s_payload = flow[key]['payload']
            # s_pktlen = flow[key]['PacketLength']
            # s_ts = flow[key]['Timestamp']
            l = len(s_payload)
            # print('key:', key, 'payload:', s_payload, 'len:', l)
            if l < pkt_num:
                flow_pkt = ''.join(s_payload)
            else:
                flow_pkt = ''.join(s_payload[:pkt_num])
            # pkt_len_feature = calculate_statistic_feature(s_pktlen)
            # if len(s_ts) > 1:
            #     s_ts_diff = np.diff(s_ts)
            #     ts_diff_feature = calculate_statistic_feature(s_ts_diff)
            # else:
            #     ts_diff_feature = '0,0,0'
            flow_bigram = bigram_generation(flow_pkt)
            # with open(flow_feature_file, 'a') as f:
            #     flow_txt = key + ',' + pkt_len_feature + ',' + ts_diff_feature + ',' + flow_bigram + ',' + label
            #     f.write(flow_txt + '\n')
            with open(output_file, 'a') as file:
                flow_txt = label + '\t' + flow_bigram
                file.write(flow_txt + '\n')


def preprocess(pcap_dir, level, dataset, output_file):
    """
    extract packet/flow payload
    :param pcap_dir: pcap所在目录
    :param level: packet/flow/burst
    :param dataset: BoT-IoT/CICIDS
    :param output_file:
    :return:
    """
    # 遍历pcap
    pcaplist = glob.glob(os.path.join(pcap_dir, '*.pcap'))

    if level != 'burst':
        with open(output_file, 'w') as file:
            file.write('label' + '\t' + 'text_a' + '\n')

    # 读取pcap中所有的包
    for i, pcap in enumerate(pcaplist):
        print('处理进度：{:.1f}%，正在处理第{}个文件{}'.format((i + 1) / len(pcaplist) * 100, i + 1, pcap))

        if dataset == 'BoT-IoT':
            label_name = str(pcap.split('/')[-1]).split('.')[0]
            # if len(label_name) > 1:
            #     label_name = '_'.join(label_name)
            # else:
            #     label_name = label_name[0]
            # label_mapping = {
            #     'DoSUDP': 0,
            #     'DDoSTCP': 1,
            #     'DDoSUDP': 2,
            #     'DoSTCP': 3,
            #     'ReconnaissanceService_Scan': 4,
            #     'ReconnaissanceOS_Fingerprint': 5,
            #     'DoSHTTP': 6,
            #     'DDoSHTTP': 7,
            #     'Normal': 8,
            #     'TheftKeylogging': 9,
            #     'TheftData_Exfiltration': 10
            # }
            label_mapping = {
                'DoSUDP': 0,
                'DDoSUDP': 1,
                'ReconnaissanceService_Scan': 2,
                'ReconnaissanceOS_Fingerprint': 2,
                'DoSHTTP': 0,
                'DDoSHTTP': 1,
                'Normal': 4,
                'TheftKeylogging': 3,
                'TheftData_Exfiltration': 3
            }
        else:
            # CICIDS2017
            label_name = str(pcap.split('/')[-1]).split('.')[0]
            # label_mapping = {
            #     'BENIGN': 0,
            #     'DoS Hulk': 1,
            #     'PortScan': 2,
            #     'DDoS': 3,
            #     'DoS GoldenEye': 4,
            #     'FTP-Patator': 5,
            #     'SSH-Patator': 6,
            #     'DoS slowloris': 7,
            #     'DoS Slowhttptest': 8,
            #     'Bot': 9,
            #     'Web Attack – Brute Force': 10,
            #     'Web Attack – XSS': 11,
            #     'Infiltration': 12,
            #     'Web Attack – Sql Injection': 13,
            #     'Heartbleed': 14
            # }
            label_mapping = {
                'BENIGN': 0,
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
                'Web Attack – XSS': 11
            }

        if level != 'burst':
            try:
                label = str(label_mapping[label_name])
            except KeyError:
                continue
        else:
            label = None

        if level == 'flow':
            get_flow_feature(pcap, label, output_file)
        elif level == 'packet':
            get_packet_feature(pcap, label, output_file)
        else:
            get_burst_feature(pcap, output_file)


def split_dataset(data_path, total_num, output_dir):
    """
    split finetune data
    :param data_path:
    :param total_num: 每一类的总数
    :param output_dir:
    :return: output train\test\valid\nolabel+label.tsv
    """
    df = pd.read_csv(data_path, sep='\t')
    print(df.label.value_counts())

    # 取样
    # 对每个组进行随机抽样
    df_sample = df.groupby('label').apply(
        lambda x: x.sample(n=len(x)) if len(x) < total_num else x.sample(n=total_num, random_state=42))
    print('df_sample\n', df_sample.label.value_counts())

    # train
    train_df, test_df = train_test_split(df_sample, test_size=0.2, stratify=df_sample['label'])
    print('train_df\n', train_df.label.value_counts())
    # # test
    # test_df, valid_df = train_test_split(test_df, test_size=0.6, stratify=test_df['label'])
    print('test_df\n', test_df.label.value_counts())
    # # valid, nolabel
    # valid_df, nolabel_df = train_test_split(valid_df, test_size=0.5, stratify=valid_df['label'])
    # print('valid_df\n', valid_df.label.value_counts())
    # print('nolabel_df\n', nolabel_df.label.value_counts())

    # label = nolabel_df['label']
    # label = pd.DataFrame(label, columns=['label'])
    # nolabel_df.drop(columns=['label'], inplace=True)
    label = test_df['label']
    label = pd.DataFrame(label, columns=['label'])
    test_df.drop(columns=['label'], inplace=True)

    # # save
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # with open(os.path.join(output_dir, 'README.txt'), 'w') as f:
    #     f.write('total_label:\n{}\n\n'.format(df.label.value_counts()))
    #     f.write('sample_label:\n{}\n\n'.format(df_sample.label.value_counts()))
    #     f.write('train_label:\n{}\n\n'.format(train_df.label.value_counts()))
    #     f.write('test_label:\n{}\n\n'.format(test_df.label.value_counts()))
    #     f.write('valid_label:\n{}\n\n'.format(valid_df.label.value_counts()))
    #     f.write('no_label:\n{}\n\n'.format(label.label.value_counts()))
    # train_df.to_csv(os.path.join(output_dir, 'train_dataset.tsv'), sep='\t', index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.tsv'), sep='\t', index=False)
    # valid_df.to_csv(os.path.join(output_dir, 'valid_dataset.tsv'), sep='\t', index=False)
    # nolabel_df.to_csv(os.path.join(output_dir, 'nolabel_dataset.tsv'), index=False)
    label.to_csv(os.path.join(output_dir, 'label.tsv'), index=False)


if __name__ == '__main__':
    dataset_dict = {
        0: 'CICIDS',
        1: 'BoT-IoT',
        2: 'Car-Hacking',
        3: 'IVN'
    }
    ds_num = 3
    dataset = dataset_dict[ds_num]
    level = 'flow'
    if dataset == 'BoT-IoT':
        pcap_dir = "../data/BoT-IoT/stime_extract_pcap"
        # flow_feature_file = '../BoT-IoT/flow_feature.csv'
        all_file = '../data/BoT-IoT/bert/{}_5TCP.tsv'.format(level)
        split_dir = '../../ET-BERT-fmy/datasets/BoT-IoT/{}_5TCP'.format(level)
    elif dataset == 'CICIDS':
        pcap_dir = "../data/CICIDS/pcap_labelled"
        # flow_feature_file = '../CICIDS/DL/flow_feature.csv'
        all_file = '../data/CICIDS/bert/{}_12.tsv'.format(level)
        split_dir = '../../ET-BERT-fmy/datasets/CICIDS/{}_12'.format(level)
    else:
        all_file = '../data/{}/packet.tsv'.format(dataset)
        split_dir = '../../ET-BERT-fmy/datasets/{}'.format(dataset)

    # burst_file = '../encrypted_traffic_burst.txt'

    # preprocess(pcap_dir, level, dataset, all_file)

    split_dataset(all_file, 20000, split_dir)
    # df = pd.read_csv(all_file, sep='\t')
    # print(df.head(10))
    # print(df.label.value_counts())