import glob
import os
import pandas as pd
from car_hackv2 import CANPreprocessor
from tsv_process import bigram_generation


class Can_inPreprocessor(CANPreprocessor):
    def __init__(self, data_dir, save_dir, readme_dir, split_path, bert_save_dir=None, has_all=False):
        super().__init__(data_dir, save_dir, readme_dir, split_path, bert_save_dir, has_all)
        self.data = None
        self.columns = ['Ts', 'ID', 'DLC', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'label', 'filename']
        self.label_mapping = {'Normal': 0, 'DoS': 1, 'Fuzzy': 2, 'Impersonation': 3}

    def read_data(self):
        """"""
        if self.has_all:
            self.data = pd.read_csv(os.path.join(self.save_dir, 'can-intrusion-all.csv'))
        else:
            datasets = []
            filenames = glob.glob(os.path.join(self.data_dir, '*.txt'))
            for filename in filenames:
                print('read file:', filename)
                fname = filename.split('/')[-1].split('.')[0]
                label = fname.split('_')[0]

                df_temp = pd.read_csv(filename)
                df_temp.iloc[:, 0] = df_temp.iloc[:, 0].str.replace(r'\s+', ',', regex=True)
                dataset = df_temp.iloc[:, 0].str.split(',', expand=True)
                columns_to_drop = [0, 2, 4, 5]
                dataset.drop(columns=columns_to_drop, inplace=True)
                if label == 'Attack':
                    dataset['label'] = 'Normal'
                else:
                    dataset['label'] = label
                dataset['filename'] = fname
                dataset.columns = self.columns
                datasets.append(dataset)

            self.data = pd.concat(datasets, ignore_index=True)
            self.data = self.data.fillna('')

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
            self.data.to_csv(os.path.join(self.save_dir, 'can-intrusion-all.csv'), index=False)


if __name__ == "__main__":
    processor = Can_inPreprocessor(
        data_dir='../data/CAN-Intrusion Dataset/',
        save_dir='../data/CAN-Intrusion Dataset/',
        readme_dir='../data/CAN-Intrusion Dataset/',
        bert_save_dir='../data/CAN-Intrusion Dataset/',
        has_all=True,
        split_path='../data/CAN-Intrusion Dataset/DL/'
    )

    # # Read datasets
    # print('read')
    # processor.read_data()

    # # Remove NaN, -Inf, +Inf, Duplicates
    # print('remove')
    # processor.remove_duplicate_values()
    # processor.remove_missing_values()
    # processor.remove_infinite_values()

    # print('save')
    # processor.save_readme()
    # processor.save_bert_data()

    # 按原始文件顺序上下两个packet为一组
    # output_file = '../data/CAN-Intrusion Dataset/encrypted_can_burst.txt'
    # df = pd.read_csv('../data/CAN-Intrusion Dataset/packet.tsv', sep='\t')
    # groups = df.groupby('label')
    # for name, group in groups:
    #     text_a_list = group['text_a'].values.tolist()
    #     for i in range(len(text_a_list)-1):
    #         with open(output_file, 'a') as file:
    #             text_a = text_a_list[i]
    #             text_b = text_a_list[i + 1]
    #             file.write(text_a + '\n')
    #             file.write(text_b + '\n\n')

    # 按原始文件顺序上下两个packet为一组
    output_file = '../data/CAN-Intrusion Dataset/encrypted_can_burst1.txt'
    df = pd.read_csv('../data/CAN-Intrusion Dataset/can-intrusion-all.csv')
    groups = df.groupby('ID')
    for name, group in groups:
        group['text_a'] = group['D0'] + group['D1'] + group['D2'] + group['D3'] + group['D4'] + group['D5'] + group['D6'] + group['D7']
        group['text_a'] = group['ID'] + group['text_a'].apply(lambda x: ' ' + bigram_generation(x))
        text_a_list = group['text_a'].values.tolist()
        for i in range(len(text_a_list)-1):
            with open(output_file, 'a') as file:
                text_a = text_a_list[i]
                text_b = text_a_list[i + 1]
                file.write(text_a + '\n')
                file.write(text_b + '\n\n')