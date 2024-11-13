import copy
import json
import os.path
import random

import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, datas):
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        return self.datas[item]


class CustomDataloader:
    def __init__(self, batch_size, data_dir):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.token2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3, }
        self.prompt1 = '请根据给定的上联生成下联：'
        self.prompt2 = '请根据给定的上下联第一个字生成完整对联：'

    def collate_fn(self, batch):
        batch_enc_input, batch_dec_input, batch_label = [], [], []
        batch_enc_len, batch_dec_len = [], []
        for text in batch:
            # 拆分上下句
            text1, text2 = text.split('，')
            if len(text1) < 3 or len(text2) < 3:
                continue
            enc_text1 = self.prompt1 + text1
            enc1 = [self.token2idx[char] if char in self.token2idx.keys() else self.token2idx['<UNK>'] for char in
                    enc_text1]
            enc1.insert(0, self.token2idx['<START>'])
            enc1.append(self.token2idx['<END>'])

            dec1 = [self.token2idx[char] if char in self.token2idx.keys() else self.token2idx['<UNK>'] for char in
                    text2]
            dec1_input = copy.deepcopy(dec1)
            dec1_input.insert(0, self.token2idx['<START>'])

            y1 = copy.deepcopy(dec1)
            y1.append(self.token2idx['<END>'])

            batch_enc_input.append(enc1)
            batch_dec_input.append(dec1_input)
            batch_enc_len.append(len(enc1))
            batch_dec_len.append(len(dec1_input))
            batch_label.append(y1)

            # 拆分上下联第一个字
            enc_text2 = text1[0] + text2[0]
            enc_text2 = self.prompt2 + enc_text2
            enc2 = [self.token2idx[char] if char in self.token2idx.keys() else self.token2idx['<UNK>'] for char in
                    enc_text2]
            enc2.insert(0, self.token2idx['<START>'])
            enc2.append(self.token2idx['<END>'])

            dec2 = [self.token2idx[char] if char in self.token2idx.keys() else self.token2idx['<UNK>'] for char in
                    text]
            dec2_input = copy.deepcopy(dec2)
            dec2_input.insert(0, self.token2idx['<START>'])

            y2 = copy.deepcopy(dec2)
            y2.append(self.token2idx['<END>'])

            batch_enc_input.append(enc2)
            batch_dec_input.append(dec2_input)
            batch_enc_len.append(len(enc2))
            batch_dec_len.append(len(dec2_input))
            batch_label.append(y2)

        # 构建mask
        enc_max_len, dec_max_len = max(batch_enc_len), max(batch_dec_len)
        batch_size = len(batch_enc_input)
        enc_mask = torch.zeros(size=(batch_size, enc_max_len))
        dec_mask = torch.zeros(size=(batch_size, dec_max_len))
        for i in range(len(batch_enc_len)):
            enc_mask[i, :batch_enc_len[i]] = 1
            dec_mask[i, :batch_dec_len[i]] = 1

            enc_pad_len = enc_max_len - batch_enc_len[i]
            batch_enc_input[i].extend([0] * enc_pad_len)

            dec_pad_len = dec_max_len - batch_dec_len[i]
            batch_dec_input[i].extend([0] * dec_pad_len)
            batch_label[i].extend([0] * dec_pad_len)

        batch_enc_input = torch.tensor(batch_enc_input, dtype=torch.int64)
        batch_dec_input = torch.tensor(batch_dec_input, dtype=torch.int64)
        batch_label = torch.tensor(batch_label, dtype=torch.int64)
        return (batch_enc_input, enc_mask.to(dtype=torch.float32), batch_dec_input,
                dec_mask.to(dtype=torch.float32)), batch_label

    def get_dataloader(self):
        new_datas = self.data_processing()
        data_len = len(new_datas)
        train_len = int(data_len * 0.8)
        new_datas_clone = new_datas[:]
        random.shuffle(new_datas_clone)
        train_datas = new_datas_clone[: train_len]
        test_datas = new_datas_clone[train_len:]

        # dataset = MyDataset(new_datas)
        # dataloader = DataLoader(
        #     dataset=dataset,
        #     shuffle=True,
        #     batch_size=self.batch_size,
        #     collate_fn=self.collate_fn
        # )

        train_dataset = MyDataset(new_datas)
        test_dataset = MyDataset(test_datas)
        train_dataloader = DataLoader(
            dataset=train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn
        )

        return train_dataloader, test_dataloader, self.token2idx

    def data_processing(self):
        vocabulary = set()
        new_datas = []
        with open(os.path.join(self.data_dir, 'poems_edge_split.txt'), 'r', encoding='utf-8') as reader:
            for line in reader.readlines():
                for text in line.strip()[1:-1].split('。'):
                    if text == '' or len(text.split('，')) != 2 or len(text.split('，')[0]) != len(text.split('，')[1]):
                        continue
                    vocabulary.update(text)
                    new_datas.append(text)
        with open(os.path.join(self.data_dir, 'poems.txt'), 'w', encoding='utf-8') as writer:
            writer.write('\n'.join(new_datas))

        vocabulary.update(self.prompt1 + self.prompt2)

        vocabulary = list(vocabulary)
        for i in ['(', '2', ';', 'í', 'ó', 'ē', 'ī', '□', '、', 'Ｃ', 'ｒ', 'ｗ', '￣']:
            vocabulary.remove(i)

        index = 4
        for token in vocabulary:
            self.token2idx[token] = index
            index += 1

        with open(os.path.join(self.data_dir, 'token2idx.json'), 'w', encoding='utf-8') as writer:
            json.dump(self.token2idx, writer, ensure_ascii=False, indent=4)

        return new_datas
