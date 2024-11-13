import copy
import os.path
import random

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig


class MyDataset(Dataset):
    def __init__(self, datas):
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        return self.datas[item]


class MT5Dataloader:
    def __init__(self, batch_size, data_dir, t5_model_dir):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_dir, legacy=False)
        self.t5_config = AutoConfig.from_pretrained(t5_model_dir)
        self.prompt1 = self.t5_tokenizer.convert_tokens_to_ids(list('请根据给定的上联生成下联:'))
        self.prompt2 = self.t5_tokenizer.convert_tokens_to_ids(list('请根据给定的上下联第一个字生成完整对联:'))

    def collate_fn(self, batch):
        batch_enc_input, batch_dec_input, batch_label = [], [], []
        batch_enc_len, batch_dec_len = [], []
        for text in batch:
            text_token_ids = self.t5_tokenizer.convert_tokens_to_ids(text)
            split_index = text_token_ids.index(self.t5_tokenizer.vocab[','])
            # 拆分上下句
            text1_ids = text_token_ids[:split_index]
            text2_ids = text_token_ids[split_index + 1:]
            if len(text1_ids) < 3 or len(text2_ids) < 3:
                continue
            enc1_ids = self.prompt1 + text1_ids
            enc1_ids.insert(0, 259)  # mt5_tokenizer 中 '▁': 259代表开始
            enc1_ids.append(self.t5_tokenizer.eos_token_id)  # 添加结束id

            dec1_ids = copy.deepcopy(text2_ids)
            dec1_ids.insert(0, self.t5_config.decoder_start_token_id)

            y1 = copy.deepcopy(text2_ids)
            y1.append(self.t5_tokenizer.eos_token_id)

            batch_enc_input.append(enc1_ids)
            batch_dec_input.append(dec1_ids)
            batch_enc_len.append(len(enc1_ids))
            batch_dec_len.append(len(dec1_ids))
            batch_label.append(y1)

            # 拆分上下联第一个字
            enc2_ids = text1_ids[:1] + text2_ids[:1]
            enc2_ids = self.prompt2 + enc2_ids
            enc2_ids.insert(0, 259)  # mt5_tokenizer 中 '▁': 259代表开始
            enc2_ids.append(self.t5_tokenizer.eos_token_id)  # 添加结束id

            dec2_ids = copy.deepcopy(text_token_ids)
            dec2_ids.insert(0, self.t5_config.decoder_start_token_id)

            y2 = copy.deepcopy(text_token_ids)
            y2.append(self.t5_tokenizer.eos_token_id)

            batch_enc_input.append(enc2_ids)
            batch_dec_input.append(dec2_ids)
            batch_enc_len.append(len(enc2_ids))
            batch_dec_len.append(len(dec2_ids))
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
        datas = self.data_processing()
        data_len = len(datas)
        train_len = int(data_len * 0.8)
        datas_clone = datas[:]
        random.shuffle(datas_clone)
        train_datas = datas_clone[: train_len]
        test_datas = datas_clone[train_len:]

        train_dataset = MyDataset(train_datas)
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

        return train_dataloader, test_dataloader, self.t5_config.vocab_size

    def data_processing(self) -> list[list]:
        new_datas = []
        with open(os.path.join(self.data_dir, 'poems_edge_split.txt'), 'r', encoding='utf-8') as reader:
            for line in reader.readlines():
                for text in line.strip()[1:-1].split('。'):
                    if text == '' or len(text.split('，')) != 2 or len(text.split('，')[0]) != len(text.split('，')[1]):
                        continue
                    new_datas.append(text)
        with open(os.path.join(self.data_dir, 'poems.txt'), 'w', encoding='utf-8') as writer:
            writer.write('\n'.join(new_datas))

        # 特殊字符处理
        final_datas = []
        for text in new_datas:
            final_datas.append(list(text.replace('，', ',')))
            for token in ['(', '2', ';', 'í', 'ó', 'ē', 'ī', '□', '、', 'Ｃ', 'ｒ', 'ｗ', '￣']:
                for i, e in enumerate(final_datas[-1]):
                    if e == token:
                        # 将一部分特殊字符替换为 unk
                        final_datas[-1][i] = self.t5_tokenizer.unk_token

        return final_datas
