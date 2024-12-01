import os.path
import random

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2Tokenizer


class MyDataset(Dataset):
    def __init__(self, datas):
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        return self.datas[item]


class GPT2Dataloader:
    def __init__(self, batch_size, data_dir, gpt2_model_dir):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_dir, legacy=False)
        self.gpt2_config = GPT2Config.from_pretrained(gpt2_model_dir)
        self.prompt1 = '请根据给定的上联生成下联'
        self.prompt2 = '请根据给定的上下联第一个字生成完整对联'

    def collate_fn(self, batch):
        batch_dec_input, batch_label = [], []
        batch_dec_len = []
        for text in batch:
            # 拆分上下句
            text1, text2 = text.split('，')
            if len(text1) != len(text2) or len(text1) not in [5, 7]:
                continue
            dec1_text = f'{self.prompt1}，上联：{text1} 下联：{text2}'
            dec1_ids = self.gpt2_tokenizer(dec1_text)['input_ids']

            y1 = dec1_ids[1:] + [self.gpt2_tokenizer.eos_token_id]

            batch_dec_input.append(dec1_ids)
            batch_dec_len.append(len(dec1_ids))
            batch_label.append(y1)

            # 拆分上下联第一个字
            dec2_text = text1[0] + text2[0]
            dec2_text = f'{self.prompt2}，上下联第一个字：{dec2_text} 上下联：{text}'
            dec2_ids = self.gpt2_tokenizer(dec2_text)['input_ids']

            y2 = dec2_ids[1:] + [self.gpt2_tokenizer.eos_token_id]

            batch_dec_input.append(dec2_ids)
            batch_dec_len.append(len(dec2_ids))
            batch_label.append(y2)

        # 构建mask
        dec_max_len = max(batch_dec_len)
        batch_size = len(batch_dec_input)
        dec_mask = torch.zeros(size=(batch_size, dec_max_len))
        pad_token_id = self.gpt2_tokenizer.eos_token_id
        for i in range(len(batch_dec_len)):
            dec_mask[i, :batch_dec_len[i]] = 1

            dec_pad_len = dec_max_len - batch_dec_len[i]
            batch_dec_input[i].extend([pad_token_id] * dec_pad_len)
            batch_label[i].extend([pad_token_id] * dec_pad_len)

        batch_dec_input = torch.tensor(batch_dec_input, dtype=torch.int64)
        batch_label = torch.tensor(batch_label, dtype=torch.int64)
        return (batch_dec_input, dec_mask.to(dtype=torch.float32), None, None), batch_label

    def get_dataloader(self, num_workers=0):
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
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            prefetch_factor=num_workers * self.batch_size
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            prefetch_factor=num_workers * self.batch_size
        )

        return train_dataloader, test_dataloader, self.gpt2_tokenizer.vocab_size

    def data_processing(self) -> list[str]:
        new_datas = []
        with open(os.path.join(self.data_dir, 'poems_edge_split.txt'), 'r', encoding='utf-8') as reader:
            for line in reader.readlines():
                for text in line.strip()[1:-1].split('。'):
                    if text == '' or len(text.split('，')) != 2 or len(text.split('，')[0]) != len(text.split('，')[1]):
                        continue
                    new_datas.append(text)

        return new_datas
