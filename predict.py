import json
import logging
import os.path

import torch

from net.custom_transformer import Transformer
from net.mt5_net import MT5BaseNet
from transformers import AutoConfig
from tokenizer.tokenizer import PegasusTokenizer


class CustomPredictor:
    def __init__(self, model_dir):
        with open(os.path.join(model_dir, 'vocab.json'), 'r', encoding='utf-8') as reader:
            self.token2idx = json.load(reader)
        self.idx2token = {v: k for k, v in self.token2idx.items()}

        # 模型参数恢复
        with open(os.path.join(model_dir, 'config.json'), 'r', encoding='utf-8') as reader:
            self.config = json.load(reader)
        self.config['is_training'] = False

        net_states = torch.load(os.path.join(model_dir, 'best.pkl'), map_location=torch.device('cpu'))['model_state']
        self.net = Transformer(**self.config)
        logging.info('正在进行模型参数恢复')
        missing_keys, unexpected_keys = self.net.load_state_dict(state_dict=net_states, strict=False)
        logging.info(f'未恢复的参数:{missing_keys}')
        logging.info(f'未解析的参数:{unexpected_keys}')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def predict(self, text: str):
        pred_tokens = []
        self.net.eval().to(device=self.device)
        enc_input_ids = torch.tensor(self.convert_token_to_id(text), dtype=torch.int64, device=self.device)[None, ...]
        dec_input_ids = [self.token2idx['<START>']]
        while True:
            output = self.net(input_ids=enc_input_ids,
                              decoder_input_ids=torch.tensor(
                                  dec_input_ids, dtype=torch.int64, device=self.device)[
                                  None, ...])  # [1, 1, vocab_size]
            pred_id = torch.argmax(torch.softmax(output[0][-1], dim=-1)).cpu().item()
            dec_input_ids.append(pred_id)
            pred_token = self.idx2token[pred_id]
            if pred_token == '<END>' or len(pred_tokens) >= 20:
                break
            pred_tokens.append(pred_token)
        return ''.join(pred_tokens)

    def convert_token_to_id(self, text):
        ids = [self.token2idx[token] if token in self.token2idx.keys() else self.token2idx['<UNK>'] for token in text]
        ids.insert(0, self.token2idx['<START>'])
        ids.append(self.token2idx['<END>'])
        return ids


class T5Predictor:
    def __init__(self, t5_model_path, model_dir):
        self.t5_tokenizer = PegasusTokenizer.from_pretrained(t5_model_path, legacy=False)
        self.t5_config = AutoConfig.from_pretrained(t5_model_path)

        # 模型参数恢复
        net_states = torch.load(os.path.join(model_dir, 'best.pkl'), map_location=torch.device('cpu'))['model_state']
        self.net = MT5BaseNet(model_path=t5_model_path)
        logging.info('正在进行模型参数恢复')
        missing_keys, unexpected_keys = self.net.load_state_dict(state_dict=net_states, strict=False)
        logging.info(f'未恢复的参数:{missing_keys}')
        logging.info(f'未解析的参数:{unexpected_keys}')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def predict(self, text: str):
        pred_tokens = []
        self.net.eval().to(device=self.device)
        input_ids = self.t5_tokenizer.convert_tokens_to_ids(list(text))
        input_ids.append(self.t5_tokenizer.vocab[self.t5_tokenizer.eos_token])
        dec_input_ids = [self.t5_config.decoder_start_token_id]
        while True:
            output = self.net(
                input_ids=torch.tensor(input_ids, dtype=torch.int64, device=self.device)[None, ...],
                decoder_input_ids=torch.tensor(dec_input_ids, dtype=torch.int64, device=self.device)[None, ...]
            )
            pred_id = torch.argmax(torch.softmax(output[0][-1], dim=-1)).cpu().item()
            dec_input_ids.append(pred_id)
            pred_token = self.t5_tokenizer.convert_ids_to_tokens(pred_id)
            if pred_token == self.t5_tokenizer.eos_token or len(pred_tokens) >= 20:
                break
            pred_tokens.append(pred_token)
        return ''.join(pred_tokens)


if __name__ == '__main__':
    predictor = CustomPredictor('./output/custom_transformer')
    print('CustomPredictor: ')
    print('请根据给定的上联生成下联：才子乘春来骋望')
    print('下联：', predictor.predict('请根据给定的上联生成下联：才子乘春来骋望'))
    print('-' * 50)
    print('请根据给定的上下联第一个字生成完整对联：风云')
    print('上下联：', predictor.predict('请根据给定的上下联第一个字生成完整对联：风云'))

    # t5predictor = T5Predictor(r'C:\Users\du\.cache\huggingface\hub\hub\t5-pegasus-small', './output/t5-pegasus')
    # print('MT5: ')
    # print('请根据给定的上联生成下联:才子乘春来骋望')
    # print('下联：', t5predictor.predict('请根据给定的上联生成下联:才子乘春来骋望'))
    # print('-' * 50)
    # print('请根据给定的上下联第一个字生成完整对联:风云')
    # print('上下联：', t5predictor.predict('请根据给定的上下联第一个字生成完整对联:风云'))
