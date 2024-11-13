import json

import torch
from transformers import AutoTokenizer, AutoModel


def test():
    t5 = AutoModel.from_pretrained(r'C:\Users\du\.cache\huggingface\hub\hub\t5-base-chinese')
    t5_tokenizer = AutoTokenizer.from_pretrained(
        r'C:\Users\du\.cache\huggingface\hub\hub\t5-base-chinese', legacy=False)
    text = '请根据给定的上联生成下联：碧林青旧竹'
    inputs = t5_tokenizer(text)
    print(inputs)
    # print(t5(
    #     input_ids=torch.tensor(inputs['input_ids'], dtype=torch.int64)[None, ...],
    #     decoder_input_ids=torch.tensor([0], dtype=torch.int64)[None, ...]
    # )['last_hidden_state'].shape)

    print(t5_tokenizer.eos_token)
    print(t5_tokenizer.pad_token)
    print(t5_tokenizer.unk_token)

    print(t5_tokenizer.decode([5252, 444, 2222, 5535]))

    # print(type(t5))
    from transformers.models.mt5.modeling_mt5 import MT5Model
    # print(t5.shared)
    #
    # vocab = dict(sorted(t5_tokenizer.vocab.items(), key=lambda item: item[1]))
    # with open(r'C:\Users\du\.cache\huggingface\hub\hub\t5-base-chinese\vocab.json', 'w', encoding='utf-8') as writer:
    #     json.dump(vocab, writer, ensure_ascii=False, indent=4)
    #
    # with open('../datas/token2idx.json', 'r', encoding='utf-8') as reader:
    #     token2idx = json.load(reader)
    #
    # unk_vocab = set()
    # for x in token2idx.keys():
    #     if x not in vocab.keys():
    #         unk_vocab.add(x)
    #
    # print(unk_vocab)
    # print(len(unk_vocab))


if __name__ == '__main__':
    test()
