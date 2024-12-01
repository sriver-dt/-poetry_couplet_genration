from os import PathLike
from typing import Union, Optional

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config


class GPT2BaseNet(nn.Module):
    def __init__(self, model_path: Union[str, PathLike]):
        super(GPT2BaseNet, self).__init__()
        self.config = GPT2Config.from_pretrained(model_path)
        self.gpt2 = GPT2Model.from_pretrained(model_path)
        self.output = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.BoolTensor] = None,
                use_cache: Optional[bool] = None,
                ):
        gpt2_output = self.gpt2(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache)
        hidden_state = gpt2_output['last_hidden_state']
        return self.output(hidden_state)
