from os import PathLike
from typing import Union, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class MT5BaseNet(nn.Module):
    def __init__(self, model_path: Union[str, PathLike]):
        super(MT5BaseNet, self).__init__()
        self.mt5_model = AutoModel.from_pretrained(model_path)
        self.config = AutoConfig.from_pretrained(model_path)
        self.output = nn.Linear(self.config.d_model, self.config.vocab_size)

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.BoolTensor] = None,
                use_cache: Optional[bool] = None,
                ):
        mt5_output = self.mt5_model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, use_cache)
        hidden_state = mt5_output['last_hidden_state']
        return self.output(hidden_state)
