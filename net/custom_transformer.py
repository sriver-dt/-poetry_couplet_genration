import math
from typing import Optional

import torch
import torch.nn as nn


class PositionEmbedding(nn.Module):
    """
    三角函数绝对位置编码
    """

    def __init__(self, hidden_size: int, max_len: int = 128):
        super(PositionEmbedding, self).__init__()
        self.pe = torch.zeros(size=(1, max_len, hidden_size))
        x = (torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
             / torch.pow(10000, torch.arange(0, hidden_size, 2, dtype=torch.float32).reshape(1, -1) / hidden_size))
        self.pe[:, :, 0::2] = torch.sin(x)
        self.pe[:, :, 1::2] = torch.cos(x)

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[-1]
        return self.pe[:, :seq_len]


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力
    """

    def __init__(self, dropout: float):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor):
        attention_head_size = query.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1)) / torch.tensor(math.sqrt(attention_head_size))
        attention_mask = (1 - attention_mask) * torch.finfo(torch.float32).min
        attention_probs = nn.functional.softmax(attention_score + attention_mask, dim=-1)
        attention_output = torch.matmul(self.dropout(attention_probs), value)
        return attention_output


class MutilHeadAttention(nn.Module):
    """
    多头注意力
    """

    def __init__(self, hidden_size: int, num_attention_heads: int, dropout: float, is_decoder: bool = False):
        super(MutilHeadAttention, self).__init__()
        self.is_decoder = is_decoder
        self.attention = ScaledDotProductAttention(dropout)
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        assert hidden_size % num_attention_heads == 0, 'hidden_size 不能被 num_attention_heads 整除'
        self.attention_head_size = hidden_size // num_attention_heads
        self.q = nn.Linear(hidden_size, self.num_attention_heads * self.attention_head_size)
        self.k = nn.Linear(hidden_size, self.num_attention_heads * self.attention_head_size)
        self.v = nn.Linear(hidden_size, self.num_attention_heads * self.attention_head_size)
        self.o = nn.Linear(self.num_attention_heads * self.attention_head_size, hidden_size)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class MultiHeadSelfAttention(MutilHeadAttention):
    """
    多头自注意力
    """

    def __init__(self, hidden_size: int, num_attention_heads: int, dropout: float, is_decoder: bool = False,
                 is_training: bool = False):
        super(MultiHeadSelfAttention, self).__init__(hidden_size, num_attention_heads, dropout, is_decoder)
        self.is_training = is_training

    def forward(self, hidden_states: torch.Tensor = None, attention_mask: torch.Tensor = None,
                position_bias: torch.Tensor = None):
        """
        :param hidden_states:
        :param attention_mask: 当为 encoder 时，传入 encode_mask, 当为 decoder 时应为 None
        :param position_bias: 当为 encoder 时为 None, 当是 decoder 时传入 decoder_mask
        :return:
        """
        batch_size, seq_len, emb_size = hidden_states.shape
        query = self.q(hidden_states).reshape(
            batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = self.k(hidden_states).reshape(
            batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = self.v(hidden_states).reshape(
            batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        query = query.reshape(batch_size * self.num_attention_heads, seq_len, -1)
        key = key.reshape(batch_size * self.num_attention_heads, seq_len, -1)
        value = value.reshape(batch_size * self.num_attention_heads, seq_len, -1)
        if self.is_decoder:
            # decoder 的 attention_mask 构建
            if position_bias is None:
                position_bias = torch.ones(size=(batch_size, seq_len), dtype=torch.float32, device=hidden_states.device)
            if attention_mask is None and self.is_training:
                attention_mask = torch.tril(
                    torch.ones(size=(seq_len, seq_len), dtype=torch.float32, device=hidden_states.device))
            elif attention_mask is None:
                attention_mask = torch.ones(size=(seq_len, seq_len), dtype=torch.float32, device=hidden_states.device)

            position_bias = position_bias[:, None, None, :].expand(-1, self.num_attention_heads, -1, -1).reshape(
                batch_size * self.num_attention_heads, 1, -1
            )
            attention_mask = attention_mask[None, :, :].expand(batch_size * self.num_attention_heads, -1, -1)
            # 将decoder_input_mask和attention_mask合并
            attention_mask = attention_mask * position_bias
        else:
            # encoder 的 attention_mask 构建
            if attention_mask is None:
                attention_mask = torch.ones(size=(batch_size, seq_len), dtype=torch.float32,
                                            device=hidden_states.device)
            attention_mask = attention_mask[:, None, None, :].expand(-1, self.num_attention_heads, -1, -1).reshape(
                batch_size * self.num_attention_heads, 1, -1
            )

        attention_output = self.attention(query, key, value, attention_mask)
        output = attention_output.reshape(
            batch_size, self.num_attention_heads, seq_len, self.attention_head_size
        ).transpose(1, 2).reshape(batch_size, seq_len, self.num_attention_heads * self.attention_head_size)
        return self.o(output)


class EncoderDecoderCrossAttention(MutilHeadAttention):
    def __init__(self, hidden_size: int, num_attention_heads: int, dropout: float):
        super(EncoderDecoderCrossAttention, self).__init__(hidden_size, num_attention_heads, dropout)

    def forward(self, encoder_hidden_states: torch.Tensor,
                attention_mask: torch.Tensor = None,
                decoder_states: torch.Tensor = None
                ):
        """
        :param encoder_hidden_states: 编码器的隐层状态
        :param attention_mask: encoder_input_mask
        :param decoder_states: 解码器self_attention的输出
        :return:
        """
        batch_size, seq_len, hidden_size = encoder_hidden_states.shape
        query = self.q(decoder_states).reshape(
            batch_size, -1, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)
        key = self.k(encoder_hidden_states).reshape(
            batch_size, seq_len, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)
        value = self.v(encoder_hidden_states).reshape(
            batch_size, seq_len, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)

        query = query.reshape(batch_size * self.num_attention_heads, -1, self.attention_head_size)
        key = key.reshape(batch_size * self.num_attention_heads, seq_len, -1)
        value = value.reshape(batch_size * self.num_attention_heads, seq_len, -1)
        if attention_mask is None:
            attention_mask = torch.ones(size=(batch_size, seq_len), dtype=torch.float32,
                                        device=encoder_hidden_states.device)
        attention_mask = attention_mask[:, None, None, :].expand(-1, self.num_attention_heads, -1, -1).reshape(
            batch_size * self.num_attention_heads, 1, -1
        )
        attention_output = self.attention(query, key, value, attention_mask)
        output = attention_output.reshape(
            batch_size, self.num_attention_heads, -1, self.attention_head_size
        ).transpose(1, 2).reshape(batch_size, -1, self.num_attention_heads * self.attention_head_size)
        return self.o(output)


class FFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout):
        super(FFN, self).__init__()
        self.wi = nn.Linear(hidden_size, intermediate_size)
        self.wo = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.act(self.wi(x))
        x = self.dropout(x)
        x = self.wo(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, dropout: float, is_decoder=False):
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiHeadSelfAttention(hidden_size, num_attention_heads, dropout, is_decoder=is_decoder)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = FFN(hidden_size, hidden_size * 4, dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(self, hidden_states: torch.Tensor = None, attention_mask: torch.Tensor = None):
        attention_output = self.self_attention(hidden_states=hidden_states, attention_mask=attention_mask)
        output = self.layer_norm1(hidden_states + self.dropout1(attention_output))  # 残差
        output = self.layer_norm2(output + self.dropout2(self.ffn(output)))
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, dropout: float, num_layers: int):
        super(TransformerEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderBlock(hidden_size, num_attention_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor = None):
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, mask)

        return hidden_states


class DecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, dropout: float, is_decoder=True, is_training=False):
        super(DecoderBlock, self).__init__()
        self.masked_self_attention = MultiHeadSelfAttention(
            hidden_size, num_attention_heads, dropout, is_decoder=is_decoder, is_training=is_training
        )
        self.cross_attention = EncoderDecoderCrossAttention(hidden_size, num_attention_heads, dropout)
        self.ffn = FFN(hidden_size, hidden_size * 4, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.layer_norm3 = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(self, encoder_hidden_states: torch.Tensor, decoder_states: torch.Tensor,
                encoder_mask: torch.Tensor = None, decoder_mask: torch.Tensor = None):
        decoder_states = decoder_states + self.dropout1(self.masked_self_attention(
            hidden_states=decoder_states, position_bias=decoder_mask
        ))
        decoder_states = self.layer_norm1(decoder_states)
        decoder_states = decoder_states + self.dropout2(self.cross_attention(
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_mask,
            decoder_states=decoder_states
        ))
        decoder_states = self.layer_norm2(decoder_states)
        decoder_states = self.layer_norm3(decoder_states + self.dropout3(self.ffn(decoder_states)))
        return decoder_states


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, dropout: float, num_layers: int, is_decoder=True,
                 is_training=False):
        super(TransformerDecoder, self).__init__()
        self.decoder_layers = nn.ModuleList(
            [DecoderBlock(hidden_size, num_attention_heads, dropout, is_decoder, is_training) for _ in
             range(num_layers)]
        )

    def forward(self, encoder_hidden_states: torch.Tensor, decoder_states: torch.Tensor,
                encoder_mask: torch.Tensor = None, decoder_mask: torch.Tensor = None
                ):
        for layer in self.decoder_layers:
            decoder_states = layer(encoder_hidden_states, decoder_states, encoder_mask, decoder_mask)

        return decoder_states


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_attention_heads: int, num_layers: int,
                 dropout: float = 0.2, is_training: bool = False):
        super(Transformer, self).__init__()
        self.word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.position_embedding = PositionEmbedding(hidden_size=hidden_size)
        self.encoder = TransformerEncoder(hidden_size, num_attention_heads, dropout, num_layers)
        self.decoder = TransformerDecoder(hidden_size, num_attention_heads, dropout, num_layers,
                                          is_training=is_training)
        # self.linear = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.BoolTensor] = None,
                ):
        encoder_emb = self.word_embedding(input_ids)  # [batch_size, seq_len, hidden_size]
        encoder_position_emb = self.position_embedding(input_ids).to(
            device=encoder_emb.device)  # [1, seq_len, hidden_size]
        encoder_emb += encoder_position_emb

        decoder_emb = self.word_embedding(decoder_input_ids)
        decoder_position_emb = self.position_embedding(decoder_input_ids).to(device=encoder_emb.device)
        decoder_emb += decoder_position_emb

        encoder_states = self.encoder(encoder_emb, attention_mask)
        decoder_states = self.decoder(encoder_states, decoder_emb, attention_mask, decoder_attention_mask)
        # hidden_states = self.linear(decoder_states)
        return self.output(decoder_states)


def test():
    vocab_size = 1000
    hidden_size = 768
    num_attention_heads = 8
    dropout = 0.3
    num_layers = 6
    is_training = False
    transformer_model = Transformer(
        vocab_size,
        hidden_size,
        num_attention_heads,
        num_layers,
        dropout,
        is_training,
    )

    input_ids = torch.randint(0, 1000, size=(4, 10))
    mask = torch.ones(size=(4, 10))

    decoder_input_ids = torch.randint(0, 1000, size=(4, 5))
    decoder_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ]
    )

    result = transformer_model(input_ids, decoder_input_ids, mask, decoder_mask)
    print(f'transform: {result.shape}')


if __name__ == '__main__':
    test()
