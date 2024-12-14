from typing import Optional

import torch
from torch import nn


class CLIPTextEmbeddings(nn.Module):

    def __init__(
        self,
        token_size: int = 49408,
        hidden_size: int = 512,
        max_position_embedding: int = 77,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(token_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embedding, hidden_size)
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embedding).reshape((1, -1)),
            persistent=False,
        ) #[1, max_position_embedding]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # tokenized id of input sentences
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        sequence_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        if position_ids is None:
            position_ids = self.position_ids[:, :sequence_length] #[1, sequence_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids) # [batch, sequence_length, hidden_size]
        
        position_embedding = self.position_embedding(position_ids) # [1, sequence_length, hidden_size]
        embeddings = inputs_embeds + position_embedding # [batch, sequence_length, hidden_size]
        return embeddings


class CLIPSdpaAttention(nn.Module):

    def __init__(self, input_size: int=512, output_size: int=512, num_heads: int=8):
        super().__init__()
        self.k_proj = nn.Linear(input_size, output_size, bias=True)
        self.v_proj = nn.Linear(input_size, output_size, bias=True)
        self.q_proj = nn.Linear(input_size, output_size, bias=True)
        self.out_proj = nn.Linear(input_size, output_size, bias=True)
        self.head_dim = self.output_size // num_heads
        self.scale = self.head_dim **-0.5 # 1/sqrt(d_h)

    def forward(self,
        hidden_states: torch.Tensor,
    ):
        q_state = self.q_proj(hidden_states) * self.scale
        return hidden_states


def get_model():
    model = CLIPTextEmbeddings()
    return model
