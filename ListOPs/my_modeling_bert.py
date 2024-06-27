import transformers.models.bert.modeling_bert as transformers_modeling_bert
import torch

class BertForSequenceClassification(transformers_modeling_bert.BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.post_init()


class BertModel(transformers_modeling_bert.BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        # Initialize weights and apply final processing
        self.pooler = BertPooler(config)
        self.post_init()


class BertPooler(transformers_modeling_bert.BertPooler):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 1]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    

class BertEmbeddings(transformers_modeling_bert.BertEmbeddings):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        super().__init__(config)
        # self.position_embeddings =  SinusoidalPositional(config.hidden_size, max_seq_length=config.max_position_embeddings)
        # Sinusoidal nn.Embedding(config.max_position_embeddings, config.hidden_size)

    def forward(
        self,
        input_ids= None,
        position_ids= None,
        token_type_ids= None,
        inputs_embeds= None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:

        # if position_ids is None:
        #     raise Exception("position_ids is None")
            # if input_ids is not None:
            #     input_shape = input_ids.size()
            # else:
            #     input_shape = inputs_embeds.size()[:-1]
            # seq_length = input_shape[1]
            # position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds

        # We don't do positional embeddings anymore
        # for i in range(position_ids.shape[1]):
        #     embeddings += self.position_embeddings(position_ids=position_ids[:,i,:])

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

import torch
import torch.nn as nn
import math
import numpy as np


class SinusoidalPositional(nn.Module):
    """
    The original positional embedding used in 'Attention is all you need'
    """
    def __init__(self, emb_dim, max_seq_length=512):
        super(SinusoidalPositional, self).__init__()
        self.max_seq_length = max_seq_length
        self.emb_dim = emb_dim
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000) / emb_dim))
        pe = torch.zeros(max_seq_length, emb_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # return a 3D pe so it can be broadcasting on the batch_size dimension
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.register_buffer("div_term", div_term, persistent=False)

    def forward(self, input_ids=None, position_ids=None):
        r"""Inputs of forward function
        Args:
            input_ids: the sequence fed to the positional encoder model (required).
        Shape:
            input_ids: [batch size, sequence length]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        if position_ids is not None:
            pe = torch.zeros(position_ids.shape[0], position_ids.shape[1], self.emb_dim, device=position_ids.device)
            # pe[:, :, 0::2] = torch.sin(position_ids.unsqueeze(2) * self.div_term)
            # pe[:, :, 1::2] = torch.cos(position_ids.unsqueeze(2) * self.div_term)
            return pe
        else:
            return self.pe[:, : input_ids.shape[1], :]
