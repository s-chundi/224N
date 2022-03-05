"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""
import math
import numpy as np
import layers
import torch
import torch.nn as nn
import util
import torch.nn.functional as F 
from torchinfo import summary
device, _ = util.get_available_devices()

class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hid_size=hidden_size)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
        
        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)
        c_emb = torch.transpose(c_emb, 1, 2)
        q_emb = torch.transpose(q_emb, 1, 2)
        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
    
class QANet(nn.Module):
    def __init__(self, word_vectors, char_vectors, hid_size, num_head=8):  # TODO: Trying 8 head attention, could go crazy.
        super().__init__()
        self.embedding = layers.Embedding(word_vectors, char_vectors, hid_size)
        self.encoder = layers.EncodingBlock(hid_size=hid_size, conv_num=4, num_head=num_head, kernel_size=7)
        self.jointattention = layers.BiDAFAttention(hid_size, 0.2)
        self.reshape = nn.Conv1d(hid_size*4, hid_size, kernel_size=1)
        self.stack = nn.ModuleList([layers.EncodingBlock(hid_size=hid_size, 
                                                conv_num=2, 
                                                num_head=num_head, 
                                                kernel_size=5) for _ in range(7)])
        self.out = layers.Output(hid_size)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = (torch.zeros_like(cw_idxs) != cw_idxs)
        q_mask = (torch.zeros_like(qw_idxs) != qw_idxs)
        context_emb = self.embedding(cc_idxs, cw_idxs)
        query_emb = self.embedding(qc_idxs, qw_idxs)
        context = self.encoder(context_emb, c_mask).transpose(1, 2)
        query = self.encoder(query_emb, q_mask).transpose(1, 2)
        x = self.jointattention(context, query, c_mask, q_mask).transpose(1, 2)
        x = self.reshape(x)
        x = F.dropout(x, p=0.1, training=self.training)
        for encoding_block in self.stack:
             x = encoding_block(x, c_mask)
        out1 = x
        for encoding_block in self.stack:
             x = encoding_block(x, c_mask)
        out2 = x
        x = F.dropout(x, p=0.1, training=self.training)
        for encoding_block in self.stack:
             x = encoding_block(x, c_mask)
        out3 = x
        start, end = self.out(out1, out2, out3, c_mask)
        return start, end

