"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
import torch.nn.functional as F

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
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

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

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class QANet(nn.Module):
    def __init__(self, word_vectors, char_vectors, hid_size):
        super().__init__()
        # TODO remove hardcoded constants, and get the embed sizes from above
        self.emb = layers.Embedding(word_vectors, char_vectors, hid_size=hid_size)
        self.encoder = layers.EmbedEncoderBlock(conv_num=4, hid_size=hid_size, kernel=7)
        self.biatt = layers.BiDAFAttention(hid_size)
        # TODO: possibly remove this conv and make a huge set of encoder blocks
        self.conv = nn.Conv1d(hid_size*4, hid_size, kernel_size=1)
        # To mimic original QANet, set this to 1 possibly
        self.stack_size = 7
        # Should I make these 3 separate stacks?
        self.stack = nn.ModuleList([layers.EmbedEncoderBlock(conv_num=2, hid_size=hid_size, kernel=5) for _ in range(self.stack_size)])
        self.out = layers.Output()

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
        c_len = torch.max(c_len)
        context = self.emb(cw_idxs, cc_idxs)
        query = self.emb(qw_idxs, qc_idxs)
        contextenc = self.encoder(context, c_mask)
        queryenc = self.encoder(query, q_mask)
        contextenc = torch.transpose(contextenc, 1, 2)
        queryenc = torch.transpose(queryenc, 1, 2)
        out = self.biatt(contextenc, queryenc, c_mask, q_mask)
        out = self.conv(out.transpose(1, 2))
        out = F.dropout(out, p=0.1, training=self.training)
        for encoder in self.stack:
             out = encoder(out, c_mask)
        out1 = out
        for encoder in self.stack:
             out = encoder(out, c_mask)
        out2 = out
        out = F.dropout(out, p=0.1, training=self.training)
        for encoder in self.stack:
             out = encoder(out, c_mask)
        out3 = out
        p1, p2 = self.out(out1, out2, out3, c_mask)
        return p1, p2