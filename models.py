"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


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
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
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

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)
        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)
        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)
        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)
        return out

class QAnet(nn.Module):
    """
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(QAnet, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)                            

        # TODO: conv number should be 4 or something. Make kernel size a hyperparameter
        self.enc = layers.Embed_encoder_block(input_size=hidden_size,
                                            conv_num=2,
                                            channels=hidden_size,
                                            kern_size=7,
                                            drop_prob=0.1)

        self.att = layers.BiDAFAttention(hidden_size=hidden_size, drop_prob=drop_prob)


        # TODO: conv number should be 4 or something. Make kernel size a hyperparameter
        self.block1 = layers.Embed_encoder_block(input_size=4 * hidden_size,
                                     conv_num=2,
                                     channels= 4 * hidden_size,
                                     kern_size=7,
                                     drop_prob=drop_prob)

        self.block2 = layers.Embed_encoder_block(input_size=4 * hidden_size,
                                     conv_num=2,
                                     channels= 4 * hidden_size,
                                     kern_size=7,
                                     drop_prob=drop_prob)

        self.block3 = layers.Embed_encoder_block(input_size=4 * hidden_size,
                                     conv_num=2,
                                     channels= 4 * hidden_size,
                                     kern_size=7,
                                     drop_prob=drop_prob)

        self.out = layers.Output(dim=hidden_size, context_len=150)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         
        q_emb = self.emb(qw_idxs)
        
        c_enc = self.enc(c_emb, c_mask)   
        q_enc = self.enc(q_emb, q_mask)
        # Done until here  
        att = self.att(c_enc, q_enc, c_mask, q_mask)    
        out1 = self.block1(att, c_mask)
        out2 = self.block2(out1, c_mask)
        out3 = self.block3(out2, c_mask)

        out = self.out(out1, out2, out3, c_mask)  # 2 tensors, each (batch_size, c_len)
        # print('Output_sizes', out[0].size())
        return out
