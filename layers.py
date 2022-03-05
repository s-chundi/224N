"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import util
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
import args
from torch.autograd import Variable
import math

hid_size = 128
device, _ = util.get_available_devices()

def position_encodings(x):
    # Inputs are [N, E, S]
    sqlen, hid_size = x.size(2), x.size(1)
    x = x.transpose(1, 2)
    # Now N S E
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    pos = torch.arange(sqlen, dtype=torch.float32)
    incr = math.log(10000.0) / (hid_size / 2. - 1)
    inds = torch.exp(torch.arange(hid_size / 2, dtype=torch.float32) * -incr)
    # creates a 64 by 150 matrix
    inds = pos.unsqueeze(1) * inds.unsqueeze(0)
    signal = torch.cat([torch.sin(inds), torch.cos(inds)], dim = 1)
    signal = signal.view(1, sqlen, hid_size)
    return (x + signal.to(device)).transpose(1, 2)


class Depth_Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depth = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
        self.point = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        x = F.relu(x)
        return x

class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = util.masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = util.masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s

class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        x = x.transpose(1, 2)
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x.transpose(1, 2)


class Embedding(nn.Module):
    def __init__(self, word_vectors, char_vectors, hid_size, k=5):
        super().__init__()
        # This only does convolution over each character
        # TODO: could we just use reshape + conv1d?
        self.conv2d = nn.Conv2d(64, hid_size, kernel_size = (1,k))
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = nn.Conv1d(300 + hid_size, hid_size, kernel_size=1)
        nn.init.kaiming_normal_(self.conv1d.weight, nonlinearity='relu')
        self.char_emb = nn.Embedding.from_pretrained(char_vectors)
        self.word_emb = nn.Embedding.from_pretrained(word_vectors)
        self.hwy = HighwayEncoder(2, hid_size)

    def forward(self, char_vectors, word_vectors):
        char_embed = self.char_emb(char_vectors)
        word_embed = self.word_emb(word_vectors)
        char_embed = char_embed.permute(0, 3, 1, 2)
        # Conv2d accepts shapes in N C H W
        char_embed = self.conv2d(char_embed)
        char_embed = F.relu(char_embed)
        # TODO: try mean instead?
        char_embed, _ = char_embed.max(dim=3)
        word_embed = F.dropout(word_embed, p=0.1, training=self.training)
        char_embed = F.dropout(char_embed, p=0.05, training=self.training)
        word_embed = word_embed.transpose(1, 2)
        embed = torch.cat([char_embed, word_embed], dim=1)
        embed = self.conv1d(embed)
        embed = self.hwy(embed)
        return embed

class Unit(nn.Module):
    def __init__(self, hid_size, kernel, drop_prob=0.05):
        super().__init__()
        self.conv = Depth_Conv(hid_size, hid_size, kernel)
        self.lnorm = nn.LayerNorm(hid_size)
        self.drop_prob = drop_prob
    def forward(self, x, use_conv=True):
        # Expects inputs in N S E Does it though?
        x = self.lnorm(x.transpose(1, 2)).transpose(1, 2)
        # Expects inputs in N E S Should we normalize over S?
        if use_conv:
            x = self.conv(x)
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        return x

class EncodingBlock(nn.Module):
    def __init__(self, hid_size, conv_num, num_head, kernel_size):
        super().__init__()
        self.units = nn.ModuleList([Unit(hid_size, kernel_size) for _ in range(conv_num)])
        self.selfattention = nn.MultiheadAttention(hid_size, num_head, 0.1)
        self.layernorm_1 = nn.LayerNorm(hid_size)
        self.layernorm_2 = nn.LayerNorm(hid_size)
        self.lin = nn.Linear(hid_size, hid_size)

    def forward(self, x, mask, dropout=0.1):
        x = position_encodings(x)
        for conv in self.units:
            res = x
            x = conv(x)
            if self.training == True and np.random.rand() < dropout:
                x = res
            else:
                x = x + res
        res = x
        x = self.layernorm_1(x.transpose(1,2)).transpose(1,2)
        x = F.dropout(x, p=dropout, training=self.training)
        x = x.permute(2, 0, 1)
        mask = ~mask
        x, _ = self.selfattention(x, x, x, key_padding_mask=mask)
        x = x.permute(1, 2, 0)
        x = F.dropout(x, p=0.1, training=self.training)
        if self.training == True and np.random.rand() < dropout:
            x = res
        else:
            x = x + res
        res = x

        x = self.layernorm_2(x.transpose(1,2)).transpose(1,2)
        x = F.dropout(x, p=dropout, training=self.training)
        # TODO: Is more nonlinearity helpful here?
        x = self.lin(x.transpose(1,2)).transpose(1,2)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        if self.training == True and np.random.rand() < dropout:
            x = res
        else:
            x = x + res
        return x


class Output(nn.Module):
    def __init__(self, hid_size, linear=False):
        super().__init__()
        self.linear = linear
        if linear:
            self.feedforward1 = nn.Linear(hid_size*2, 1)
            self.feedforward2 = nn.Linear(hid_size*2, 1)
        else:
            self.feedforward1 = nn.Conv1d(hid_size*2, 1, kernel_size=1, bias=False)
            self.feedforward2 = nn.Conv1d(hid_size*2, 1, kernel_size=1, bias=False)
        

    def forward(self, M1, M2, M3, mask):
        if self.linear:
            start = torch.cat([M1, M2], dim=1).transpose(1,2)
            end = torch.cat([M1, M3], dim=1).transpose(1,2)
        else:
            start = torch.cat([M1, M2], dim=1)
            end = torch.cat([M1, M3], dim=1)
        start = self.feedforward1(start)
        end = self.feedforward2(end)
        start = util.masked_softmax(start.squeeze(), mask, log_softmax=True)
        end = util.masked_softmax(end.squeeze(), mask, log_softmax=True)
        return start, end


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)
        

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        self.rnn.flatten_parameters()
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x



class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob, x=8):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(x * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)
        self.att_linear_2 = nn.Linear(x * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2