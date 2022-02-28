"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

from numpy import char
import torch
import torch.nn as nn
import torch.nn.functional as F
import util
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
import args
from torch.autograd import Variable
import math

# TODO: More descriptive names for EVERYTHING. Make these args somehow
hid_size = 128


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
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


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
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

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

'''

NEW CODE AAAAAAAAAAAAAAHHHHHHHHHH

'''
# Done!
class Depthwise_conv(nn.Module):
    def __init__(self, in_chan, out_chan, kern_size):
        super().__init__()
        # Depth with groups = in_chan will convolve over each channel of the input. What type of padding should I use? 
        # TODO: Remove hardcoded 2
        self.depth = nn.Conv1d(in_channels=in_chan, out_channels=in_chan, kernel_size=kern_size, groups=in_chan, padding=kern_size//2, bias=False)
        # Point with groups = 1 will convolve 1x1 across channels
        self.point = nn.Conv1d(in_channels=in_chan, out_channels=out_chan, kernel_size=1, padding=0, bias=True)
    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        x = F.relu(x)
        return x
# Done!
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class SelfAttention(nn.Module):
    def __init__(self, num_heads=2, hid_size=128):
        super().__init__()
        self.multatt = nn.MultiheadAttention(hid_size, num_heads=num_heads, dropout=0.05)
        # These could be removed for some reason
        self.conv1 = nn.Conv1d(hid_size, hid_size, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hid_size, hid_size, kernel_size=5, padding=2)
    def forward(self, query, mask):
        # Everything needs to be passed in (Sl, N, E)
        # Self attention has the same key, query, value
        values = query.permute(2, 0, 1)
        key = self.conv1(query).permute(2, 0, 1)
        query = self.conv2(query).permute(2, 0, 1)
        mask = ~mask
        # TODO: Implement multihead dot product attention with convs, it could be much faster
        output, _ = self.multatt(query, key, values, key_padding_mask=mask)
        # Back to N Sl E
        output = output.permute(1, 0, 2)
        return output

# Done!
class Embedding(nn.Module):
    def __init__(self, word_vectors, char_vectors, hid_size):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.chars = nn.Embedding.from_pretrained(char_vectors)
        self.embed_dim = self.chars.embedding_dim
        self.conv = nn.Conv1d(char_vectors.size(1)*16, 200, kernel_size=5, padding=2)
        self.proj = nn.Conv1d(200+word_vectors.size(1), hid_size, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.hwy = HighwayEncoder(2, hid_size)

    def forward(self, x, y):
        emb = self.embed(x)   
        chars = self.chars(y)
        chars = chars.reshape(chars.size(0), chars.size(1), chars.size(2)*self.embed_dim)
        chars = chars.permute(0, 2, 1)
        chars = self.conv(chars)
        chars = chars.permute(0, 2, 1)
        emb = torch.cat((emb, chars), dim=2)
        emb = F.dropout(emb, 0.1, self.training)
        emb = self.proj(emb.permute(0, 2, 1)).permute(0, 2, 1) 
        emb = self.hwy(emb)   
        emb = emb.permute(0, 2, 1)
        return emb
# Done! 
class Unit(nn.Module):
    def __init__(self, hid_size, kernel, drop_prob):
        super().__init__()
        self.conv = Depthwise_conv(hid_size, hid_size, kernel)
        self.lnorm = nn.LayerNorm(hid_size)
        self.drop_prob = drop_prob
    def forward(self, x, use_conv=True):
        # Expects inputs in N S E
        x = self.lnorm(x.permute(0, 2, 1)).permute(0, 2, 1)
        # Expects inputs in N E S Should we normalize over S?
        if use_conv:
            x = self.conv(x)
        x = F.dropout(x, p=self.drop_prob, training=self.training)
        return x

# Done!
class EmbedEncoderBlock(nn.Module):
    def __init__(self, conv_num, hid_size, kernel, num_heads=2):
        super().__init__()
        self.units = nn.ModuleList([Unit(hid_size, kernel=kernel, drop_prob=0.1) for _ in range(conv_num)])
        self.unit1 = Unit(hid_size, kernel=kernel, drop_prob=0.1)
        self.selfatt = SelfAttention(num_heads=num_heads, hid_size=hid_size)
        self.unit2 = Unit(hid_size, kernel=kernel, drop_prob=0.1)
        self.feed_forward = nn.Conv1d(hid_size, hid_size, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # TODO: determine which initialization works best. Also cite this
        # self.pos_emb = nn.Parameter(torch.randn(hid_size, 400))
        self.pos_emb = PositionalEncoding(hid_size, dropout=0.1, max_len=500)

    def forward(self, x, mask):
        # TODO is mask needed for this part? Can we make these better?
        out = self.pos_emb(x.transpose(1, 2)).transpose(1, 2)
        for unit in self.units:
            res = out
            out = unit(out)
            out = res + out
        res = out
        out = self.unit1(out, use_conv=False)
        out = self.selfatt(out, mask)
        out = out.permute(0, 2, 1)
        out = res + out
        res = out

        out = self.unit2(out, use_conv=False)
        out = self.feed_forward(out)
        out = self.relu1(out)
        out = res + out
        return out

# Done!
class Output(nn.Module):
    def __init__(self, hid_size=128):
        super().__init__()
        # TODO: make these linear have to do some transposes etc.
        # TODO: test linear vs transposes
        
        # This was for convolution
        self.conv1 = nn.Conv1d(hid_size*2, 1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hid_size*2, 1, kernel_size=3, padding=1)
        
        # This is for linear
        self.lin1 = nn.Linear(hid_size*2, 1)
        self.lin2 = nn.Linear(hid_size*2, 1)

    def forward(self, M1, M2, M3, mask):
        bs, hid_size, sq_len = M1.size()
        start = torch.cat([M1, M2], dim=1)
        end = torch.cat([M1, M3], dim=1)
        use_conv = False
        if use_conv:
            start = self.conv1(start).squeeze(dim=1)
            end = self.conv2(end).squeeze(dim=1)
        else:
            start = self.lin1(start.permute(0, 2, 1)).squeeze(dim=-1)
            end = self.lin2(end.permute(0, 2, 1)).squeeze(dim=-1)

        start = util.masked_softmax(start, mask, log_softmax=True)
        end = util.masked_softmax(end, mask, log_softmax=True)
        return start, end