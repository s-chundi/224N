"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax

# this is d from the paper
d = 128
'''
TODO: Add character embeddings
1.The characterembedding is obtained as follows: Each character is represented as a trainable vector of dimension
p2 = 200, meaning each word can be viewed as the concatenation of the embedding vectors for each
of its characters. The length of each word is either truncated or padded to 16. We take maximum
value of each row of this matrix to get a fixed-size vector representation of each word. Finally, the
output of a given word x from this layer is the concatenation [xw; xc] ∈ Rp1+p2
, where xw and xc are the word embedding and the convolution output of character embedding of x respectively.
Following Seo et al. (2016), we also adopt a two-layer highway network (Srivastava et al., 2015) on
top of this representation. For simplicity, we also use x to denote the output of this layer
'''
class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


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
'''
2. Embedding Encoder Layer. The encoder layer is a stack of the following basic building block:
[convolution-layer × # + self-attention-layer + feed-forward-layer], as illustrated in the upper right
of Figure 1. We use depthwise separable convolutions (Chollet, 2016) (Kaiser et al., 2017) rather
than traditional ones, as we observe that it is memory efficient and has better generalization. The
kernel size is 7, the number of filters is d = 128 and the number of conv layers within a block is
4. For the self-attention-layer, we adopt the multi-head attention mechanism defined in (Vaswani
et al., 2017a) which, for each position in the input, called the query, computes a weighted sum of all
positions, or keys, in the input based on the similarity between the query and key as measured by
the dot product. The number of heads is 8 throughout all the layers. Each of these basic operations
(conv/self-attention/ffn) is placed inside a residual block, shown lower-right in Figure 1. For an
input x and a given operation f, the output is f(layernorm(x)) +x, meaning there is a full identity
path from the input to output of each block, where layernorm indicates layer-normalization proposed
in (Ba et al., 2016). The total number of encoder blocks is 1. Note that the input of this layer is
a vector of dimension p1 + p2 = 500 for each individual word, which is immediately mapped to
d = 128 by a one-dimensional convolution. The output of this layer is a also of dimension d = 128.
'''
class Depthwise_conv(nn.Module):
    def __init__(self, in_chan, out_chan, kern_size):
        super().__init__()
        # Depth with groups = in_chan will convolve over each channel of the input. What type of padding should I use?
        self.depth = nn.Conv1d(in_channels=in_chan, out_channels=in_chan, kernel_size=kern_size, groups=in_chan, padding=3, bias=False)
        # Point with groups = 1 will convolve 1x1 across channels
        self.point = nn.Conv1d(in_channels=in_chan, out_channels=out_chan, kernel_size=1, padding=0, bias=True)
    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        x = F.relu(x)
        return x

class PositionalEncoding(nn.Module):
    '''
    from the paper "Attention is all you need"
    and the pytorch website: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    '''
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Embed_encoder_block(nn.Module):
    '''
    The encoder layer is a stack of the following basic building block:
    [convolution-layer × # + self-attention-layer + feed-forward-layer]
    TODO: I'm pretty sure channels is the same as input_size
    TODO: I think we need cmask here? looks like: return output * mask
    '''
    def __init__(self, input_size, conv_num, channels, kern_size, drop_prob=0):
        super().__init__()
        self.conv_num = conv_num
        self.input_size = input_size
        self.convs = [Depthwise_conv(input_size, input_size, kern_size) for i in range(conv_num)]
        # TODO: properly define input size, channels, etc.
        self.self_attention = nn.MultiheadAttention(input_size, num_heads=4, dropout=drop_prob)
        # TODO: padding = 3 should actually be kernel size // 2 right?
        self.feed_forward = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=kern_size, padding=3)
        self.norms = [nn.LayerNorm(input_size) for i in range(conv_num + 2)]
        self.dropout = nn.Dropout(drop_prob)
        self.positional = PositionalEncoding(input_size)

    def forward(self, x, mask):

        '''
        The blocks are (layer norm + conv)*conv_num + layer norm + self_attention + layer norm + Feed forward
        '''
        # Save the residual 
        x = x.permute(1, 0, 2)
        out = self.positional(x)
        out = out.permute(1, 0, 2)
        for i in range(self.conv_num):
            res = out
            x = self.norms[i](res)
            # dropout prob should be pretty small no?
            x = self.dropout(x)
            # Conv is (N, C, L)
            x = self.convs[i](x.transpose(1,2)).transpose(1,2)
            out = x + res
        out = self.norms[-2](out)
        out = out.permute(1, 0, 2)
        # why is pytorch so weird
        out, _ = self.self_attention(out, out, out, mask==False)
        out = out.permute(1, 0, 2)
        out = self.norms[-1](out)
        out = self.feed_forward(out.transpose(1,2)).transpose(1,2)
        out = F.relu(out)
        return out



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
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2).expand([-1, c_len, -1])
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
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
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

class Output(nn.Module):
    def __init__(self, dim, context_len = 500):
        '''
        dim is dimension of everything
        '''
        super().__init__()
        self.dim = dim
        self.l1 = nn.Linear(dim*8, 1)
        self.l2 = nn.Linear(dim*8, 1)
    
    def forward(self, output1, output2, output3, context_mask):
        start = torch.cat((output1, output2), 2)
        start = self.l1(start)
        end = torch.cat((output1, output3), 2)
        end = self.l2(end)
        # TODO see if we don't need squeeze()
        start = masked_softmax(start.squeeze(), context_mask, log_softmax= True)
        end = masked_softmax(end.squeeze(), context_mask, log_softmax= True)

        return start, end